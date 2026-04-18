use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::distributions::{MvDistribution, GaussianMvDistribution};


pub trait ParameterSet {
    fn get_parameters(&self) -> DVector<f64>;
    fn set_parameters(&mut self, params: &DVector<f64>);
    fn get_num_parameters(&self) -> usize;
}


pub trait StateSpaceModel<T: MvDistribution, S: MvDistribution> {
    //T: observed state distribution, S: initial state distribution
    
    fn get_parameters_as_vector(&self) -> DVector<f64>;

    fn set_parameters_as_vector(&mut self, params: &DVector<f64>);

    fn get_num_parameters(&self) -> usize;

    fn forecast(&self, observations: &Vec<DMatrix<f64>>, forecast_steps: &usize) -> Vec<T>;

    fn filter_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<T>;

    fn smooth_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<T>;

    fn sample(&self, num_observations: &usize, initial_state: Option<S>, seed: Option<u64>) -> (Vec<DVector<f64>>, Vec<DVector<f64>>);
}

pub trait DifferentiableOnce {
    fn get_gradient(&self) -> DVector<f64>;
}

pub trait DifferentiableTwice {
    fn get_hessian(&self) -> DMatrix<f64>;
}

pub struct LowerTriangularMatrix {
    size: usize,
    diagonal: DVector<f64>,
    lower_elements: DVector<f64>,
}

impl LowerTriangularMatrix {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            diagonal: DVector::from_element(size, 1.0),
            lower_elements: DVector::zeros(size * (size - 1) / 2),
        }
    }

    pub fn new_with_values(size: usize, diagonal_value: f64, lower_value: f64) -> Self {
        Self {
            size,
            diagonal: DVector::from_element(size, diagonal_value),
            lower_elements: DVector::from_element(size * (size - 1) / 2, lower_value),
        }
    }

    pub fn to_dense(&self) -> DMatrix<f64> {
        let mut mat = DMatrix::zeros(self.size, self.size);
        let mut idx = 0;

        for i in 0..self.size {
            mat[(i, i)] = self.diagonal[i];
            for j in 0..i {
                mat[(i, j)] = self.lower_elements[idx];
                idx += 1;
            }

        }

        return mat;
    }

    pub fn get_diagonal(&self) -> DVector<f64> {
        return self.diagonal.clone();
    }

    pub fn set_diagonal(&mut self, diagonal: DVector<f64>) {
        self.diagonal = diagonal;
    }

    pub fn get_lower_elements(&self) -> DVector<f64> {
        return self.lower_elements.clone();
    }

    pub fn set_lower_elements(&mut self, lower_elements: DVector<f64>) {
        self.lower_elements = lower_elements;
    }

    pub fn get_size(&self) -> usize {
        return self.size;
    }

    pub fn get_num_parameters(&self) -> usize {
        return self.size + self.size * (self.size - 1) / 2;
    }

    pub fn get_parameters_as_vector(&self) -> DVector<f64> {
        return DVector::from_iterator(self.get_num_parameters(), self.diagonal.iter().cloned().chain(self.lower_elements.iter().cloned()));
    }

    pub fn set_parameters_from_vector(&mut self, params: &DVector<f64>) {
        let size = self.size;
        self.diagonal = params.rows(0, size).into_owned();
        self.lower_elements = params.rows(size, size * (size - 1) / 2).into_owned();
    }
}


pub struct LinearGaussianStateSpaceParameters {
    size_state: usize,
    size_observation: usize,
    num_parameters: usize,

    pub initial_mean: DVector<f64>,
    pub initial_cov_dec: LowerTriangularMatrix,

    pub transition_matrix: DMatrix<f64>,
    pub observation_matrix: DMatrix<f64>,
    pub process_noise_cov_dec: LowerTriangularMatrix, 
    pub observation_noise_cov_dec: LowerTriangularMatrix,
}

impl LinearGaussianStateSpaceParameters {
    pub fn new(size_state: usize, size_observation: usize) -> Self {
        let num_parameters = size_state  // initial_mean
            + size_state * (size_state + 1) / 2  // initial_cov_dec
            + size_state * size_state  // transition_matrix
            + size_observation * size_state  // observation_matrix
            + size_state * (size_state + 1) / 2  // process_noise_cov_dec
            + size_observation * (size_observation + 1) / 2; // observation_noise_cov_dec
        Self {
            size_state,
            size_observation,
            num_parameters,

            initial_mean: DVector::zeros(size_state),
            initial_cov_dec: LowerTriangularMatrix::new(size_state),

            transition_matrix: DMatrix::identity(size_state, size_state),
            observation_matrix: DMatrix::identity(size_observation, size_state),
            process_noise_cov_dec: LowerTriangularMatrix::new(size_state),
            observation_noise_cov_dec: LowerTriangularMatrix::new(size_observation),
        }
    }

    pub fn get_initial_mean(&self) -> DVector<f64> {
        return self.initial_mean.clone(); 
    }

    pub fn set_initial_mean(&mut self, initial_mean: DVector<f64>) {
        self.initial_mean = initial_mean;
    }

    pub fn get_initial_cov(&self) -> DMatrix<f64> {
        &self.initial_cov_dec.to_dense() * self.initial_cov_dec.to_dense().transpose()
    }

    pub fn set_initial_cov_dec(&mut self, non_zero_elements: &DVector<f64>) {
        self.initial_cov_dec.set_parameters_from_vector(non_zero_elements);
    }

    pub fn get_transition_matrix(&self) -> DMatrix<f64> {
        return self.transition_matrix.clone();
    }

    pub fn set_transition_matrix(&mut self, transition_matrix: DMatrix<f64>) {
        self.transition_matrix = transition_matrix;
    }

    pub fn get_observation_matrix(&self) -> DMatrix<f64> {
        return self.observation_matrix.clone();
    }

    pub fn set_observation_matrix(&mut self, observation_matrix: DMatrix<f64>) {
        self.observation_matrix = observation_matrix;
    }

    pub fn get_process_noise_cov(&self) -> DMatrix<f64> {
        &self.process_noise_cov_dec.to_dense() * self.process_noise_cov_dec.to_dense().transpose()
    }

    pub fn set_process_noise_cov_dec(&mut self, elements: &DVector<f64>) {
        self.process_noise_cov_dec.set_parameters_from_vector(elements);
    }

    pub fn get_observation_noise_cov(&self) -> DMatrix<f64> {
        &self.observation_noise_cov_dec.to_dense() * self.observation_noise_cov_dec.to_dense().transpose()
    }

    pub fn set_observation_noise_cov_dec(&mut self, elements: &DVector<f64>) {
        self.observation_noise_cov_dec.set_parameters_from_vector(elements);
    }

}

impl ParameterSet for LinearGaussianStateSpaceParameters {
    fn get_parameters(&self) -> DVector<f64> {
        let initial_cov_dec_vector = self.initial_cov_dec.get_parameters_as_vector();
        let transition_matrix_vector = DVector::from_iterator(self.size_state * self.size_state, self.transition_matrix.iter().cloned());
        let observation_matrix_vector = DVector::from_iterator(self.size_observation * self.size_state, self.observation_matrix.iter().cloned());
        let process_noise_cov_dec_vector = self.process_noise_cov_dec.get_parameters_as_vector();
        let observation_noise_cov_dec_vector = self.observation_noise_cov_dec.get_parameters_as_vector();

        return DVector::from_iterator(
            self.initial_mean.len()
                + initial_cov_dec_vector.len()
                + transition_matrix_vector.len()
                + observation_matrix_vector.len()
                + process_noise_cov_dec_vector.len()
                + observation_noise_cov_dec_vector.len(), 
            self.initial_mean.iter().cloned()
                .chain(initial_cov_dec_vector.iter().cloned())
                .chain(transition_matrix_vector.iter().cloned())
                .chain(observation_matrix_vector.iter().cloned())
                .chain(process_noise_cov_dec_vector.iter().cloned())
                .chain(observation_noise_cov_dec_vector.iter().cloned())
        );
    }

    fn set_parameters(&mut self, params: &DVector<f64>) {
        let mut idx = 0;

        self.initial_mean = params.rows(idx, self.size_state).into_owned();
        idx += self.size_state;

        let num_initial_cov_dec_params = self.initial_cov_dec.get_num_parameters();
        self.initial_cov_dec.set_parameters_from_vector(&params.rows(idx, num_initial_cov_dec_params).into_owned());
        idx += num_initial_cov_dec_params;

        self.transition_matrix = DMatrix::from_iterator(self.size_state, self.size_state, params.rows(idx, self.size_state * self.size_state).iter().cloned());
        idx += self.size_state * self.size_state;

        self.observation_matrix = DMatrix::from_iterator(self.size_observation, self.size_state, params.rows(idx, self.size_observation * self.size_state).iter().cloned());
        idx += self.size_observation * self.size_state;

        let num_process_noise_cov_dec_params = self.process_noise_cov_dec.get_num_parameters();
        self.process_noise_cov_dec.set_parameters_from_vector(&params.rows(idx, num_process_noise_cov_dec_params).into_owned());
        idx += num_process_noise_cov_dec_params;

        let num_observation_noise_cov_dec_params = self.observation_noise_cov_dec.get_num_parameters();
        self.observation_noise_cov_dec.set_parameters_from_vector(&params.rows(idx, num_observation_noise_cov_dec_params).into_owned());
    }

    fn get_num_parameters(&self) -> usize {
        return self.num_parameters;
    }
}

pub struct LinearGaussianStateSpaceModel {
    pub parameters: LinearGaussianStateSpaceParameters
}

impl LinearGaussianStateSpaceModel {
    pub fn new(size_state: usize, size_observation: usize) -> Self {
        Self {
            parameters: LinearGaussianStateSpaceParameters::new(size_state, size_observation),
        }
    }

    pub fn calculate_num_parameters(size_state: usize, size_observation: usize) -> usize {
        return size_state  // initial_mean
            + size_state * (size_state + 1) / 2  // initial_cov_dec
            + size_state * size_state  // transition_matrix
            + size_observation * size_state  // observation_matrix
            + size_state * (size_state + 1) / 2  // process_noise_cov_dec
            + size_observation * (size_observation + 1) / 2; // observation_noise_cov_dec
    }

    pub fn new_from_parameter_vector(params: &DVector<f64>, size_state: usize, size_observation: usize) -> Self {
        let mut model = Self::new(size_state, size_observation);
        model.parameters.set_parameters(params);
        return model;
    }

    pub fn log_likelihood(&self, observations: &Vec<DMatrix<f64>>) -> f64 {
        let filtered_states = self.filter_state(observations);
        let observation_matrix = self.parameters.get_observation_matrix();
        let observation_noise_cov = self.parameters.get_observation_noise_cov();

        let mut log_likelihood = 0.0;

        for (t, obs) in observations.iter().enumerate() {
            let predicted_observation_mean = &observation_matrix * &filtered_states[t].mean;
            let predicted_observation_cov = &observation_matrix * &filtered_states[t].cov * observation_matrix.transpose() + &observation_noise_cov;

            let obs_dist = GaussianMvDistribution {
                mean: predicted_observation_mean,
                cov: predicted_observation_cov,
            };

            let obs_vec = obs.column(0).into_owned();
            log_likelihood += obs_dist.log_prob(&obs_vec);
        }

        return log_likelihood;
    }

    fn filter_state_internal(&self, observations: &Vec<DMatrix<f64>>) -> (Vec<GaussianMvDistribution>, Vec<GaussianMvDistribution>) {
        //return predicted states AND filtered states

        let num_observations = observations.len();

        let mut current_state_mean = self.parameters.get_initial_mean();
        let mut current_state_cov = self.parameters.get_initial_cov();

        let transition_matrix = self.parameters.get_transition_matrix();
        let observation_matrix = self.parameters.get_observation_matrix();
        let process_noise_cov = self.parameters.get_process_noise_cov();
        let observation_noise_cov = self.parameters.get_observation_noise_cov();

        let mut predicted_states = vec![];
        let mut filtered_states = vec![];

        for t in 0..num_observations {
            let next_mean = &transition_matrix * &current_state_mean; 
            let next_cov = &transition_matrix * &current_state_cov * &transition_matrix.transpose() + &process_noise_cov;

            let predicted_state = GaussianMvDistribution {
                mean: next_mean.clone(),
                cov: next_cov.clone(),
            };
            predicted_states.push(predicted_state);

            let predicted_observation_mean = &observation_matrix * &next_mean;
            let predicted_observation_cov = &observation_matrix * &next_cov * observation_matrix.transpose() + &observation_noise_cov;

            let current_observation = &observations[t];

            let current_error = current_observation - &predicted_observation_mean;
            let kalman_gain = &next_cov * observation_matrix.transpose() * &predicted_observation_cov.try_inverse().unwrap();

            let updated_mean = &next_mean + &kalman_gain * &current_error;
            let updated_cov = &next_cov - &kalman_gain * &observation_matrix * &next_cov;
            let filtered_current_state = GaussianMvDistribution {
                mean: updated_mean,
                cov: updated_cov,
            };

            filtered_states.push(filtered_current_state.clone());
        }

        return (predicted_states, filtered_states);

    }
}

impl StateSpaceModel<GaussianMvDistribution, GaussianMvDistribution> for LinearGaussianStateSpaceModel {
    fn get_parameters_as_vector(&self) -> DVector<f64> {
        self.parameters.get_parameters()
    }

    fn set_parameters_as_vector(&mut self, params: &DVector<f64>) {
        self.parameters.set_parameters(params);
    }

    fn get_num_parameters(&self) -> usize {
        self.parameters.get_num_parameters()
    }

    fn forecast(&self, observations: &Vec<DMatrix<f64>>, forecast_steps: &usize) -> Vec<GaussianMvDistribution> {
        let filtered_states;

        let initial_mean = self.parameters.get_initial_mean();
        let initial_cov = self.parameters.get_initial_cov();

        let transition_matrix = self.parameters.get_transition_matrix();
        let observation_matrix = self.parameters.get_observation_matrix();
        let process_noise_cov = self.parameters.get_process_noise_cov();
        let observation_noise_cov = self.parameters.get_observation_noise_cov();

        if observations.len() > 0 {
            filtered_states = self.filter_state(observations);
        } else {
            let initial_state = GaussianMvDistribution {
                mean: initial_mean,
                cov: initial_cov,
            };
            filtered_states = vec![initial_state];
        }

        let mut latest_state = filtered_states.last().unwrap().clone();
        let mut forecasted_observations = vec![];

        for _ in 0..*forecast_steps {
            let next_mean = &transition_matrix * &latest_state.mean;
            let next_cov = &transition_matrix * &latest_state.cov * transition_matrix.transpose() + &process_noise_cov;

            latest_state.mean = next_mean;
            latest_state.cov = next_cov;

            let forecasted_observation_mean = &observation_matrix * &latest_state.mean;
            let forecasted_observation_cov = &observation_matrix * &latest_state.cov * observation_matrix.transpose() + &observation_noise_cov;

            forecasted_observations.push(GaussianMvDistribution {
                mean: forecasted_observation_mean,
                cov: forecasted_observation_cov,
            });
        }

        return forecasted_observations;
    }

    fn filter_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<GaussianMvDistribution> {
        let (_, filtered_states) = self.filter_state_internal(observations);
        return filtered_states;
    }

    fn smooth_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<GaussianMvDistribution> {
        let (predicted_states, filtered_states) = self.filter_state_internal(observations);

        let transition_matrix = self.parameters.get_transition_matrix();

        let num_observations = observations.len();
        let mut smoothed_states = vec![filtered_states.last().unwrap().clone()];

        for t in (0..num_observations-1).rev() {
            let filtered_state = &filtered_states[t];
            let predicted_next_state = &predicted_states[t+1];
            let smoothed_next_state = smoothed_states.last().unwrap();

            let smoothing_gain = &filtered_state.cov * transition_matrix.transpose() * predicted_next_state.cov.clone().try_inverse().unwrap();

            let smoothed_mean = &filtered_state.mean + &smoothing_gain * (&smoothed_next_state.mean - &predicted_next_state.mean);
            let smoothed_cov = &filtered_state.cov + &smoothing_gain * (&smoothed_next_state.cov - &predicted_next_state.cov) * smoothing_gain.transpose();

            smoothed_states.push(GaussianMvDistribution {
                mean: smoothed_mean,
                cov: smoothed_cov,
            });
        }

        smoothed_states.reverse();
        return smoothed_states;
    }

    fn sample(&self, num_obserations: &usize, initial_state: Option<GaussianMvDistribution>, seed: Option<u64>) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let mut current_state = initial_state.unwrap_or_else(|| GaussianMvDistribution {
            mean: self.parameters.get_initial_mean(),
            cov: self.parameters.get_initial_cov(),
        });

        let transition_matrix = &self.parameters.get_transition_matrix();
        let observation_matrix = &self.parameters.get_observation_matrix();
        let process_noise_cov = &self.parameters.get_process_noise_cov();
        let observation_noise_cov = &self.parameters.get_observation_noise_cov();
            
        let mut states = vec![];
        let mut observations = vec![];

        let mut seeded_rng = seed.map(StdRng::seed_from_u64);

        for _ in 0..*num_obserations {
            current_state = GaussianMvDistribution {
                mean: transition_matrix * &current_state.mean,
                cov: transition_matrix * &current_state.cov * transition_matrix.transpose() + process_noise_cov,
            };

            match seeded_rng.as_mut() {
                Some(rng) => states.push(current_state.sample_with_rng(rng)),
                None => states.push(current_state.sample()),
            }

            let observation_mean = observation_matrix * &current_state.mean;
            let observation_cov = observation_matrix * &current_state.cov * observation_matrix.transpose() + observation_noise_cov;

            let observation_dist = GaussianMvDistribution {
                mean: observation_mean,
                cov: observation_cov,
            };

            match seeded_rng.as_mut() {
                Some(rng) => observations.push(observation_dist.sample_with_rng(rng)),
                None => observations.push(observation_dist.sample()),
            }
        }

        return (states, observations);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_linear_gaussian_state_space_model_forecast() {
        let size_state = 2;
        let size_observation = 2;
        let model = LinearGaussianStateSpaceModel::new(size_state, size_observation);

        let forecast = model.forecast(&vec![], &3);

        assert_eq!(forecast.len(), 3);
        for obs in forecast {
            assert_eq!(obs.mean.len(), size_observation);
            assert_eq!(obs.cov.nrows(), size_observation);
            assert_eq!(obs.cov.ncols(), size_observation);
        }
    }

    #[test]
    fn test_linear_gaussian_state_space_model_filter() {
        let size_state = 2;
        let size_observation = 2;
        let model = LinearGaussianStateSpaceModel::new(size_state, size_observation);

        let observations = vec![
            DMatrix::from_vec(size_observation, 1, vec![1.0, 0.0]),
            DMatrix::from_vec(size_observation, 1, vec![0.0, 1.0]),
        ];

        let filtered_states = model.filter_state(&observations);

        assert_eq!(filtered_states.len(), observations.len());
        for state in filtered_states {
            assert_eq!(state.mean.len(), size_state);
            assert_eq!(state.cov.nrows(), size_state);
            assert_eq!(state.cov.ncols(), size_state);
        }
    }

    #[test]
    fn test_linear_gaussian_state_space_model_smooth() {
        let size_state = 2;
        let size_observation = 2;
        let model = LinearGaussianStateSpaceModel::new(size_state, size_observation);

        let observations = vec![
            DMatrix::from_vec(size_observation, 1, vec![1.0, 0.0]),
            DMatrix::from_vec(size_observation, 1, vec![0.0, 1.0]),
        ];

        let smoothed_states = model.smooth_state(&observations);

        assert_eq!(smoothed_states.len(), observations.len());
        for state in smoothed_states {
            assert_eq!(state.mean.len(), size_state);
            assert_eq!(state.cov.nrows(), size_state);
            assert_eq!(state.cov.ncols(), size_state);
        }
    }

    #[test]
    fn test_linear_gaussian_state_space_model_sample() {
        let size_state = 2;
        let size_observation = 2;
        let model = LinearGaussianStateSpaceModel::new(size_state, size_observation);

        let (states, observations) = model.sample(&5, None, None);

        assert_eq!(states.len(), 5);
        for state in &states {
            assert_eq!(state.len(), size_state);
            assert!(state.iter().all(|x| x.is_finite()));
        }

        assert_eq!(observations.len(), 5);
        for obs in &observations {
            assert_eq!(obs.len(), size_observation);
            assert!(obs.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_seeded_sample_is_deterministic() {
        let model = LinearGaussianStateSpaceModel::new(2, 2);

        let (states_a, obs_a) = model.sample(&10, None, Some(42));
        let (states_b, obs_b) = model.sample(&10, None, Some(42));

        for (a, b) in states_a.iter().zip(states_b.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in obs_a.iter().zip(obs_b.iter()) {
            assert_eq!(a, b);
        }

        // Different seed produces different results
        let (states_c, _) = model.sample(&10, None, Some(99));
        assert_ne!(states_a, states_c);
    }

    // --- LowerTriangularMatrix tests ---

    #[test]
    fn test_lower_triangular_new() {
        let ltm = LowerTriangularMatrix::new(3);
        assert_eq!(ltm.get_size(), 3);
        assert_eq!(ltm.get_diagonal(), DVector::from_element(3, 1.0));
        assert_eq!(ltm.get_lower_elements(), DVector::zeros(3));
        assert_eq!(ltm.get_num_parameters(), 6); // 3 + 3*(3-1)/2
    }

    #[test]
    fn test_lower_triangular_new_with_values() {
        let ltm = LowerTriangularMatrix::new_with_values(3, 2.0, 0.5);
        assert_eq!(ltm.get_size(), 3);
        assert_eq!(ltm.get_diagonal(), DVector::from_element(3, 2.0));
        assert_eq!(ltm.get_lower_elements(), DVector::from_element(3, 0.5));
    }

    #[test]
    fn test_lower_triangular_to_dense() {
        let ltm = LowerTriangularMatrix::new_with_values(3, 2.0, 0.5);
        let dense = ltm.to_dense();
        assert_eq!(dense.nrows(), 3);
        assert_eq!(dense.ncols(), 3);
        // Diagonal
        assert_eq!(dense[(0, 0)], 2.0);
        assert_eq!(dense[(1, 1)], 2.0);
        assert_eq!(dense[(2, 2)], 2.0);
        // Lower elements
        assert_eq!(dense[(1, 0)], 0.5);
        assert_eq!(dense[(2, 0)], 0.5);
        assert_eq!(dense[(2, 1)], 0.5);
        // Upper triangle is zero
        assert_eq!(dense[(0, 1)], 0.0);
        assert_eq!(dense[(0, 2)], 0.0);
        assert_eq!(dense[(1, 2)], 0.0);
    }

    #[test]
    fn test_lower_triangular_set_diagonal_and_lower() {
        let mut ltm = LowerTriangularMatrix::new(2);
        ltm.set_diagonal(DVector::from_vec(vec![3.0, 4.0]));
        ltm.set_lower_elements(DVector::from_vec(vec![1.5]));

        assert_eq!(ltm.get_diagonal(), DVector::from_vec(vec![3.0, 4.0]));
        assert_eq!(ltm.get_lower_elements(), DVector::from_vec(vec![1.5]));

        let dense = ltm.to_dense();
        assert_eq!(dense[(0, 0)], 3.0);
        assert_eq!(dense[(1, 1)], 4.0);
        assert_eq!(dense[(1, 0)], 1.5);
    }

    #[test]
    fn test_lower_triangular_parameters_round_trip() {
        let mut ltm = LowerTriangularMatrix::new_with_values(3, 2.0, 0.5);
        let params = ltm.get_parameters_as_vector();
        assert_eq!(params.len(), 6);
        // [diag0, diag1, diag2, lower0, lower1, lower2]
        assert_eq!(params[0], 2.0);
        assert_eq!(params[1], 2.0);
        assert_eq!(params[2], 2.0);
        assert_eq!(params[3], 0.5);
        assert_eq!(params[4], 0.5);
        assert_eq!(params[5], 0.5);

        // Modify via set_parameters_from_vector
        let new_params = DVector::from_vec(vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        ltm.set_parameters_from_vector(&new_params);
        assert_eq!(ltm.get_diagonal(), DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(ltm.get_lower_elements(), DVector::from_vec(vec![0.1, 0.2, 0.3]));
    }

    // --- LinearGaussianStateSpaceParameters tests ---

    #[test]
    fn test_parameters_setters() {
        let mut params = LinearGaussianStateSpaceParameters::new(2, 2);

        params.set_initial_mean(DVector::from_vec(vec![1.0, 2.0]));
        assert_eq!(params.get_initial_mean(), DVector::from_vec(vec![1.0, 2.0]));

        params.set_initial_cov_dec(&DVector::from_vec(vec![2.0, 3.0, 0.5]));
        let cov = params.get_initial_cov();
        assert!(cov[(0, 0)] > 0.0); // L * L^T is PSD

        params.set_transition_matrix(DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.8]));
        assert_eq!(params.get_transition_matrix()[(0, 0)], 0.9);

        params.set_observation_matrix(DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]));
        assert_eq!(params.get_observation_matrix()[(0, 0)], 1.0);

        params.set_process_noise_cov_dec(&DVector::from_vec(vec![0.5, 0.5, 0.1]));
        let q = params.get_process_noise_cov();
        assert!(q[(0, 0)] > 0.0);

        params.set_observation_noise_cov_dec(&DVector::from_vec(vec![0.3, 0.3, 0.0]));
        let r = params.get_observation_noise_cov();
        assert!(r[(0, 0)] > 0.0);
    }

    #[test]
    fn test_parameter_set_round_trip() {
        let mut params = LinearGaussianStateSpaceParameters::new(2, 2);
        params.set_initial_mean(DVector::from_vec(vec![1.0, 2.0]));
        params.set_transition_matrix(DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.8]));

        let vec = params.get_parameters();
        let expected_len = 2  // initial_mean
            + 3  // initial_cov_dec (2+1)
            + 4  // transition_matrix (2x2)
            + 4  // observation_matrix (2x2)
            + 3  // process_noise_cov_dec (2+1)
            + 3; // observation_noise_cov_dec (2+1)
        assert_eq!(vec.len(), expected_len);

        // Round-trip: set_parameters then get_parameters should give same vector
        let mut params2 = LinearGaussianStateSpaceParameters::new(2, 2);
        params2.set_parameters(&vec);
        let vec2 = params2.get_parameters();
        for i in 0..vec.len() {
            assert!((vec[i] - vec2[i]).abs() < 1e-12, "Mismatch at index {}: {} vs {}", i, vec[i], vec2[i]);
        }
    }

    // --- StateSpaceModel trait method delegation tests ---

    #[test]
    fn test_model_get_set_parameters_as_vector() {
        let mut model = LinearGaussianStateSpaceModel::new(2, 2);
        let params = model.get_parameters_as_vector();
        assert!(!params.is_empty());

        // Modify and set back
        let mut modified = params.clone();
        modified[0] = 99.0;
        model.set_parameters_as_vector(&modified);

        let retrieved = model.get_parameters_as_vector();
        assert_eq!(retrieved[0], 99.0);
    }

    #[test]
    fn test_forecast_with_observations() {
        let model = LinearGaussianStateSpaceModel::new(2, 2);
        let observations = vec![
            DMatrix::from_vec(2, 1, vec![1.0, 0.5]),
            DMatrix::from_vec(2, 1, vec![0.5, 1.0]),
            DMatrix::from_vec(2, 1, vec![1.5, 0.3]),
        ];

        let forecast = model.forecast(&observations, &2);
        assert_eq!(forecast.len(), 2);
        for f in &forecast {
            assert_eq!(f.mean.len(), 2);
            assert_eq!(f.cov.nrows(), 2);
        }
    }

    #[test]
    fn test_sample_with_initial_state() {
        let model = LinearGaussianStateSpaceModel::new(2, 2);
        let initial = GaussianMvDistribution {
            mean: DVector::from_vec(vec![5.0, 5.0]),
            cov: DMatrix::identity(2, 2) * 0.1,
        };

        let (states, obs) = model.sample(&5, Some(initial), Some(123));
        assert_eq!(states.len(), 5);
        assert_eq!(obs.len(), 5);
    }

    // --- Placeholder trait impl tests ---

    #[test]
    fn test_differentiable_once_placeholder() {
        let model = LinearGaussianStateSpaceModel::new(2, 2);
        let grad = model.get_gradient();
        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0], 0.0);
    }

    #[test]
    fn test_differentiable_twice_placeholder() {
        let model = LinearGaussianStateSpaceModel::new(2, 2);
        let hess = model.get_hessian();
        assert_eq!(hess.nrows(), 1);
        assert_eq!(hess.ncols(), 1);
        assert_eq!(hess[(0, 0)], 0.0);
    }
}


impl DifferentiableOnce for LinearGaussianStateSpaceModel {
    fn get_gradient(&self) -> DVector<f64> {
        //placeholder
        return DVector::zeros(1);
    }
}

impl DifferentiableTwice for LinearGaussianStateSpaceModel {
    fn get_hessian(&self) -> DMatrix<f64> {
        //placeholder
        return DMatrix::zeros(1, 1);
    }
}
        

