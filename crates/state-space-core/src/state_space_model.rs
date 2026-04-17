use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::distributions::{MvDistribution, GaussianMvDistribution};


pub trait ParameterSet {
    fn get_parameters(&self) -> DVector<f64>;
    fn set_parameters(&mut self, params: &DVector<f64>);
}


pub trait StateSpaceModel<T: MvDistribution, S: MvDistribution> {
    //T: observed state distribution, S: initial state distribution
    
    fn get_parameters_as_vector(&self) -> DVector<f64>;

    fn set_parameters_as_vector(&mut self, params: &DVector<f64>);

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


pub struct LinearGaussianStateSpaceParameters {
    size_state: usize,
    size_observation: usize,

    pub initial_mean: DVector<f64>,
    pub initial_cov_dec: DMatrix<f64>, // cov=L*L^T

    pub transition_matrix: DMatrix<f64>,
    pub observation_matrix: DMatrix<f64>,
    pub process_noise_cov_dec: DMatrix<f64>,
    pub observation_noise_cov_dec: DMatrix<f64>,
}

impl LinearGaussianStateSpaceParameters {
    pub fn new(size_state: usize, size_observation: usize) -> Self {
        Self {
            size_state,
            size_observation,

            initial_mean: DVector::zeros(size_state),
            initial_cov_dec: DMatrix::identity(size_state, size_state),

            transition_matrix: DMatrix::identity(size_state, size_state),
            observation_matrix: DMatrix::identity(size_observation, size_state),
            process_noise_cov_dec: DMatrix::identity(size_state, size_state),
            observation_noise_cov_dec: DMatrix::identity(size_observation, size_observation),
        }
    }

    pub fn get_initial_mean(&self) -> DVector<f64> {
        return self.initial_mean.clone(); 
    }

    pub fn set_initial_mean(&mut self, initial_mean: DVector<f64>) {
        self.initial_mean = initial_mean;
    }

    pub fn get_initial_cov(&self) -> DMatrix<f64> {
        &self.initial_cov_dec * self.initial_cov_dec.transpose()
    }

    pub fn set_initial_cov_dec(&mut self, initial_cov_dec: DMatrix<f64>) {
        self.initial_cov_dec = initial_cov_dec;
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
        &self.process_noise_cov_dec * self.process_noise_cov_dec.transpose()
    }

    pub fn set_process_noise_cov_dec(&mut self, process_noise_cov_dec: DMatrix<f64>) {
        self.process_noise_cov_dec = process_noise_cov_dec;
    }

    pub fn get_observation_noise_cov(&self) -> DMatrix<f64> {
        &self.observation_noise_cov_dec * self.observation_noise_cov_dec.transpose()
    }

    pub fn set_observation_noise_cov_dec(&mut self, observation_noise_cov_dec: DMatrix<f64>) {
        self.observation_noise_cov_dec = observation_noise_cov_dec;
    }

}

impl ParameterSet for LinearGaussianStateSpaceParameters {
    fn get_parameters(&self) -> DVector<f64> {
        let initial_cov_dec_vector = DVector::from_iterator(self.size_state * self.size_state, self.initial_cov_dec.iter().cloned());
        let transition_matrix_vector = DVector::from_iterator(self.size_state * self.size_state, self.transition_matrix.iter().cloned());
        let observation_matrix_vector = DVector::from_iterator(self.size_observation * self.size_state, self.observation_matrix.iter().cloned());
        let process_noise_cov_dec_vector = DVector::from_iterator(self.size_state * self.size_state, self.process_noise_cov_dec.iter().cloned());
        let observation_noise_cov_dec_vector = DVector::from_iterator(self.size_observation * self.size_observation, self.observation_noise_cov_dec.iter().cloned());

        return DVector::from_iterator(
            self.size_state + 
            self.size_state * self.size_state + 
            self.size_state * self.size_state + 
            self.size_observation * self.size_state + 
            self.size_state * self.size_state + 
            self.size_observation * self.size_observation, 
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

        self.initial_cov_dec = DMatrix::from_iterator(self.size_state, self.size_state, params.rows(idx, self.size_state * self.size_state).iter().cloned());
        idx += self.size_state * self.size_state;

        self.transition_matrix = DMatrix::from_iterator(self.size_state, self.size_state, params.rows(idx, self.size_state * self.size_state).iter().cloned());
        idx += self.size_state * self.size_state;

        self.observation_matrix = DMatrix::from_iterator(self.size_observation, self.size_state, params.rows(idx, self.size_observation * self.size_state).iter().cloned());
        idx += self.size_observation * self.size_state;

        self.process_noise_cov_dec = DMatrix::from_iterator(self.size_state, self.size_state, params.rows(idx, self.size_state * self.size_state).iter().cloned());
        idx += self.size_state * self.size_state;

        self.observation_noise_cov_dec = DMatrix::from_iterator(self.size_observation, self.size_observation, params.rows(idx, self.size_observation * self.size_observation).iter().cloned());
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
        

