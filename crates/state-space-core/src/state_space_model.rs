use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::marker::PhantomData;

use crate::{
    distributions::{CenteredGaussianDistribution, Distribution, GaussianDistribution},
    linear_algebra::SchurStableMatrix,
    parameter_set::ParameterSet,
};

pub trait StateSpaceModel {
    type Parameters: ParameterSet;
    type InitialStateDist: Distribution;
    type StateDist: Distribution;
    type ObsDist: Distribution;
    type ForcDist: Distribution;
    type FiltDist: Distribution;
    type SmoothDist: Distribution;

    fn get_parameters(&self) -> DVector<f64>;
    fn set_parameters(&mut self, params: &DVector<f64>);
    fn get_num_parameters(&self) -> usize;

    fn forecast(
        &self,
        observations: &Vec<DMatrix<f64>>,
        forecast_steps: &usize,
        observed_control_variables: Option<&Vec<DMatrix<f64>>>,
        forecast_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> Vec<Self::ForcDist>;
    fn filter_state(
        &self,
        observations: &Vec<DMatrix<f64>>,
        observed_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> Vec<Self::FiltDist>;
    fn smooth_state(
        &self,
        observations: &Vec<DMatrix<f64>>,
        observed_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> Vec<Self::SmoothDist>;
    fn sample(
        &self,
        num_observations: &usize,
        initial_state: Option<Self::StateDist>,
        observed_control_variables: Option<&Vec<DMatrix<f64>>>,
        seed: Option<u64>,
    ) -> (Vec<DVector<f64>>, Vec<DVector<f64>>);
}

pub struct LinearStateSpaceParameters<InitialDist, StateDist, ObsDist>
where
    InitialDist: Distribution,
    StateDist: Distribution,
    ObsDist: Distribution,
{
    dim_state: usize,
    dim_observation: usize,
    num_parameters: usize,

    initial_state_dist: InitialDist,

    transition_matrix: SchurStableMatrix,
    observation_matrix: DMatrix<f64>,
    state_dist: StateDist,
    observation_dist: ObsDist,
}

impl<S, T, U> LinearStateSpaceParameters<S, T, U>
where
    S: Distribution + Clone,
    T: Distribution + Clone,
    U: Distribution + Clone,
{
    pub fn new_from_dist(
        initial_state_dist: S,
        state_dist: T,
        observation_dist: U,
    ) -> anyhow::Result<Self> {
        let dim_state = initial_state_dist.get_dim();
        if state_dist.get_dim() != dim_state {
            return Err(anyhow::anyhow!(
                "State distribution dimension does not match initial state distribution dimension"
            ));
        }

        let dim_observation = observation_dist.get_dim();

        let transition_matrix = SchurStableMatrix::new(dim_state);
        let observation_matrix = DMatrix::identity(dim_observation, dim_state);

        let num_parameters = initial_state_dist.get_num_parameters()
            + transition_matrix.get_num_parameters()
            + observation_matrix.nrows() * observation_matrix.ncols()
            + state_dist.get_num_parameters()
            + observation_dist.get_num_parameters();
        Ok(Self {
            dim_state,
            dim_observation,
            num_parameters,
            initial_state_dist,
            transition_matrix,
            observation_matrix,
            state_dist,
            observation_dist,
        })
    }

    pub fn get_transition_matrix(&self) -> DMatrix<f64> {
        self.transition_matrix.to_dense()
    }

    pub fn get_observation_matrix(&self) -> DMatrix<f64> {
        self.observation_matrix.clone()
    }

    pub fn get_initial_state_dist(&self) -> S {
        self.initial_state_dist.clone()
    }

    pub fn get_state_dist(&self) -> T {
        self.state_dist.clone()
    }

    pub fn get_observation_dist(&self) -> U {
        self.observation_dist.clone()
    }
}

impl<S: Distribution, T: Distribution, U: Distribution> ParameterSet
    for LinearStateSpaceParameters<S, T, U>
{
    fn get_parameters(&self) -> DVector<f64> {
        let mut params = Vec::with_capacity(self.num_parameters);

        params.extend(self.initial_state_dist.get_parameters().iter().cloned());
        params.extend(
            self.transition_matrix
                .get_parameters_as_vector()
                .iter()
                .cloned(),
        );
        params.extend(self.observation_matrix.iter().cloned());
        params.extend(self.state_dist.get_parameters().iter().cloned());
        params.extend(self.observation_dist.get_parameters().iter().cloned());

        return DVector::from_vec(params);
    }

    fn set_parameters(&mut self, params: &DVector<f64>) -> anyhow::Result<()> {
        if params.len() != self.num_parameters {
            return Err(anyhow::anyhow!(
                "Parameter vector length does not match expected number of parameters"
            ));
        }

        let mut idx = 0;

        let initial_state_params_len = self.initial_state_dist.get_num_parameters();
        self.initial_state_dist
            .set_parameters(&params.rows(idx, initial_state_params_len).into())?;
        idx += initial_state_params_len;

        self.transition_matrix.set_parameters_from_vector(
            &params
                .rows(idx, self.transition_matrix.get_num_parameters())
                .into(),
        )?;
        idx += self.transition_matrix.get_num_parameters();

        let obs_matrix_len = self.observation_matrix.nrows() * self.observation_matrix.ncols();
        self.observation_matrix.copy_from(&DMatrix::from_row_slice(
            self.observation_matrix.nrows(),
            self.observation_matrix.ncols(),
            params.rows(idx, obs_matrix_len).as_slice(),
        ));
        idx += obs_matrix_len;

        self.state_dist.set_parameters(
            &params
                .rows(idx, self.state_dist.get_num_parameters())
                .into(),
        )?;
        idx += self.state_dist.get_num_parameters();

        self.observation_dist.set_parameters(
            &params
                .rows(idx, self.observation_dist.get_num_parameters())
                .into(),
        )?;

        Ok(())
    }

    fn get_num_parameters(&self) -> usize {
        return self.num_parameters;
    }
}

pub struct LinearGaussianStateSpaceModel {
    parameters: LinearStateSpaceParameters<
        GaussianDistribution,
        CenteredGaussianDistribution,
        CenteredGaussianDistribution,
    >,

    filter_dist: PhantomData<GaussianDistribution>,
    smoothing_dist: PhantomData<GaussianDistribution>,
    forecast_dist: PhantomData<GaussianDistribution>,
}

impl LinearGaussianStateSpaceModel {
    pub fn new(size_state: usize, size_observation: usize) -> Self {
        let initial_dist = GaussianDistribution::new_with_dim(size_state);
        let state_dist = CenteredGaussianDistribution::new_with_dim(size_state);
        let obs_dist = CenteredGaussianDistribution::new_with_dim(size_observation);

        let filter_dist = PhantomData;
        let smoothing_dist = PhantomData;
        let forecast_dist = PhantomData;

        Self {
            parameters: LinearStateSpaceParameters::new_from_dist(
                initial_dist,
                state_dist,
                obs_dist,
            )
            .unwrap(),
            filter_dist,
            smoothing_dist,
            forecast_dist,
        }
    }

    pub fn log_likelihood(
        &self,
        observations: &Vec<DMatrix<f64>>,
        observation_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> anyhow::Result<f64> {
        let filtered_states = self.filter_state(observations, observation_control_variables);
        let observation_matrix = self.parameters.get_observation_matrix();
        let observation_noise_cov = self.parameters.get_observation_dist().get_cov();

        let mut log_likelihood = 0.0;

        for (t, obs) in observations.iter().enumerate() {
            let predicted_observation_mean = &observation_matrix * &filtered_states[t].get_mean();
            let predicted_observation_cov = &observation_matrix
                * &filtered_states[t].get_cov()
                * observation_matrix.transpose()
                + &observation_noise_cov;

            let obs_dist = GaussianDistribution::new_from_params(
                predicted_observation_mean,
                predicted_observation_cov,
            )
            .unwrap();

            let obs_vec = obs.column(0).into_owned();
            log_likelihood += obs_dist.log_prob(&obs_vec)?;
        }

        Ok(log_likelihood)
    }

    fn filter_state_internal(
        &self,
        observations: &Vec<DMatrix<f64>>,
        _observed_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> (Vec<GaussianDistribution>, Vec<GaussianDistribution>) {
        //return predicted states AND filtered states
        let num_observations = observations.len();

        let initial_dist = &(self.parameters.get_initial_state_dist());
        let state_dist = &(self.parameters.get_state_dist());
        let obs_dist = &(self.parameters.get_observation_dist());

        let current_state_mean = initial_dist.get_mean();
        let current_state_cov = initial_dist.get_cov();

        let transition_matrix = self.parameters.get_transition_matrix();
        let observation_matrix = self.parameters.get_observation_matrix();
        let process_noise_cov = state_dist.get_cov();
        let observation_noise_cov = obs_dist.get_cov();

        let mut predicted_states = vec![];
        let mut filtered_states = vec![];

        for t in 0..num_observations {
            let next_mean = &transition_matrix * &current_state_mean;
            let next_cov = &transition_matrix * &current_state_cov * &transition_matrix.transpose()
                + &process_noise_cov;

            let predicted_state =
                GaussianDistribution::new_from_params(next_mean.clone(), next_cov.clone()).unwrap();
            predicted_states.push(predicted_state);

            let predicted_observation_mean = &observation_matrix * &next_mean;
            let predicted_observation_cov =
                &observation_matrix * &next_cov * observation_matrix.transpose()
                    + &observation_noise_cov;

            let current_observation = &observations[t];

            let current_error = current_observation - &predicted_observation_mean;
            let kalman_gain = &next_cov
                * observation_matrix.transpose()
                * &predicted_observation_cov.try_inverse().unwrap();

            let updated_mean = &next_mean + &kalman_gain * &current_error;
            let updated_cov = &next_cov - &kalman_gain * &observation_matrix * &next_cov;
            let filtered_current_state =
                GaussianDistribution::new_from_params(updated_mean, updated_cov).unwrap();

            filtered_states.push(filtered_current_state.clone());
        }

        return (predicted_states, filtered_states);
    }
}

impl StateSpaceModel for LinearGaussianStateSpaceModel {
    type Parameters = LinearStateSpaceParameters<
        GaussianDistribution,
        CenteredGaussianDistribution,
        CenteredGaussianDistribution,
    >;
    type InitialStateDist = GaussianDistribution;
    type StateDist = GaussianDistribution;
    type ObsDist = GaussianDistribution;
    type ForcDist = GaussianDistribution;
    type FiltDist = GaussianDistribution;
    type SmoothDist = GaussianDistribution;

    fn get_parameters(&self) -> DVector<f64> {
        let parameters = self.parameters.get_parameters();
        parameters
    }

    fn set_parameters(&mut self, params: &DVector<f64>) {
        self.parameters.set_parameters(params);
    }

    fn get_num_parameters(&self) -> usize {
        self.parameters.get_num_parameters()
    }

    fn forecast(
        &self,
        observations: &Vec<DMatrix<f64>>,
        forecast_steps: &usize,
        observed_control_variables: Option<&Vec<DMatrix<f64>>>,
        _forecast_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> Vec<GaussianDistribution> {
        let filtered_states;

        let initial_dist = &(self.parameters.get_initial_state_dist());
        let state_dist = &(self.parameters.get_state_dist());
        let obs_dist = &(self.parameters.get_observation_dist());

        let current_state_mean = initial_dist.get_mean();
        let current_state_cov = initial_dist.get_cov();

        let transition_matrix = self.parameters.get_transition_matrix();
        let observation_matrix = self.parameters.get_observation_matrix();
        let process_noise_cov = state_dist.get_cov();
        let observation_noise_cov = obs_dist.get_cov();

        if observations.len() > 0 {
            filtered_states = self.filter_state(observations, observed_control_variables);
        } else {
            let initial_state =
                GaussianDistribution::new_from_params(current_state_mean, current_state_cov)
                    .unwrap();
            filtered_states = vec![initial_state];
        }

        let mut latest_state = filtered_states.last().unwrap().clone();
        let mut forecasted_observations = vec![];

        for _ in 0..*forecast_steps {
            let next_mean = &transition_matrix * &latest_state.get_mean();
            let next_cov =
                &transition_matrix * &latest_state.get_cov() * transition_matrix.transpose()
                    + &process_noise_cov;

            latest_state = GaussianDistribution::new_from_params(next_mean, next_cov).unwrap();

            let forecasted_observation_mean: DVector<f64> =
                &observation_matrix * &latest_state.get_mean();
            let forecasted_observation_cov =
                &observation_matrix * &latest_state.get_cov() * &observation_matrix.transpose()
                    + &observation_noise_cov;

            forecasted_observations.push(
                GaussianDistribution::new_from_params(
                    forecasted_observation_mean,
                    forecasted_observation_cov,
                )
                .unwrap(),
            );
        }

        return forecasted_observations;
    }

    fn filter_state(
        &self,
        observations: &Vec<DMatrix<f64>>,
        observed_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> Vec<GaussianDistribution> {
        let (_, filtered_states) =
            self.filter_state_internal(observations, observed_control_variables);
        return filtered_states;
    }

    fn smooth_state(
        &self,
        observations: &Vec<DMatrix<f64>>,
        observed_control_variables: Option<&Vec<DMatrix<f64>>>,
    ) -> Vec<GaussianDistribution> {
        let (predicted_states, filtered_states) =
            self.filter_state_internal(observations, observed_control_variables);

        let transition_matrix = self.parameters.get_transition_matrix();

        let num_observations = observations.len();
        let mut smoothed_states = vec![filtered_states.last().unwrap().clone()];

        for t in (0..num_observations - 1).rev() {
            let filtered_state = &filtered_states[t];
            let predicted_next_state = &predicted_states[t + 1];
            let smoothed_next_state = smoothed_states.last().unwrap();

            let smoothing_gain = &filtered_state.get_cov()
                * transition_matrix.transpose()
                * predicted_next_state
                    .get_cov()
                    .clone()
                    .try_inverse()
                    .unwrap();

            let smoothed_mean = &filtered_state.get_mean()
                + &smoothing_gain
                    * (&smoothed_next_state.get_mean() - &predicted_next_state.get_mean());
            let smoothed_cov = &filtered_state.get_cov()
                + &smoothing_gain
                    * (&smoothed_next_state.get_cov() - &predicted_next_state.get_cov())
                    * smoothing_gain.transpose();

            smoothed_states
                .push(GaussianDistribution::new_from_params(smoothed_mean, smoothed_cov).unwrap());
        }

        smoothed_states.reverse();
        return smoothed_states;
    }

    fn sample(
        &self,
        num_obserations: &usize,
        initial_state: Option<GaussianDistribution>,
        _observed_control_variables: Option<&Vec<DMatrix<f64>>>,
        seed: Option<u64>,
    ) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let initial_dist = &(self.parameters.get_initial_state_dist());
        let state_dist = &(self.parameters.get_state_dist());
        let obs_dist = &(self.parameters.get_observation_dist());

        let transition_matrix = self.parameters.get_transition_matrix();
        let observation_matrix = self.parameters.get_observation_matrix();
        let process_noise_cov = state_dist.get_cov();
        let observation_noise_cov = obs_dist.get_cov();

        let mut current_state = initial_state.unwrap_or_else(|| {
            GaussianDistribution::new_from_params(initial_dist.get_mean(), initial_dist.get_cov())
                .unwrap()
        });

        let mut states = vec![];
        let mut observations = vec![];

        let mut seeded_rng = seed.map(StdRng::seed_from_u64);

        for _ in 0..*num_obserations {
            current_state = GaussianDistribution::new_from_params(
                &transition_matrix * &current_state.get_mean(),
                &transition_matrix * &current_state.get_cov() * &transition_matrix.transpose()
                    + &process_noise_cov,
            )
            .unwrap();

            match seeded_rng.as_mut() {
                Some(rng) => states.push(current_state.sample_with_rng(rng)),
                None => states.push(current_state.sample()),
            }

            let observation_mean = &observation_matrix * &current_state.get_mean();
            let observation_cov =
                &observation_matrix * &current_state.get_cov() * &observation_matrix.transpose()
                    + &observation_noise_cov;

            let observation_dist = GaussianDistribution::new_from_params(
                observation_mean.clone(),
                observation_cov.clone(),
            )
            .unwrap();

            match seeded_rng.as_mut() {
                Some(rng) => observations.push(observation_dist.sample_with_rng(rng)),
                None => observations.push(observation_dist.sample()),
            }
        }

        return (states, observations);
    }
}

/*
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
        assert_eq!(
            ltm.get_lower_elements(),
            DVector::from_vec(vec![0.1, 0.2, 0.3])
        );
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
            assert!(
                (vec[i] - vec2[i]).abs() < 1e-12,
                "Mismatch at index {}: {} vs {}",
                i,
                vec[i],
                vec2[i]
            );
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
        let initial = GaussianDistribution {
            mean: DVector::from_vec(vec![5.0, 5.0]),
            cov: DMatrix::identity(2, 2) * 0.1,
        };

        let (states, obs) = model.sample(&5, Some(initial), Some(123));
        assert_eq!(states.len(), 5);
        assert_eq!(obs.len(), 5);
    }
}

*/
