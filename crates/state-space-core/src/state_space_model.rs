use nalgebra::{DMatrix, DVector};

use crate::distributions::{MvDistribution, GaussianMvDistribution};

pub trait StateSpaceModel<T: MvDistribution> {
    fn forecast(&self, observations: &Vec<DMatrix<f64>>, forecast_steps: &usize) -> Vec<T>;

    fn filter_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<T>;

    fn smooth_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<T>;
}


pub struct LinearGaussianStateSpaceModel {
    pub size: usize,

    pub initial_distribution: GaussianMvDistribution,

    pub transition_matrix: DMatrix<f64>,
    pub observation_matrix: DMatrix<f64>,
    pub process_noise_cov: DMatrix<f64>,
    pub observation_noise_cov: DMatrix<f64>,
}

impl LinearGaussianStateSpaceModel {
    pub fn new(size_state: usize, size_observation: usize) -> Self {
        let initial_distribution = GaussianMvDistribution {
            mean: DVector::zeros(size_state),
            cov: DMatrix::identity(size_state, size_state),
        };

        Self {
            size: size_state,

            initial_distribution,

            transition_matrix: DMatrix::identity(size_state, size_state),
            observation_matrix: DMatrix::identity(size_observation, size_state),
            process_noise_cov: DMatrix::identity(size_state, size_state),
            observation_noise_cov: DMatrix::identity(size_observation, size_observation),
        }
    }
}

impl StateSpaceModel<GaussianMvDistribution> for LinearGaussianStateSpaceModel {
    fn forecast(&self, observations: &Vec<DMatrix<f64>>, forecast_steps: &usize) -> Vec<GaussianMvDistribution> {
        let filtered_states;

        if observations.len() > 0 {
            filtered_states = self.filter_state(observations);
        } else {
            filtered_states = vec![self.initial_distribution.clone()];
        }

        let mut latest_state = filtered_states.last().unwrap().clone();
        let mut forecasted_observations = vec![];

        for _ in 0..*forecast_steps {
            let next_mean = &self.transition_matrix * &latest_state.mean;
            let next_cov = &self.transition_matrix * &latest_state.cov * self.transition_matrix.transpose() + &self.process_noise_cov;

            latest_state.mean = next_mean;
            latest_state.cov = next_cov;

            let forecasted_observation_mean = &self.observation_matrix * &latest_state.mean;
            let forecasted_observation_cov = &self.observation_matrix * &latest_state.cov * self.observation_matrix.transpose() + &self.observation_noise_cov;

            forecasted_observations.push(GaussianMvDistribution {
                mean: forecasted_observation_mean,
                cov: forecasted_observation_cov,
            });
        }

        return forecasted_observations;
    }

    fn filter_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<GaussianMvDistribution> {
        let num_observations = observations.len();

        let mut current_state = self.initial_distribution.clone();

        let mut filtered_states = vec![];

        for t in 0..num_observations {
            let next_mean = &self.transition_matrix * &current_state.mean; 
            let next_cov = &self.transition_matrix * &current_state.cov * &self.transition_matrix.transpose() + &self.process_noise_cov;

            let predicted_observation_mean = &self.observation_matrix * &next_mean;
            let predicted_observation_cov = &self.observation_matrix * &next_cov * self.observation_matrix.transpose() + &self.observation_noise_cov;

            let current_observation = &observations[t];

            let current_error = current_observation - &predicted_observation_mean;
            let kalman_gain = &next_cov * self.observation_matrix.transpose() * &predicted_observation_cov.try_inverse().unwrap();

            let updated_mean = &next_mean + &kalman_gain * &current_error;
            let updated_cov = &next_cov - &kalman_gain * &self.observation_matrix * &next_cov;
            let filtered_current_state = GaussianMvDistribution {
                mean: updated_mean,
                cov: updated_cov,
            };

            filtered_states.push(filtered_current_state.clone());
        }

        return filtered_states;
    }

    fn smooth_state(&self, observations: &Vec<DMatrix<f64>>) -> Vec<GaussianMvDistribution> {
        return vec![];
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
}

        

