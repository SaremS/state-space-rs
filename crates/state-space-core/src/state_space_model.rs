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
        return vec![];
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
}
        

