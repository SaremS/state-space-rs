use pyo3::prelude::*;

#[pymodule]
mod state_space_rs {
    use pyo3::prelude::*;
    use nalgebra::DMatrix;
    use state_space_core::distributions::GaussianMvDistribution;
    use state_space_core::state_space_model::{LinearGaussianStateSpaceModel, StateSpaceModel};

    #[pyclass]
    #[pyo3(name = "GaussianDistribution")]
    struct PyGaussianDistribution {
        #[pyo3(get)]
        mean: Vec<f64>,
        #[pyo3(get)]
        cov: Vec<Vec<f64>>,
    }

    fn gaussian_to_py(dist: GaussianMvDistribution) -> PyGaussianDistribution {
        let mean = dist.mean.as_slice().to_vec();
        let nrows = dist.cov.nrows();
        let ncols = dist.cov.ncols();
        let cov = (0..nrows)
            .map(|i| (0..ncols).map(|j| dist.cov[(i, j)]).collect())
            .collect();
        PyGaussianDistribution { mean, cov }
    }

    fn observations_to_dmatrices(observations: &[Vec<f64>]) -> Vec<DMatrix<f64>> {
        observations
            .iter()
            .map(|obs| DMatrix::from_vec(obs.len(), 1, obs.clone()))
            .collect()
    }

    #[pyclass]
    #[pyo3(name = "LinearGaussianSSM")]
    struct PyLinearGaussianSSM {
        inner: LinearGaussianStateSpaceModel,
    }

    #[pymethods]
    impl PyLinearGaussianSSM {
        #[new]
        fn new(size_state: usize, size_observation: usize) -> Self {
            Self {
                inner: LinearGaussianStateSpaceModel::new(size_state, size_observation),
            }
        }

        fn forecast(
            &self,
            observations: Vec<Vec<f64>>,
            forecast_steps: usize,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let obs = observations_to_dmatrices(&observations);
            let result = self.inner.forecast(&obs, &forecast_steps);
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }

        fn filter_state(
            &self,
            observations: Vec<Vec<f64>>,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let obs = observations_to_dmatrices(&observations);
            let result = self.inner.filter_state(&obs);
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }

        fn smooth_state(
            &self,
            observations: Vec<Vec<f64>>,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let obs = observations_to_dmatrices(&observations);
            let result = self.inner.smooth_state(&obs);
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }
    }
}

