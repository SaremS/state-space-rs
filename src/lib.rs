use pyo3::prelude::*;

#[pymodule]
mod state_space_rs {
    use nalgebra::DMatrix;
    use numpy::ndarray::Array2;
    use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
    use pyo3::prelude::*;
    use state_space_core::distributions::GaussianMvDistribution;
    use state_space_core::state_space_model::{LinearGaussianStateSpaceModel, StateSpaceModel};

    #[pyclass]
    #[pyo3(name = "GaussianDistribution")]
    struct PyGaussianDistribution {
        mean_data: Vec<f64>,
        cov_data: Vec<f64>,
        cov_rows: usize,
        cov_cols: usize,
    }

    #[pymethods]
    impl PyGaussianDistribution {
        #[getter]
        fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            PyArray1::from_slice(py, &self.mean_data)
        }

        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            let arr = Array2::from_shape_fn((self.cov_rows, self.cov_cols), |(i, j)| {
                self.cov_data[i * self.cov_cols + j]
            });
            PyArray2::from_owned_array(py, arr)
        }
    }

    fn gaussian_to_py(dist: GaussianMvDistribution) -> PyGaussianDistribution {
        let mean_data = dist.mean.as_slice().to_vec();
        let nrows = dist.cov.nrows();
        let ncols = dist.cov.ncols();
        let mut cov_data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                cov_data.push(dist.cov[(i, j)]);
            }
        }
        PyGaussianDistribution {
            mean_data,
            cov_data,
            cov_rows: nrows,
            cov_cols: ncols,
        }
    }

    fn observations_to_dmatrices(obs: PyReadonlyArray2<f64>) -> Vec<DMatrix<f64>> {
        let array = obs.as_array();
        let (n_obs, obs_dim) = (array.nrows(), array.ncols());
        (0..n_obs)
            .map(|i| {
                let row: Vec<f64> = (0..obs_dim).map(|j| array[[i, j]]).collect();
                DMatrix::from_vec(obs_dim, 1, row)
            })
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
            observations: PyReadonlyArray2<f64>,
            forecast_steps: usize,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let obs = observations_to_dmatrices(observations);
            let result = self.inner.forecast(&obs, &forecast_steps);
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }

        fn filter_state(
            &self,
            observations: PyReadonlyArray2<f64>,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let obs = observations_to_dmatrices(observations);
            let result = self.inner.filter_state(&obs);
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }

        fn smooth_state(
            &self,
            observations: PyReadonlyArray2<f64>,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let obs = observations_to_dmatrices(observations);
            let result = self.inner.smooth_state(&obs);
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }
    }
}

