use pyo3::prelude::*;

#[pymodule]
mod state_space_rs {
    use nalgebra::{DMatrix, DVector};
    use numpy::ndarray::{Array1, Array2};
    use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use state_space_core::distributions::{Distribution, GaussianDistribution};
    use state_space_core::state_space_model::{LinearGaussianStateSpaceModel, StateSpaceModel};

    fn anyhow_to_pyerr(err: impl std::fmt::Display) -> PyErr {
        PyValueError::new_err(err.to_string())
    }

    fn py_to_dmatrix(arr: PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
        let a = arr.as_array();
        let (nr, nc) = (a.nrows(), a.ncols());
        let data: Vec<f64> = (0..nr)
            .flat_map(|i| (0..nc).map(move |j| a[[i, j]]))
            .collect();
        DMatrix::from_row_slice(nr, nc, &data)
    }

    fn py_to_dvector(arr: PyReadonlyArray1<'_, f64>) -> PyResult<DVector<f64>> {
        Ok(DVector::from_vec(arr.as_slice()?.to_vec()))
    }

    fn dmatrix_to_py<'py>(py: Python<'py>, mat: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
        let arr = Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(i, j)| mat[(i, j)]);
        PyArray2::from_owned_array(py, arr)
    }

    fn dvector_to_py<'py>(py: Python<'py>, vec: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
        let arr = Array1::from_vec(vec.as_slice().to_vec());
        PyArray1::from_owned_array(py, arr)
    }

    fn observations_to_dmatrices(observations: PyReadonlyArray2<'_, f64>) -> Vec<DMatrix<f64>> {
        let array = observations.as_array();
        let (n_obs, obs_dim) = (array.nrows(), array.ncols());
        (0..n_obs)
            .map(|i| {
                let row: Vec<f64> = (0..obs_dim).map(|j| array[[i, j]]).collect();
                DMatrix::from_vec(obs_dim, 1, row)
            })
            .collect()
    }

    fn py_sequence_to_dmatrices(
        values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<Vec<DMatrix<f64>>>> {
        match values {
            None => Ok(None),
            Some(values) => {
                let arrays = values.extract::<Vec<PyReadonlyArray2<'_, f64>>>()?;
                Ok(Some(arrays.into_iter().map(py_to_dmatrix).collect()))
            }
        }
    }

    #[pyclass(name = "GaussianDistribution")]
    struct PyGaussianDistribution {
        inner: GaussianDistribution,
    }

    #[pymethods]
    impl PyGaussianDistribution {
        #[new]
        fn py_new(mean: PyReadonlyArray1<'_, f64>, cov: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
            let mean = py_to_dvector(mean)?;
            let cov = py_to_dmatrix(cov);
            let inner = GaussianDistribution::new_from_params(mean, cov).map_err(anyhow_to_pyerr)?;
            Ok(Self { inner })
        }

        #[getter]
        fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            dvector_to_py(py, &self.inner.get_mean())
        }

        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            dmatrix_to_py(py, &self.inner.get_cov())
        }

        fn log_prob(&self, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
            self.inner
                .log_prob(&py_to_dvector(x)?)
                .map_err(anyhow_to_pyerr)
        }
    }

    fn gaussian_to_py(dist: GaussianDistribution) -> PyGaussianDistribution {
        PyGaussianDistribution { inner: dist }
    }

    #[pyclass(name = "LinearGaussianSSM")]
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

        fn get_num_parameters(&self) -> usize {
            self.inner.get_num_parameters()
        }

        fn get_parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            let parameters = self.inner.get_parameters();
            dvector_to_py(py, &parameters)
        }

        fn set_parameters(&mut self, params: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
            let params_vec = py_to_dvector(params)?;
            if params_vec.len() != self.inner.get_num_parameters() {
                return Err(PyValueError::new_err(format!(
                    "expected parameter vector of length {}, got {}",
                    self.inner.get_num_parameters(),
                    params_vec.len()
                )));
            }
            self.inner.set_parameters(&params_vec);
            Ok(())
        }

        #[pyo3(signature = (observations, observed_control_variables=None))]
        fn log_likelihood(
            &self,
            observations: PyReadonlyArray2<'_, f64>,
            observed_control_variables: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<f64> {
            let observations = observations_to_dmatrices(observations);
            let observed_control_variables = py_sequence_to_dmatrices(observed_control_variables)?;
            self.inner
                .log_likelihood(&observations, observed_control_variables.as_ref())
                .map_err(anyhow_to_pyerr)
        }

        #[pyo3(signature = (observations, forecast_steps, observed_control_variables=None, forecast_control_variables=None))]
        fn forecast(
            &self,
            observations: PyReadonlyArray2<'_, f64>,
            forecast_steps: usize,
            observed_control_variables: Option<&Bound<'_, PyAny>>,
            forecast_control_variables: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let observations = observations_to_dmatrices(observations);
            let observed_control_variables = py_sequence_to_dmatrices(observed_control_variables)?;
            let forecast_control_variables = py_sequence_to_dmatrices(forecast_control_variables)?;
            let result = self.inner.forecast(
                &observations,
                &forecast_steps,
                observed_control_variables.as_ref(),
                forecast_control_variables.as_ref(),
            );
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }

        #[pyo3(signature = (observations, observed_control_variables=None))]
        fn filter_state(
            &self,
            observations: PyReadonlyArray2<'_, f64>,
            observed_control_variables: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let observations = observations_to_dmatrices(observations);
            let observed_control_variables = py_sequence_to_dmatrices(observed_control_variables)?;
            let result = self
                .inner
                .filter_state(&observations, observed_control_variables.as_ref());
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }

        #[pyo3(signature = (observations, observed_control_variables=None))]
        fn smooth_state(
            &self,
            observations: PyReadonlyArray2<'_, f64>,
            observed_control_variables: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Vec<PyGaussianDistribution>> {
            let observations = observations_to_dmatrices(observations);
            let observed_control_variables = py_sequence_to_dmatrices(observed_control_variables)?;
            let result = self
                .inner
                .smooth_state(&observations, observed_control_variables.as_ref());
            Ok(result.into_iter().map(gaussian_to_py).collect())
        }

        #[pyo3(signature = (num_observations, initial_state=None, observed_control_variables=None, seed=None))]
        fn sample<'py>(
            &self,
            py: Python<'py>,
            num_observations: usize,
            initial_state: Option<PyRef<'py, PyGaussianDistribution>>,
            observed_control_variables: Option<&Bound<'py, PyAny>>,
            seed: Option<u64>,
        ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
            let initial_state = initial_state.map(|state| state.inner.clone());
            let observed_control_variables = py_sequence_to_dmatrices(observed_control_variables)?;
            let (states, observations) = self.inner.sample(
                &num_observations,
                initial_state,
                observed_control_variables.as_ref(),
                seed,
            );

            let state_dim = states.first().map_or(0, |state| state.len());
            let observation_dim = observations.first().map_or(0, |obs| obs.len());

            let states_arr =
                Array2::from_shape_fn((states.len(), state_dim), |(i, j)| states[i][j]);
            let observations_arr = Array2::from_shape_fn(
                (observations.len(), observation_dim),
                |(i, j)| observations[i][j],
            );

            Ok((
                PyArray2::from_owned_array(py, states_arr),
                PyArray2::from_owned_array(py, observations_arr),
            ))
        }
    }
}
