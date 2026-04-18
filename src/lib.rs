use pyo3::prelude::*;

#[pymodule]
mod state_space_rs {
    use nalgebra::{DMatrix, DVector};
    use numpy::ndarray::{Array1, Array2};
    use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;
    use state_space_core::distributions::GaussianMvDistribution;
    use state_space_core::state_space_model::{
        DifferentiableOnce, DifferentiableTwice, LinearGaussianStateSpaceModel,
        ParameterSet, StateSpaceModel,
    };

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

    fn dmatrix_to_py<'py>(py: Python<'py>, mat: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
        let arr = Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(i, j)| mat[(i, j)]);
        PyArray2::from_owned_array(py, arr)
    }

    fn dvector_to_py<'py>(py: Python<'py>, vec: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
        let arr = Array1::from_vec(vec.as_slice().to_vec());
        PyArray1::from_owned_array(py, arr)
    }

    fn py_to_dmatrix(arr: PyReadonlyArray2<f64>) -> DMatrix<f64> {
        let a = arr.as_array();
        let (nr, nc) = (a.nrows(), a.ncols());
        let data: Vec<f64> = (0..nr)
            .flat_map(|i| (0..nc).map(move |j| a[[i, j]]))
            .collect();
        DMatrix::from_row_slice(nr, nc, &data)
    }

    fn py_to_dvector(arr: PyReadonlyArray1<f64>) -> PyResult<DVector<f64>> {
        Ok(DVector::from_vec(arr.as_slice()?.to_vec()))
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

        #[staticmethod]
        fn calculate_num_parameters(size_state: usize, size_observation: usize) -> usize {
            LinearGaussianStateSpaceModel::calculate_num_parameters(size_state, size_observation)
        }

        #[staticmethod]
        fn from_parameter_vector(
            params: PyReadonlyArray1<f64>,
            size_state: usize,
            size_observation: usize,
        ) -> PyResult<Self> {
            let params_vec = py_to_dvector(params)?;
            Ok(Self {
                inner: LinearGaussianStateSpaceModel::new_from_parameter_vector(
                    &params_vec, size_state, size_observation,
                ),
            })
        }

        // --- Parameter vector accessors (ParameterSet / StateSpaceModel) ---

        fn get_num_parameters(&self) -> usize {
            self.inner.get_num_parameters()
        }

        fn get_parameters_as_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            let params = self.inner.get_parameters_as_vector();
            dvector_to_py(py, &params)
        }

        fn set_parameters_from_vector(&mut self, params: PyReadonlyArray1<f64>) -> PyResult<()> {
            let params_vec = py_to_dvector(params)?;
            self.inner.set_parameters_as_vector(&params_vec);
            Ok(())
        }

        // --- Individual parameter getters ---

        fn get_initial_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            dvector_to_py(py, &self.inner.parameters.get_initial_mean())
        }

        fn get_initial_cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            dmatrix_to_py(py, &self.inner.parameters.get_initial_cov())
        }

        fn get_transition_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            dmatrix_to_py(py, &self.inner.parameters.get_transition_matrix())
        }

        fn get_observation_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            dmatrix_to_py(py, &self.inner.parameters.get_observation_matrix())
        }

        fn get_process_noise_cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            dmatrix_to_py(py, &self.inner.parameters.get_process_noise_cov())
        }

        fn get_observation_noise_cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            dmatrix_to_py(py, &self.inner.parameters.get_observation_noise_cov())
        }

        // --- Individual parameter setters ---

        fn set_initial_mean(&mut self, mean: PyReadonlyArray1<f64>) -> PyResult<()> {
            self.inner.parameters.set_initial_mean(py_to_dvector(mean)?);
            Ok(())
        }

        fn set_initial_cov_dec(&mut self, elements: PyReadonlyArray1<f64>) -> PyResult<()> {
            let v = py_to_dvector(elements)?;
            self.inner.parameters.set_initial_cov_dec(&v);
            Ok(())
        }

        fn set_transition_matrix(&mut self, matrix: PyReadonlyArray2<f64>) -> PyResult<()> {
            self.inner.parameters.set_transition_matrix(py_to_dmatrix(matrix));
            Ok(())
        }

        fn set_observation_matrix(&mut self, matrix: PyReadonlyArray2<f64>) -> PyResult<()> {
            self.inner.parameters.set_observation_matrix(py_to_dmatrix(matrix));
            Ok(())
        }

        fn set_process_noise_cov_dec(&mut self, elements: PyReadonlyArray1<f64>) -> PyResult<()> {
            let v = py_to_dvector(elements)?;
            self.inner.parameters.set_process_noise_cov_dec(&v);
            Ok(())
        }

        fn set_observation_noise_cov_dec(&mut self, elements: PyReadonlyArray1<f64>) -> PyResult<()> {
            let v = py_to_dvector(elements)?;
            self.inner.parameters.set_observation_noise_cov_dec(&v);
            Ok(())
        }

        // --- Differentiable placeholders ---

        fn get_gradient<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            dvector_to_py(py, &self.inner.get_gradient())
        }

        fn get_hessian<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            dmatrix_to_py(py, &self.inner.get_hessian())
        }

        // --- Core model methods ---

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

        #[pyo3(signature = (num_observations, initial_mean=None, initial_cov=None, seed=None))]
        fn sample<'py>(
            &self,
            py: Python<'py>,
            num_observations: usize,
            initial_mean: Option<PyReadonlyArray1<'py, f64>>,
            initial_cov: Option<PyReadonlyArray2<'py, f64>>,
            seed: Option<u64>,
        ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
            let initial_state = match (initial_mean, initial_cov) {
                (Some(mean), Some(cov)) => {
                    let mean_vec = py_to_dvector(mean)?;
                    let cov_mat = py_to_dmatrix(cov);
                    Some(GaussianMvDistribution { mean: mean_vec, cov: cov_mat })
                }
                (None, None) => None,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "initial_mean and initial_cov must both be provided or both be None",
                )),
            };

            let (states, observations) = self.inner.sample(&num_observations, initial_state, seed);

            let state_dim = states.first().map_or(0, |s| s.len());
            let obs_dim = observations.first().map_or(0, |o| o.len());

            let states_arr = Array2::from_shape_fn(
                (states.len(), state_dim),
                |(i, j)| states[i][j],
            );
            let obs_arr = Array2::from_shape_fn(
                (observations.len(), obs_dim),
                |(i, j)| observations[i][j],
            );

            Ok((
                PyArray2::from_owned_array(py, states_arr),
                PyArray2::from_owned_array(py, obs_arr),
            ))
        }
    }
}

