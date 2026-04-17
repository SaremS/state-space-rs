use pyo3::prelude::*;

use crate::state_space_model::*;

#[pymodule]
mod state_space_rs {
    use pyo3::prelude::*;

    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pyfunction]
    fn add(a: i64, b: i64) -> PyResult<i64> {
        Ok(a + b)
    }
    
    #[pyfunction]
    fn create_linear_gaussian_state_space_model(size_state: usize, size_observation: usize) -> PyResult<LinearGaussianStateSpaceModel> {
        Ok(LinearGaussianStateSpaceModel::new(size_state, size_observation))
    }
}

