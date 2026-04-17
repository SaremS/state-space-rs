use pyo3::prelude::*;

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
}
