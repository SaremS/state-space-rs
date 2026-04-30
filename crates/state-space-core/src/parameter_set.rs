use nalgebra::DVector;

pub trait ParameterSet {
    fn get_parameters(&self) -> DVector<f64>;
    fn set_parameters(&mut self, params: &DVector<f64>) -> anyhow::Result<()>;
    fn get_num_parameters(&self) -> usize;
}
