use nalgebra::{DMatrix, DVector, Dynamic, linalg::Cholesky};
use rand::Rng;
use rand::rng;
use rand_distr::{Distribution as Dist, Normal};

use crate::linear_algebra::LowerTriangularMatrix;
use crate::parameter_set::ParameterSet;

pub trait Distribution: Send + Sync {
    type Parameters: ParameterSet;

    fn new() -> Self
    where
        Self: Sized;
    fn new_from_parameter_set(parameter_set: Self::Parameters) -> Self
    where
        Self: Sized;
    fn new_with_dim(dim: usize) -> Self
    where
        Self: Sized;
    fn get_dim(&self) -> usize;
    fn log_prob(&self, x: &DVector<f64>) -> anyhow::Result<f64>;
    fn sample(&self) -> DVector<f64>;
    fn sample_with_rng(&self, rng: &mut dyn Rng) -> DVector<f64>;
    fn get_parameters(&self) -> DVector<f64>;
    fn set_parameters(&mut self, params: &DVector<f64>) -> anyhow::Result<()>;
    fn get_num_parameters(&self) -> usize;
}

#[derive(Clone)]
pub struct GaussianParameterSet {
    pub mean: DVector<f64>,
    pub cov: LowerTriangularMatrix,
}

impl ParameterSet for GaussianParameterSet {
    fn get_parameters(&self) -> DVector<f64> {
        let mut params = self.mean.clone();
        params.extend(
            self.cov
                .get_parameters_as_vector()
                .as_slice()
                .iter()
                .copied(),
        );
        params
    }

    fn set_parameters(&mut self, params: &DVector<f64>) -> anyhow::Result<()> {
        if params.len() != self.get_num_parameters() {
            return Err(anyhow::anyhow!(
                "Parameter vector length does not match mean and covariance size"
            ));
        }

        let mean_len = self.mean.len();
        self.mean = params.rows(0, mean_len).into();
        let cov_flat = params.rows(mean_len, params.len() - mean_len).into_owned();

        let result = self.cov.set_parameters_from_vector(&cov_flat);

        result
    }

    fn get_num_parameters(&self) -> usize {
        self.mean.len() + self.cov.get_num_parameters()
    }
}

#[derive(Clone)]
pub struct GaussianDistribution {
    parameter_set: GaussianParameterSet,
}

impl GaussianDistribution {
    fn sample_impl(&self, rng: &mut dyn Rng) -> DVector<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z: DVector<f64> =
            DVector::from_fn(self.parameter_set.mean.len(), |_, _| normal.sample(rng));
        let cov_lower = self.parameter_set.cov.to_dense();
        &self.parameter_set.mean + cov_lower * z
    }

    pub fn new_from_params(mean: DVector<f64>, cov: DMatrix<f64>) -> anyhow::Result<Self> {
        let cov_chol = LowerTriangularMatrix::new_from_posdef_dense(cov.clone()).map_err(|e| {
            anyhow::anyhow!(
                "Failed to create GaussianDistribution from mean and covariance: {}",
                e
            )
        })?;

        let parameter_set = GaussianParameterSet {
            mean,
            cov: cov_chol,
        };

        Ok(Self { parameter_set })
    }

    pub fn new_from_params_cholesky(
        mean: DVector<f64>,
        cov_cholesky: Cholesky<f64, Dynamic>,
    ) -> Self {
        let cov_chol = LowerTriangularMatrix::new_from_cholesky(cov_cholesky);

        let parameter_set = GaussianParameterSet {
            mean,
            cov: cov_chol,
        };

        Self { parameter_set }
    }

    pub fn get_mean(&self) -> DVector<f64> {
        self.parameter_set.mean.clone()
    }

    pub fn get_cov(&self) -> DMatrix<f64> {
        let cov_chol = self.parameter_set.cov.get_cholesky_representation();
        cov_chol.l().transpose() * cov_chol.l()
    }

    pub fn get_cov_cholesky(&self) -> Cholesky<f64, Dynamic> {
        self.parameter_set.cov.get_cholesky_representation().clone()
    }
}

impl Distribution for GaussianDistribution {
    type Parameters = GaussianParameterSet;

    fn new() -> Self {
        let parameter_set = GaussianParameterSet {
            mean: DVector::zeros(1),
            cov: LowerTriangularMatrix::new(1),
        };

        Self { parameter_set }
    }

    fn new_from_parameter_set(parameter_set: GaussianParameterSet) -> Self {
        Self { parameter_set }
    }

    fn new_with_dim(dim: usize) -> Self {
        let parameter_set = GaussianParameterSet {
            mean: DVector::zeros(dim),
            cov: LowerTriangularMatrix::new(dim),
        };

        Self { parameter_set }
    }

    fn get_dim(&self) -> usize {
        self.parameter_set.mean.len()
    }

    fn log_prob(&self, x: &DVector<f64>) -> anyhow::Result<f64> {
        let mean = &self.parameter_set.mean;
        let d = mean.len() as f64;

        let l_matrix = self.parameter_set.cov.get_diagonal();

        let log_det_cov: f64 = l_matrix.iter().map(|val| val.ln()).sum::<f64>() * 2.0;

        let diff = x - mean;

        let cov_L = self.parameter_set.cov.get_cholesky_representation();
        let precision_times_diff = cov_L.solve(&diff);
        let mahalanobis_squared = diff.dot(&precision_times_diff);

        let result = -0.5 * mahalanobis_squared
            - 0.5 * log_det_cov
            - 0.5 * d * (2.0 * std::f64::consts::PI).ln();

        Ok(result)
    }

    fn sample(&self) -> DVector<f64> {
        self.sample_impl(&mut rng())
    }

    fn set_parameters(&mut self, params: &DVector<f64>) -> anyhow::Result<()> {
        if params.len() != self.parameter_set.get_num_parameters() {
            return Err(anyhow::anyhow!(
                "Parameter vector length does not match mean and covariance size"
            ));
        }

        let result = self.parameter_set.set_parameters(params);

        result
    }

    fn get_parameters(&self) -> DVector<f64> {
        self.parameter_set.get_parameters()
    }

    fn get_num_parameters(&self) -> usize {
        self.parameter_set.get_num_parameters()
    }

    fn sample_with_rng(&self, rng: &mut dyn Rng) -> DVector<f64> {
        self.sample_impl(rng)
    }
}

#[derive(Clone)]
pub struct CenteredGaussianParameterSet {
    pub cov: LowerTriangularMatrix,
}

impl ParameterSet for CenteredGaussianParameterSet {
    fn get_parameters(&self) -> DVector<f64> {
        let params = self.cov.get_parameters_as_vector().clone();
        params
    }

    fn set_parameters(&mut self, params: &DVector<f64>) -> anyhow::Result<()> {
        if params.len() != self.get_num_parameters() {
            return Err(anyhow::anyhow!(
                "Parameter vector length does not match covariance size"
            ));
        }

        let result = self.cov.set_parameters_from_vector(params);

        result
    }

    fn get_num_parameters(&self) -> usize {
        self.cov.get_num_parameters()
    }
}

#[derive(Clone)]
pub struct CenteredGaussianDistribution {
    parameter_set: CenteredGaussianParameterSet,
}

impl CenteredGaussianDistribution {
    fn sample_impl(&self, rng: &mut dyn Rng) -> DVector<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let d = self.parameter_set.cov.get_size();
        let z: DVector<f64> = DVector::from_fn(d, |_, _| normal.sample(rng));
        let cov_lower = self.parameter_set.cov.to_dense();
        cov_lower * z
    }

    pub fn get_mean(&self) -> DVector<f64> {
        let dim = self.parameter_set.cov.get_size();

        DVector::<f64>::zeros(dim)
    }

    pub fn get_cov(&self) -> DMatrix<f64> {
        let cov_chol = self.parameter_set.cov.get_cholesky_representation();
        cov_chol.l().transpose() * cov_chol.l()
    }

    pub fn get_cov_cholesky(&self) -> Cholesky<f64, Dynamic> {
        self.parameter_set.cov.get_cholesky_representation().clone()
    }
}

impl Distribution for CenteredGaussianDistribution {
    type Parameters = CenteredGaussianParameterSet;

    fn new() -> Self {
        let parameter_set = CenteredGaussianParameterSet {
            cov: LowerTriangularMatrix::new(1),
        };

        Self { parameter_set }
    }

    fn new_from_parameter_set(parameter_set: CenteredGaussianParameterSet) -> Self {
        Self { parameter_set }
    }

    fn new_with_dim(dim: usize) -> Self {
        let parameter_set = CenteredGaussianParameterSet {
            cov: LowerTriangularMatrix::new(dim),
        };

        Self { parameter_set }
    }

    fn get_dim(&self) -> usize {
        self.parameter_set.cov.get_size()
    }

    fn log_prob(&self, x: &DVector<f64>) -> anyhow::Result<f64> {
        let d = self.get_dim() as f64;

        let l_matrix = self.parameter_set.cov.get_diagonal();

        let log_det_cov: f64 = l_matrix.iter().map(|val| val.ln()).sum::<f64>() * 2.0;

        let cov_L = self.parameter_set.cov.get_cholesky_representation();
        let precision_times_diff = cov_L.solve(x);
        let mahalanobis_squared = x.dot(&precision_times_diff);

        let result = -0.5 * mahalanobis_squared
            - 0.5 * log_det_cov
            - 0.5 * d * (2.0 * std::f64::consts::PI).ln();

        Ok(result)
    }

    fn sample(&self) -> DVector<f64> {
        self.sample_impl(&mut rng())
    }

    fn set_parameters(&mut self, params: &DVector<f64>) -> anyhow::Result<()> {
        if params.len() != self.parameter_set.get_num_parameters() {
            return Err(anyhow::anyhow!(
                "Parameter vector length does not match mean and covariance size"
            ));
        }

        let result = self.parameter_set.set_parameters(params);
        result
    }

    fn get_parameters(&self) -> DVector<f64> {
        self.parameter_set.get_parameters()
    }

    fn get_num_parameters(&self) -> usize {
        self.parameter_set.get_num_parameters()
    }

    fn sample_with_rng(&self, rng: &mut dyn Rng) -> DVector<f64> {
        self.sample_impl(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_gaussian_is_send_sync() {
        _assert_send_sync::<GaussianDistribution>();
    }

    #[test]
    fn test_gaussian_log_prob() {
        let parameter_set = GaussianParameterSet {
            mean: DVector::from_vec(vec![0.0, 0.0]),
            cov: LowerTriangularMatrix::new_with_values(2, 1.0, 0.0),
        };

        let dist = GaussianDistribution::new_from_parameter_set(parameter_set);

        let x = DVector::from_vec(vec![1.0, 1.0]);
        let log_prob = dist.log_prob(&x).unwrap();
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_gaussian_sample() {
        let parameter_set = GaussianParameterSet {
            mean: DVector::from_vec(vec![0.0, 0.0]),
            cov: LowerTriangularMatrix::new_with_values(2, 1.0, 0.0),
        };
        let dist = GaussianDistribution::new_from_parameter_set(parameter_set);

        let sample = dist.sample();
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_centered_gaussian_is_send_sync() {
        _assert_send_sync::<GaussianDistribution>();
    }

    #[test]
    fn test_centered_gaussian_log_prob() {
        let parameter_set = CenteredGaussianParameterSet {
            cov: LowerTriangularMatrix::new_with_values(2, 1.0, 0.0),
        };

        let dist = CenteredGaussianDistribution::new_from_parameter_set(parameter_set);

        let x = DVector::from_vec(vec![1.0, 1.0]);
        let log_prob = dist.log_prob(&x).unwrap();
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_centered_gaussian_sample() {
        let parameter_set = CenteredGaussianParameterSet {
            cov: LowerTriangularMatrix::new_with_values(2, 1.0, 0.0),
        };
        let dist = CenteredGaussianDistribution::new_from_parameter_set(parameter_set);

        let sample = dist.sample();
        assert_eq!(sample.len(), 2);
    }
}
