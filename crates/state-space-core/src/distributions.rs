use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::rng;
use rand_distr::{Distribution as Dist, Normal};

pub trait Distribution: Send + Sync {
    fn log_prob(&self, x: &DVector<f64>) -> anyhow::Result<f64>;
    fn sample(&self) -> DVector<f64>;
    fn sample_with_rng(&self, rng: &mut dyn Rng) -> DVector<f64>;
}

#[derive(Clone)]
pub struct GaussianDistribution {
    pub mean: DVector<f64>,
    pub cov: DMatrix<f64>,
}

impl GaussianDistribution {
    fn sample_impl(&self, rng: &mut dyn Rng) -> DVector<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z: DVector<f64> = DVector::from_fn(self.mean.len(), |_, _| normal.sample(rng));
        let cov_cholesky = self.cov.clone().cholesky().unwrap();
        &self.mean + cov_cholesky.l() * z
    }
}

impl Distribution for GaussianDistribution {
    fn log_prob(&self, x: &DVector<f64>) -> anyhow::Result<f64> {
        let d = self.mean.len() as f64;
        let cov_inv = self.cov.clone().try_inverse().ok_or_else(|| anyhow::anyhow!("Covariance matrix is not invertible"))?;
        let diff = x - &self.mean;
        let exponent = -0.5 * diff.transpose() * cov_inv * diff;
        let log_det_cov = self.cov.determinant().ln();

        let result = exponent[(0, 0)] - 0.5 * log_det_cov - 0.5 * d * (2.0 * std::f64::consts::PI).ln();
        Ok(result)
    }

    fn sample(&self) -> DVector<f64> {
        self.sample_impl(&mut rng())
    }

    fn sample_with_rng(&self, rng: &mut dyn Rng) -> DVector<f64> {
        self.sample_impl(rng)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_gaussian_is_send_sync() {
        _assert_send_sync::<GaussianDistribution>();
    }

    #[test]
    fn test_gaussian_log_prob() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::identity(2, 2);
        let dist = GaussianDistribution { mean, cov };

        let x = DVector::from_vec(vec![1.0, 1.0]);
        let log_prob = dist.log_prob(&x).unwrap();
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_gaussian_sample() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::identity(2, 2);
        let dist = GaussianDistribution { mean, cov };

        let sample = dist.sample();
        assert_eq!(sample.len(), 2);
    }
}
