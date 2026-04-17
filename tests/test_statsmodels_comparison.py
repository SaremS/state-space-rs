"""
Compare our Rust Kalman filter/smoother against statsmodels.

Observations are generated with numpy (seeded) so both implementations
receive identical data. We test across 3 seeds × 4 sequence lengths.
"""

import numpy as np
import pytest
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

from state_space_rs import LinearGaussianSSM

SEEDS = [42, 123, 7]
NUM_OBS = [10, 50, 100, 1000]
K_STATES = 1
K_OBS = 1

# Tolerances – both implementations solve the same linear algebra,
# so differences should only stem from floating-point ordering.
MEAN_ATOL = 1e-10
COV_ATOL = 1e-10


def _generate_observations(seed: int, n: int) -> np.ndarray:
    """Sample a random-walk + noise process matching our model defaults.

    Model defaults: transition=I, observation=I, process_noise=I,
    obs_noise=I, initial_mean=0, initial_cov=I.
    """
    rng = np.random.RandomState(seed)
    state = rng.multivariate_normal(np.zeros(K_STATES), np.eye(K_STATES))
    observations = np.empty((n, K_OBS), dtype=np.float64)
    for t in range(n):
        state = state + rng.multivariate_normal(np.zeros(K_STATES), np.eye(K_STATES))
        obs = state + rng.multivariate_normal(np.zeros(K_OBS), np.eye(K_OBS))
        observations[t] = obs
    return observations


def _build_statsmodels(observations: np.ndarray) -> KalmanSmoother:
    kf = KalmanSmoother(k_endog=K_OBS, k_states=K_STATES)
    kf["design"] = np.eye(K_OBS, K_STATES)
    kf["transition"] = np.eye(K_STATES)
    kf["selection"] = np.eye(K_STATES)
    kf["state_cov"] = np.eye(K_STATES)
    kf["obs_cov"] = np.eye(K_OBS)
    kf.initialize_known(np.zeros(K_STATES), np.eye(K_STATES))
    kf.bind(observations.T.copy(order="C"))
    return kf


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", NUM_OBS)
def test_filter_matches_statsmodels(seed: int, n: int):
    observations = _generate_observations(seed, n)

    # Rust
    model = LinearGaussianSSM(size_state=K_STATES, size_observation=K_OBS)
    rust_filtered = model.filter_state(observations)
    rust_means = np.array([d.mean for d in rust_filtered])
    rust_covs = np.array([d.cov for d in rust_filtered])

    # statsmodels
    kf = _build_statsmodels(observations)
    result = kf.filter()
    sm_means = result.filtered_state.T  # (nobs, k_states)
    sm_covs = np.moveaxis(result.filtered_state_cov, -1, 0)  # (nobs, k, k)

    np.testing.assert_allclose(rust_means, sm_means, atol=MEAN_ATOL,
                               err_msg=f"Filter means differ (seed={seed}, n={n})")
    np.testing.assert_allclose(rust_covs, sm_covs, atol=COV_ATOL,
                               err_msg=f"Filter covs differ (seed={seed}, n={n})")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", NUM_OBS)
def test_smoother_matches_statsmodels(seed: int, n: int):
    observations = _generate_observations(seed, n)

    # Rust
    model = LinearGaussianSSM(size_state=K_STATES, size_observation=K_OBS)
    rust_smoothed = model.smooth_state(observations)
    rust_means = np.array([d.mean for d in rust_smoothed])
    rust_covs = np.array([d.cov for d in rust_smoothed])

    # statsmodels
    kf = _build_statsmodels(observations)
    result = kf.smooth()
    sm_means = result.smoothed_state.T  # (nobs, k_states)
    sm_covs = np.moveaxis(result.smoothed_state_cov, -1, 0)  # (nobs, k, k)

    np.testing.assert_allclose(rust_means, sm_means, atol=MEAN_ATOL,
                               err_msg=f"Smoother means differ (seed={seed}, n={n})")
    np.testing.assert_allclose(rust_covs, sm_covs, atol=COV_ATOL,
                               err_msg=f"Smoother covs differ (seed={seed}, n={n})")
