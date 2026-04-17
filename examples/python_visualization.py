import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

from state_space_rs import LinearGaussianSSM


Z_CRITICAL_90_PERCENT = 1.6448536269514722


def _extract_mean_and_bounds(distributions):
    means = np.array([dist.mean[0] for dist in distributions], dtype=np.float64)
    stds = np.sqrt(np.array([dist.cov[0, 0] for dist in distributions], dtype=np.float64))
    half_width = Z_CRITICAL_90_PERCENT * stds
    return means, means - half_width, means + half_width


def _build_statsmodels(k_states, k_obs, observations):
    """Build a KalmanSmoother with the same defaults as our Rust model."""
    kf = KalmanSmoother(k_endog=k_obs, k_states=k_states)

    kf["design"] = np.eye(k_obs, k_states)          # observation_matrix
    kf["transition"] = np.eye(k_states)              # transition_matrix
    kf["selection"] = np.eye(k_states)               # selection (identity)
    kf["state_cov"] = np.eye(k_states)               # process_noise_cov
    kf["obs_cov"] = np.eye(k_obs)                    # observation_noise_cov

    kf.initialize_known(
        np.zeros(k_states),                           # initial_mean
        np.eye(k_states),                             # initial_cov
    )

    # statsmodels expects shape (k_endog, nobs)
    kf.bind(observations.T.copy(order="C"))
    return kf


def _plot_comparison(
    title,
    rust_means, rust_lower, rust_upper,
    sm_means, sm_lower, sm_upper,
    latent_states, observations,
):
    t = np.arange(latent_states.shape[0])

    fig, (ax_state, ax_obs) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Rust estimates
    ax_state.plot(t, rust_means, color="tab:blue", label="Rust estimate")
    ax_state.fill_between(t, rust_lower, rust_upper, color="tab:blue", alpha=0.15)

    # statsmodels estimates
    ax_state.plot(t, sm_means, color="tab:red", linestyle="--", label="statsmodels estimate")
    ax_state.fill_between(t, sm_lower, sm_upper, color="tab:red", alpha=0.10)

    ax_state.scatter(t, latent_states[:, 0], color="tab:orange", s=20, zorder=5, label="True latent state")
    ax_state.set_ylabel("Latent state")
    ax_state.set_title(title)
    ax_state.legend(loc="best")

    ax_obs.scatter(t, observations[:, 0], color="tab:green", s=20, label="Observed")
    ax_obs.set_xlabel("Time step")
    ax_obs.set_ylabel("Observation")
    ax_obs.legend(loc="best")

    fig.tight_layout()


def main():
    k_states = 1
    k_obs = 1
    model = LinearGaussianSSM(size_state=k_states, size_observation=k_obs)

    latent_states, observations = model.sample(num_observations=50)

    # --- Rust ----------------------------------------------------------
    filtered = model.filter_state(observations)
    smoothed = model.smooth_state(observations)

    rust_filt_mean, rust_filt_lo, rust_filt_hi = _extract_mean_and_bounds(filtered)
    rust_smooth_mean, rust_smooth_lo, rust_smooth_hi = _extract_mean_and_bounds(smoothed)

    # --- statsmodels ---------------------------------------------------
    kf = _build_statsmodels(k_states, k_obs, observations)

    filt_result = kf.filter()
    sm_filt_mean = filt_result.filtered_state[0]
    sm_filt_std = np.sqrt(filt_result.filtered_state_cov[0, 0])
    sm_filt_lo = sm_filt_mean - Z_CRITICAL_90_PERCENT * sm_filt_std
    sm_filt_hi = sm_filt_mean + Z_CRITICAL_90_PERCENT * sm_filt_std

    smooth_result = kf.smooth()
    sm_smooth_mean = smooth_result.smoothed_state[0]
    sm_smooth_std = np.sqrt(smooth_result.smoothed_state_cov[0, 0])
    sm_smooth_lo = sm_smooth_mean - Z_CRITICAL_90_PERCENT * sm_smooth_std
    sm_smooth_hi = sm_smooth_mean + Z_CRITICAL_90_PERCENT * sm_smooth_std

    # --- plots ---------------------------------------------------------
    _plot_comparison(
        "Filtered: Rust vs statsmodels",
        rust_filt_mean, rust_filt_lo, rust_filt_hi,
        sm_filt_mean, sm_filt_lo, sm_filt_hi,
        latent_states, observations,
    )

    _plot_comparison(
        "Smoothed: Rust vs statsmodels",
        rust_smooth_mean, rust_smooth_lo, rust_smooth_hi,
        sm_smooth_mean, sm_smooth_lo, sm_smooth_hi,
        latent_states, observations,
    )

    plt.show()


if __name__ == "__main__":
    main()
