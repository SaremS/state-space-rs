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


def main():
    k_states = 1
    k_obs = 1
    seeds = [42, 123, 7]
    n_obs = 50
    model = LinearGaussianSSM(size_state=k_states, size_observation=k_obs)

    fig, axes = plt.subplots(len(seeds), 2, figsize=(14, 4 * len(seeds)), sharex=True)

    for row, seed in enumerate(seeds):
        _, observations = model.sample(num_observations=n_obs, seed=seed)

        # Rust
        filtered = model.filter_state(observations)
        smoothed = model.smooth_state(observations)
        rust_filt_mean, rust_filt_lo, rust_filt_hi = _extract_mean_and_bounds(filtered)
        rust_smooth_mean, rust_smooth_lo, rust_smooth_hi = _extract_mean_and_bounds(smoothed)

        # statsmodels
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

        t = np.arange(n_obs)

        for col, (title_prefix, r_mean, r_lo, r_hi, s_mean, s_lo, s_hi) in enumerate([
            ("Filtered", rust_filt_mean, rust_filt_lo, rust_filt_hi,
             sm_filt_mean, sm_filt_lo, sm_filt_hi),
            ("Smoothed", rust_smooth_mean, rust_smooth_lo, rust_smooth_hi,
             sm_smooth_mean, sm_smooth_lo, sm_smooth_hi),
        ]):
            ax = axes[row, col]
            ax.plot(t, r_mean, color="tab:blue", label="Rust")
            ax.fill_between(t, r_lo, r_hi, color="tab:blue", alpha=0.15)
            ax.plot(t, s_mean, color="tab:red", linestyle="--", label="statsmodels")
            ax.fill_between(t, s_lo, s_hi, color="tab:red", alpha=0.10)
            ax.set_title(f"{title_prefix} (seed={seed})")
            ax.set_ylabel("Latent state")
            if row == len(seeds) - 1:
                ax.set_xlabel("Time step")
            if row == 0 and col == 0:
                ax.legend(loc="best")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
