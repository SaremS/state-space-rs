import numpy as np
import matplotlib.pyplot as plt

from state_space_rs import LinearGaussianSSM


Z_90 = 1.6448536269514722


def _extract_mean_and_bounds(distributions):
    means = np.array([dist.mean[0] for dist in distributions], dtype=np.float64)
    stds = np.sqrt(np.array([dist.cov[0, 0] for dist in distributions], dtype=np.float64))
    half_width = Z_90 * stds
    return means, means - half_width, means + half_width


def _plot_estimates(title, means, lower, upper, latent_states, observations):
    t = np.arange(latent_states.shape[0])

    fig, (ax_state, ax_obs) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_state.plot(t, means, color="tab:blue", label="Estimated latent mean")
    ax_state.fill_between(t, lower, upper, color="tab:blue", alpha=0.2, label="90% interval")
    ax_state.scatter(t, latent_states[:, 0], color="tab:orange", s=20, label="True latent state")
    ax_state.set_ylabel("Latent state")
    ax_state.set_title(title)
    ax_state.legend(loc="best")

    ax_obs.scatter(t, observations[:, 0], color="tab:green", s=20, label="Observed state")
    ax_obs.set_xlabel("Time step")
    ax_obs.set_ylabel("Observation")
    ax_obs.legend(loc="best")

    fig.tight_layout()


def main():
    model = LinearGaussianSSM(size_state=1, size_observation=1)

    latent_states, observations = model.sample(num_observations=50)

    filtered = model.filter_state(observations)
    smoothed = model.smooth_state(observations)

    filtered_mean, filtered_lower, filtered_upper = _extract_mean_and_bounds(filtered)
    smoothed_mean, smoothed_lower, smoothed_upper = _extract_mean_and_bounds(smoothed)

    _plot_estimates(
        "Filtered latent state distribution",
        filtered_mean,
        filtered_lower,
        filtered_upper,
        latent_states,
        observations,
    )

    _plot_estimates(
        "Smoothed latent state distribution",
        smoothed_mean,
        smoothed_lower,
        smoothed_upper,
        latent_states,
        observations,
    )

    plt.show()


if __name__ == "__main__":
    main()
