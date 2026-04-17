import numpy as np
from state_space_rs import LinearGaussianSSM


def test_forecast_empty_observations():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    observations = np.empty((0, size_obs), dtype=np.float64)
    forecast = model.forecast(observations, 3)

    assert len(forecast) == 3
    for dist in forecast:
        assert isinstance(dist.mean, np.ndarray)
        assert dist.mean.shape == (size_obs,)
        assert isinstance(dist.cov, np.ndarray)
        assert dist.cov.shape == (size_obs, size_obs)


def test_filter_state():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    observations = np.array([[1.0, 0.0], [0.0, 1.0]])
    filtered = model.filter_state(observations)

    assert len(filtered) == observations.shape[0]
    for state in filtered:
        assert isinstance(state.mean, np.ndarray)
        assert state.mean.shape == (size_state,)
        assert isinstance(state.cov, np.ndarray)
        assert state.cov.shape == (size_state, size_state)


def test_smooth_state():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    observations = np.array([[1.0, 0.0], [0.0, 1.0]])
    filtered = model.smooth_state(observations)

    assert len(filtered) == observations.shape[0]
    for state in filtered:
        assert isinstance(state.mean, np.ndarray)
        assert state.mean.shape == (size_state,)
        assert isinstance(state.cov, np.ndarray)
        assert state.cov.shape == (size_state, size_state)
