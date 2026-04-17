import state_space_rs
from state_space_rs import LinearGaussianSSM


def test_forecast_empty_observations():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    forecast = model.forecast([], 3)

    assert len(forecast) == 3
    for dist in forecast:
        assert len(dist.mean) == size_obs
        assert len(dist.cov) == size_obs
        assert len(dist.cov[0]) == size_obs


def test_filter_state():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    observations = [[1.0, 0.0], [0.0, 1.0]]
    filtered = model.filter_state(observations)

    assert len(filtered) == len(observations)
    for state in filtered:
        assert len(state.mean) == size_state
        assert len(state.cov) == size_state
        assert len(state.cov[0]) == size_state


def test_smooth_state():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    observations = [[1.0, 0.0], [0.0, 1.0]]
    filtered = model.smooth_state(observations)

    assert len(filtered) == len(observations)
    for state in filtered:
        assert len(state.mean) == size_state
        assert len(state.cov) == size_state
        assert len(state.cov[0]) == size_state
