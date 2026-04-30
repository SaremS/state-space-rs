import numpy as np

from state_space_rs import GaussianDistribution, LinearGaussianSSM


def _expected_num_parameters(size_state: int, size_obs: int) -> int:
    return (
        size_state  # initial mean
        + (size_state * (size_state + 1)) // 2  # initial covariance cholesky params
        + 2 * size_state * size_state  # SchurStableMatrix parameters
        + size_obs * size_state  # observation matrix
        + (size_state * (size_state + 1)) // 2  # process noise covariance params
        + (size_obs * (size_obs + 1)) // 2  # observation noise covariance params
    )


def test_gaussian_distribution_properties_and_log_prob():
    dist = GaussianDistribution(
        np.array([0.0, 0.0], dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )

    np.testing.assert_array_equal(dist.mean, np.zeros(2))
    np.testing.assert_array_equal(dist.cov, np.eye(2))

    log_prob = dist.log_prob(np.array([0.5, -0.25], dtype=np.float64))
    assert isinstance(log_prob, float)
    assert np.isfinite(log_prob)


def test_forecast_empty_observations():
    model = LinearGaussianSSM(2, 2)

    observations = np.empty((0, 2), dtype=np.float64)
    forecast = model.forecast(observations, 3)

    assert len(forecast) == 3
    for dist in forecast:
        assert isinstance(dist.mean, np.ndarray)
        assert dist.mean.shape == (2,)
        assert isinstance(dist.cov, np.ndarray)
        assert dist.cov.shape == (2, 2)


def test_filter_state():
    model = LinearGaussianSSM(2, 2)

    observations = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    filtered = model.filter_state(observations)

    assert len(filtered) == observations.shape[0]
    for state in filtered:
        assert isinstance(state.mean, np.ndarray)
        assert state.mean.shape == (2,)
        assert isinstance(state.cov, np.ndarray)
        assert state.cov.shape == (2, 2)


def test_smooth_state():
    model = LinearGaussianSSM(2, 2)

    observations = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    smoothed = model.smooth_state(observations)

    assert len(smoothed) == observations.shape[0]
    for state in smoothed:
        assert isinstance(state.mean, np.ndarray)
        assert state.mean.shape == (2,)
        assert isinstance(state.cov, np.ndarray)
        assert state.cov.shape == (2, 2)


def test_sample_no_initial_state():
    model = LinearGaussianSSM(2, 2)

    states, observations = model.sample(5)

    assert isinstance(states, np.ndarray)
    assert states.shape == (5, 2)
    assert np.all(np.isfinite(states))

    assert isinstance(observations, np.ndarray)
    assert observations.shape == (5, 2)
    assert np.all(np.isfinite(observations))


def test_sample_with_initial_state_distribution():
    model = LinearGaussianSSM(2, 2)
    initial_state = GaussianDistribution(
        np.array([1.0, 2.0], dtype=np.float64),
        np.eye(2, dtype=np.float64) * 0.5,
    )

    states, observations = model.sample(5, initial_state=initial_state)

    assert isinstance(states, np.ndarray)
    assert states.shape == (5, 2)
    assert np.all(np.isfinite(states))

    assert isinstance(observations, np.ndarray)
    assert observations.shape == (5, 2)
    assert np.all(np.isfinite(observations))


def test_seeded_sample_is_deterministic():
    model = LinearGaussianSSM(size_state=2, size_observation=2)

    states_a, obs_a = model.sample(10, seed=42)
    states_b, obs_b = model.sample(10, seed=42)

    np.testing.assert_array_equal(states_a, states_b)
    np.testing.assert_array_equal(obs_a, obs_b)

    states_c, _ = model.sample(10, seed=99)
    assert not np.array_equal(states_a, states_c)


def test_get_set_parameters_round_trip():
    model = LinearGaussianSSM(2, 2)

    params = model.get_parameters()
    assert isinstance(params, np.ndarray)
    assert len(params) == model.get_num_parameters()

    mutated = params.copy()
    mutated[0] = 1.5
    mutated[-1] = 0.75
    model.set_parameters(mutated)

    np.testing.assert_allclose(model.get_parameters(), mutated)


def test_set_parameters_rejects_wrong_length():
    model = LinearGaussianSSM(2, 2)

    wrong = np.zeros(model.get_num_parameters() - 1, dtype=np.float64)
    try:
        model.set_parameters(wrong)
    except ValueError as exc:
        assert "expected parameter vector" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_get_num_parameters():
    model = LinearGaussianSSM(2, 2)
    assert model.get_num_parameters() == _expected_num_parameters(2, 2)

    model_3_1 = LinearGaussianSSM(3, 1)
    assert model_3_1.get_num_parameters() == _expected_num_parameters(3, 1)


def test_log_likelihood():
    model = LinearGaussianSSM(2, 2)
    observations = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)

    ll = model.log_likelihood(observations)

    assert isinstance(ll, float)
    assert np.isfinite(ll)


def test_gaussian_distribution_log_prob_from_filtered_state():
    model = LinearGaussianSSM(2, 2)
    observations = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    filtered = model.filter_state(observations)
    log_prob = filtered[0].log_prob(np.array([0.5, 0.5], dtype=np.float64))

    assert isinstance(log_prob, float)
    assert np.isfinite(log_prob)
