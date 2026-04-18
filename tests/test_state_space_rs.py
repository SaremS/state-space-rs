import numpy as np
import pytest
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


def test_sample_no_initial_state():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    num_observations = 5
    states, observations = model.sample(num_observations)

    assert isinstance(states, np.ndarray)
    assert states.shape == (num_observations, size_state)
    assert np.all(np.isfinite(states))

    assert isinstance(observations, np.ndarray)
    assert observations.shape == (num_observations, size_obs)
    assert np.all(np.isfinite(observations))


def test_sample_with_initial_state():
    size_state = 2
    size_obs = 2
    model = LinearGaussianSSM(size_state, size_obs)

    initial_mean = np.array([1.0, 2.0])
    initial_cov = np.eye(size_state) * 0.5

    num_observations = 5
    states, observations = model.sample(
        num_observations,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    assert isinstance(states, np.ndarray)
    assert states.shape == (num_observations, size_state)
    assert np.all(np.isfinite(states))

    assert isinstance(observations, np.ndarray)
    assert observations.shape == (num_observations, size_obs)
    assert np.all(np.isfinite(observations))


def test_seeded_sample_is_deterministic():
    model = LinearGaussianSSM(size_state=2, size_observation=2)

    states_a, obs_a = model.sample(10, seed=42)
    states_b, obs_b = model.sample(10, seed=42)

    np.testing.assert_array_equal(states_a, states_b)
    np.testing.assert_array_equal(obs_a, obs_b)

    # Different seed gives different results
    states_c, _ = model.sample(10, seed=99)
    assert not np.array_equal(states_a, states_c)


# --- Parameter getter/setter tests ---


def test_get_set_initial_mean():
    model = LinearGaussianSSM(2, 2)
    default_mean = model.get_initial_mean()
    np.testing.assert_array_equal(default_mean, np.zeros(2))

    model.set_initial_mean(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(model.get_initial_mean(), np.array([1.0, 2.0]))


def test_get_set_initial_cov_dec():
    model = LinearGaussianSSM(2, 2)
    # Default initial_cov should be identity (L=I, cov=I*I^T=I)
    default_cov = model.get_initial_cov()
    np.testing.assert_array_almost_equal(default_cov, np.eye(2))

    # Set lower-triangular decomposition: [diag0, diag1, lower01]
    model.set_initial_cov_dec(np.array([2.0, 3.0, 0.5]))
    cov = model.get_initial_cov()
    assert cov.shape == (2, 2)
    # L = [[2, 0], [0.5, 3]], cov = L @ L.T
    L = np.array([[2.0, 0.0], [0.5, 3.0]])
    np.testing.assert_array_almost_equal(cov, L @ L.T)


def test_get_set_transition_matrix():
    model = LinearGaussianSSM(2, 2)
    default_tm = model.get_transition_matrix()
    np.testing.assert_array_almost_equal(default_tm, np.eye(2))

    new_tm = np.array([[0.9, 0.1], [0.0, 0.8]])
    model.set_transition_matrix(new_tm)
    np.testing.assert_array_almost_equal(model.get_transition_matrix(), new_tm)


def test_get_set_observation_matrix():
    model = LinearGaussianSSM(2, 2)
    default_om = model.get_observation_matrix()
    np.testing.assert_array_almost_equal(default_om, np.eye(2))

    new_om = np.array([[1.0, 0.5], [0.0, 1.0]])
    model.set_observation_matrix(new_om)
    np.testing.assert_array_almost_equal(model.get_observation_matrix(), new_om)


def test_get_set_process_noise_cov_dec():
    model = LinearGaussianSSM(2, 2)
    default_q = model.get_process_noise_cov()
    np.testing.assert_array_almost_equal(default_q, np.eye(2))

    model.set_process_noise_cov_dec(np.array([0.5, 0.5, 0.1]))
    q = model.get_process_noise_cov()
    assert q.shape == (2, 2)
    L = np.array([[0.5, 0.0], [0.1, 0.5]])
    np.testing.assert_array_almost_equal(q, L @ L.T)


def test_get_set_observation_noise_cov_dec():
    model = LinearGaussianSSM(2, 2)
    default_r = model.get_observation_noise_cov()
    np.testing.assert_array_almost_equal(default_r, np.eye(2))

    model.set_observation_noise_cov_dec(np.array([0.3, 0.3, 0.0]))
    r = model.get_observation_noise_cov()
    assert r.shape == (2, 2)
    L = np.array([[0.3, 0.0], [0.0, 0.3]])
    np.testing.assert_array_almost_equal(r, L @ L.T)


# --- Parameter vector round-trip ---


def test_get_set_parameters_as_vector():
    model = LinearGaussianSSM(2, 2)
    model.set_initial_mean(np.array([1.0, 2.0]))
    model.set_transition_matrix(np.array([[0.9, 0.1], [0.0, 0.8]]))

    params = model.get_parameters_as_vector()
    assert isinstance(params, np.ndarray)
    assert len(params) == model.get_num_parameters()

    # Round-trip: create new model and set same params
    model2 = LinearGaussianSSM(2, 2)
    model2.set_parameters_from_vector(params)
    params2 = model2.get_parameters_as_vector()
    np.testing.assert_array_almost_equal(params, params2)


def test_get_num_parameters():
    model = LinearGaussianSSM(2, 2)
    expected = (
        2           # initial_mean
        + 3         # initial_cov_dec (2 + 2*1/2)
        + 4         # transition_matrix (2x2)
        + 4         # observation_matrix (2x2)
        + 3         # process_noise_cov_dec
        + 3         # observation_noise_cov_dec
    )
    assert model.get_num_parameters() == expected


def test_calculate_num_parameters_static():
    expected = (
        2 + 3 + 4 + 4 + 3 + 3
    )
    assert LinearGaussianSSM.calculate_num_parameters(2, 2) == expected

    # 3-state, 1-obs
    expected_3_1 = (
        3           # initial_mean
        + 6         # initial_cov_dec (3 + 3)
        + 9         # transition_matrix (3x3)
        + 3         # observation_matrix (1x3)
        + 6         # process_noise_cov_dec
        + 1         # observation_noise_cov_dec
    )
    assert LinearGaussianSSM.calculate_num_parameters(3, 1) == expected_3_1


def test_from_parameter_vector():
    model = LinearGaussianSSM(2, 2)
    model.set_initial_mean(np.array([1.0, 2.0]))
    params = model.get_parameters_as_vector()

    model2 = LinearGaussianSSM.from_parameter_vector(params, 2, 2)
    params2 = model2.get_parameters_as_vector()
    np.testing.assert_array_almost_equal(params, params2)


# --- Differentiable placeholders ---


def test_get_gradient():
    model = LinearGaussianSSM(2, 2)
    grad = model.get_gradient()
    assert isinstance(grad, np.ndarray)
    assert len(grad) == 1
    assert grad[0] == 0.0


def test_get_hessian():
    model = LinearGaussianSSM(2, 2)
    hess = model.get_hessian()
    assert isinstance(hess, np.ndarray)
    assert hess.shape == (1, 1)
    assert hess[0, 0] == 0.0


# --- Edge case: mismatched initial_mean/initial_cov ---


def test_sample_mismatched_initial_raises():
    model = LinearGaussianSSM(2, 2)
    with pytest.raises(ValueError, match="both be provided or both be None"):
        model.sample(5, initial_mean=np.array([1.0, 2.0]))
