import state_space_rs


def test_sum_as_string():
    assert state_space_rs.sum_as_string(2, 3) == "5"
    assert state_space_rs.sum_as_string(0, 0) == "0"


def test_add():
    assert state_space_rs.add(2, 3) == 5
    assert state_space_rs.add(-1, 1) == 0
    assert state_space_rs.add(-5, -3) == -8
