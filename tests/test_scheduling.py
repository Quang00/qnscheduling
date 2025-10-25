import pytest

from scheduling.scheduling import edf_parallel


@pytest.fixture
def patch_hyperperiod(monkeypatch):

    def _patch(value):
        monkeypatch.setitem(
            edf_parallel.__globals__, "hyperperiod", lambda _: value
        )

    return _patch


def test_invalid_horizon_hp_0(patch_hyperperiod):
    patch_hyperperiod(0.0)
    is_feasible, _ = edf_parallel(
        {"A": 0}, {"A": 10}, {"A": 1}, {"A": set()}, horizon_cycles=1
    )
    assert is_feasible is False


def test_invalid_horizon_0(patch_hyperperiod):
    patch_hyperperiod(10.0)
    is_feasible, _ = edf_parallel(
        {"A": 0}, {"A": 10}, {"A": 1}, {"A": set()}, horizon_cycles=0
    )
    assert is_feasible is False


def test_utilization_gt_1(patch_hyperperiod):
    patch_hyperperiod(10.0)
    rel = {"A": 0.0}
    periods = {"A": 10.0}
    durations = {"A": 11.0}
    parallel = {}
    is_feasible, _ = edf_parallel(rel, periods, durations, parallel, 1)
    assert is_feasible is False


def test_schedule_single_pga(patch_hyperperiod):
    patch_hyperperiod(10.0)
    rel = {"A": 0.0}
    periods = {"A": 10.0}
    durations = {"A": 10.0}
    parallel = {"A": set()}
    is_feasible, schedule = edf_parallel(
        rel, periods, durations, parallel, horizon_cycles=1
    )
    assert is_feasible
    assert len(schedule) == 1
    name, start, end, dl = schedule[0]
    assert name == "A0"
    assert start == 0.0
    assert end == 10.0
    assert dl == 10.0
    assert end == dl


def test_schedule_two_pgas_parallel(patch_hyperperiod):
    patch_hyperperiod(10.0)
    rel = {"A": 0.0, "B": 0.0}
    periods = {"A": 10.0, "B": 10.0}
    durations = {"A": 5.0, "B": 5.0}
    parallel = {"A": {"B"}, "B": {"A"}}
    is_feasible, schedule = edf_parallel(
        rel, periods, durations, parallel, horizon_cycles=1
    )
    assert is_feasible
    assert {s[0] for s in schedule} == {"A0", "B0"}
    times = {name: (start, end) for name, start, end, _ in schedule}
    assert times["A0"] == (0.0, 5.0)
    assert times["B0"] == (0.0, 5.0)


def test_deadline_miss(patch_hyperperiod):
    patch_hyperperiod(10.0)
    rel = {"A": 0.0, "B": 0.0}
    periods = {"A": 10.0, "B": 10.0}
    durations = {"A": 10.000001, "B": 10.0}
    parallel = {"A": set(), "B": set()}
    is_feasible, _ = edf_parallel(rel, periods, durations, parallel, 1)
    assert is_feasible is False
