import numpy as np
import pytest

from scheduling.simulation import simulate_dynamic, simulate_static


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def base_pga_parameters():
    return {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "memory": 2,
        "epr_pairs": 2,
        "slot_duration": 0.1,
    }


@pytest.fixture
def pga_params():
    return {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "memory": 1,
        "epr_pairs": 1,
        "slot_duration": 0.1,
    }


def _run_static(schedule, period, end_time, base_params, rng):
    return simulate_static(
        schedule=schedule,
        app_specs={"A": {"instances": 1}},
        pga_parameters={"A": dict(base_params)},
        pga_rel_times={"A": 0.0},
        pga_periods={"A": period},
        pga_network_paths={"A": ["Alice", "Bob"]},
        policies={"A": "deadline"},
        rng=rng,
    )


def _run_dynamic(
    app_specs,
    durations,
    pga_params,
    rng,
    max_window_time=20.0,
):
    return simulate_dynamic(
        app_specs=app_specs,
        durations=durations,
        pga_parameters={"A": pga_params},
        pga_rel_times={"A": 0.0},
        pga_network_paths={"A": [["Alice", "Bob"]]},
        rng=rng,
        all_links=[("Alice", "Bob")],
        horizon_time=max_window_time,
    )


def test_simulate_static_success_basic(base_pga_parameters, rng):
    df, _, _, _, _ = _run_static(
        schedule=[("A0", 0.0, 5.0, 10.0)],
        period=10.0,
        end_time=5.0,
        base_params=base_pga_parameters,
        rng=rng,
    )
    assert df.loc[0, "status"] == "completed"


def test_simulate_static_fails_basic(base_pga_parameters, rng):
    df, _, _, _, _ = _run_static(
        schedule=[("A0", 0.0, 0.01, 5.0)],
        period=5.0,
        end_time=0.01,
        base_params=base_pga_parameters,
        rng=rng,
    )
    assert df.loc[0, "status"] == "failed"


def test_simulate_dynamic_basic(rng):
    app_specs = {"A": {"instances": 1, "period": 10.0, "policy": "deadline"}}
    pga_params = {
        "p_gen": 0.99,
        "p_swap": 1.0,
        "memory": 2,
        "epr_pairs": 1,
        "slot_duration": 0.01,
    }
    result = _run_dynamic(app_specs, {"A": 0.5}, pga_params, rng)
    df = result[0]
    assert len(df) > 0


def test_simulate_dynamic_drop_and_defer(rng):
    blocker = {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "p_packet": 0.9,
        "memory": 1,
        "epr_pairs": 1,
        "slot_duration": 10.0,
    }
    shared = {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "p_packet": 0.9,
        "memory": 1,
        "epr_pairs": 1,
        "slot_duration": 1.0,
    }

    df = simulate_dynamic(
        app_specs={
            "A": {"instances": 1, "period": 100.0, "policy": "deadline"},
            "B": {"instances": 1, "period": 6.0, "policy": "deadline"},
            "C": {"instances": 1, "period": 10.0, "policy": "deadline"},
        },
        durations={"A": 20.0, "B": 3.0, "C": 3.0},
        pga_parameters={"A": blocker, "B": shared, "C": shared},
        pga_rel_times={"A": 0.0, "B": 5.0, "C": 5.0},
        pga_network_paths={p: [["Alice", "Bob"]] for p in ("A", "B", "C")},
        rng=rng,
        arrival_rate=None,
        all_links=[("Alice", "Bob")],
        horizon_time=20.0,
    )[0]
    statuses = set(df["status"])
    assert "drop" in statuses
    assert "defer" in statuses


def test_simulate_dynamic_drop_exceeds_period(pga_params, rng):
    app_specs = {"A": {"instances": 2, "period": 1.0, "policy": "deadline"}}
    df = _run_dynamic(app_specs, {"A": 5.0}, pga_params, rng)[0]
    assert "drop" in set(df["status"])


def test_simulate_dynamic_retry(pga_params, rng):
    app_specs = {"A": {"instances": 20, "period": 1.0, "policy": "deadline"}}
    params = {**pga_params, "p_gen": 0.5}
    df = _run_dynamic(app_specs, {"A": 0.1}, params, rng)[0]
    assert "retry" in set(df["status"])
