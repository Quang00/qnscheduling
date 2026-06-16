import numpy as np
import pytest

from scheduling.simulation import simulate_dynamic


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def base_pga_parameters():
    return {
        "p_swap": 1.0,
        "memory": 2,
        "epr_pairs": 2,
        "slot_duration": 0.1,
    }


@pytest.fixture
def pga_params():
    return {
        "p_swap": 1.0,
        "memory": 1,
        "epr_pairs": 1,
        "slot_duration": 0.1,
    }


def _run_dynamic(
    app_specs,
    durations,
    pga_params,
    rng,
    max_window_time=20.0,
    rates=None,
):
    if rates is None:
        rates = {("Alice", "Bob"): 1.0}
    return simulate_dynamic(
        app_specs=app_specs,
        durations=durations,
        pga_parameters={"A": pga_params},
        pga_rel_times={"A": 0.0},
        pga_network_paths={"A": [["Alice", "Bob"]]},
        rng=rng,
        all_links=[("Alice", "Bob")],
        horizon_time=max_window_time,
        rates=rates,
    )


def test_simulate_dynamic_basic(rng):
    app_specs = {
        "A": {"instances": 1, "deadline_budget": 10.0}
    }
    pga_params = {
        "p_swap": 1.0,
        "memory": 2,
        "epr_pairs": 1,
        "slot_duration": 0.01,
    }
    result = _run_dynamic(
        app_specs,
        {"A": 0.5},
        pga_params,
        rng,
        rates={("Alice", "Bob"): 0.99},
    )
    df = result[0]
    assert len(df) > 0


def test_simulate_dynamic_drop_and_defer(rng):
    blocker = {
        "p_swap": 1.0,
        "p_packet": 0.9,
        "memory": 1,
        "epr_pairs": 1,
        "slot_duration": 10.0,
    }
    shared = {
        "p_swap": 1.0,
        "p_packet": 0.9,
        "memory": 1,
        "epr_pairs": 1,
        "slot_duration": 1.0,
    }

    df = simulate_dynamic(
        app_specs={
            "A": {"instances": 1, "deadline_budget": 100.0},
            "B": {"instances": 1, "deadline_budget": 6.0},
            "C": {"instances": 1, "deadline_budget": 10.0},
        },
        durations={"A": 20.0, "B": 3.0, "C": 3.0},
        pga_parameters={"A": blocker, "B": shared, "C": shared},
        pga_rel_times={"A": 0.0, "B": 5.0, "C": 5.0},
        pga_network_paths={p: [["Alice", "Bob"]] for p in ("A", "B", "C")},
        rng=rng,
        all_links=[("Alice", "Bob")],
        horizon_time=20.0,
        rates={("Alice", "Bob"): 1.0},
    )[0]
    statuses = set(df["status"])
    assert "drop" in statuses


def test_simulate_dynamic_drop_exceeds_deadline(pga_params, rng):
    app_specs = {
        "A": {"instances": 2, "deadline_budget": 1.0}
    }
    df = _run_dynamic(app_specs, {"A": 5.0}, pga_params, rng)[0]
    assert "drop" in set(df["status"])


def test_simulate_dynamic_retry(pga_params, rng):
    app_specs = {
        "A": {"instances": 20, "deadline_budget": 1.0}
    }
    df = _run_dynamic(
        app_specs,
        {"A": 0.1},
        pga_params,
        rng,
        rates={("Alice", "Bob"): 0.5},
    )[0]
    assert "failed" in set(df["status"])
