import numpy as np
import pytest

from scheduling.simulation import simulate_static


@pytest.fixture
def base_pga_parameters():
    return {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "memory_lifetime": 2,
        "epr_pairs": 2,
        "slot_duration": 0.1,
    }


def test_simulation_completes_basic(base_pga_parameters):
    pga_parameters = {"A": dict(base_pga_parameters)}
    pga_rel_times = {"A": 0.0}
    pga_periods = {"A": 10.0}
    pga_network_paths = {"A": ["Alice", "Bob"]}
    policies = {"A": "deadline"}
    schedule = [("A0", 0.0, 5.0, 10.0)]
    rng = np.random.default_rng(42)

    df, _, _, _, _ = simulate_static(
        schedule=schedule,
        app_specs={"A": {"instances": 1}},
        pga_parameters=pga_parameters,
        pga_rel_times=pga_rel_times,
        pga_periods=pga_periods,
        pga_network_paths=pga_network_paths,
        policies=policies,
        rng=rng,
    )

    assert df.loc[0, "status"] == "completed"


def test_simulation_fails_basic(base_pga_parameters):
    pga_parameters = {"A": dict(base_pga_parameters)}
    pga_rel_times = {"A": 0.0}
    pga_periods = {"A": 5.0}
    pga_network_paths = {"A": ["Alice", "Bob"]}
    policies = {"A": "deadline"}
    schedule = [("A0", 0.0, 0.01, 5.0)]
    rng = np.random.default_rng(42)

    df, _, _, _, _ = simulate_static(
        schedule=schedule,
        app_specs={"A": {"instances": 1}},
        pga_parameters=pga_parameters,
        pga_rel_times=pga_rel_times,
        pga_periods=pga_periods,
        pga_network_paths=pga_network_paths,
        policies=policies,
        rng=rng,
    )

    assert df.loc[0, "status"] == "failed"
