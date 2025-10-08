import numpy as np
import pytest

from scheduling.simulation import simulate_periodicity


@pytest.fixture
def base_job_parameters():
    return {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "memory_lifetime": 2,
        "epr_pairs": 2,
        "slot_duration": 0.1,
    }


def test_simulation_completes_basic(base_job_parameters):
    job_parameters = {"A": dict(base_job_parameters)}
    job_rel_times = {"A": 0.0}
    job_periods = {"A": 10.0}
    job_network_paths = {"A": ["Alice", "Bob"]}
    policies = {"A": "deadline"}
    distances = {("Alice", "Bob"): 0.0}
    schedule = [("A0", 0.0, 5.0, 10.0)]
    rng = np.random.default_rng(42)

    df, _, _ = simulate_periodicity(
        schedule=schedule,
        job_parameters=job_parameters,
        job_rel_times=job_rel_times,
        job_periods=job_periods,
        job_network_paths=job_network_paths,
        policies=policies,
        distances=distances,
        rng=rng,
    )

    assert df.loc[0, "status"] == "completed"


def test_simulation_fails_basic(base_job_parameters):
    job_parameters = {"A": dict(base_job_parameters)}
    job_rel_times = {"A": 0.0}
    job_periods = {"A": 5.0}
    job_network_paths = {"A": ["Alice", "Bob"]}
    policies = {"A": "deadline"}
    distances = {("Alice", "Bob"): 0.0}
    schedule = [("A0", 0.0, 0.01, 5.0)]
    rng = np.random.default_rng(42)

    df, _, _ = simulate_periodicity(
        schedule=schedule,
        job_parameters=job_parameters,
        job_rel_times=job_rel_times,
        job_periods=job_periods,
        job_network_paths=job_network_paths,
        policies=policies,
        distances=distances,
        rng=rng,
    )

    assert df.loc[0, "status"] == "failed"
