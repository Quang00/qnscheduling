"""
Scheduling
----------
This module implements a feasibility test and scheduling algorithm for
probabilistic jobs in a quantum network. It uses an Earliest Deadline First
(EDF) approach to schedule jobs while considering their parallelization
capabilities. The schedule is constructed over a specified number of
hyperperiod cycles that ensure jobs are executed within their deadlines if
feasible.
"""

from math import floor
from typing import Dict, List, Set, Tuple

from utils.helper import hyperperiod

EPS = 1e-12


def edf_parallel(
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    durations: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
    horizon_cycles: int,
) -> Tuple[bool, List[Tuple[str, float, float, float]] | str]:
    """Edf scheduling with parallelization capabilities. The static schedule is
    constructed over a given number of hyperperiod cycles. It checks if the
    set of jobs is feasible and returns the schedule if so.

    Args:
        job_rel_times (Dict[str, float]): Job relative start times.
        job_periods (Dict[str, float]): Job periods.
        durations (Dict[str, float]): Job execution durations.
        parallel_apps (Dict[str, Set[str]]): Dictionary mapping each job to a
        set of jobs that can run in parallel with it.
        horizon_cycles (int): Number of hyperperiod cycles to consider.

    Returns:
        Tuple[bool, List[Tuple[str, float, float, float]] | str]: A tuple where
        the first element indicates if the schedule is feasible, and the second
        element is either the schedule (job_name, start, end, deadline) or an
        error message (if not).
    """
    jobs = list(job_periods.keys())

    U = {j: float(durations[j]) / float(job_periods[j]) for j in jobs}
    for j in jobs:
        if U[j] > 1.0 + EPS:
            return False, (
                f"Infeasible: job '{j}' has utilization > 1 "
                f"(duration={durations[j]:.9f} > period={job_periods[j]:.9f})"
            )

    H = float(hyperperiod(job_periods))
    if H <= 0.0 or horizon_cycles < 1:
        return False, "invalid horizon"

    # Parallelization conflicts
    conflicts = {
        a: {b for b in jobs if b != a and b not in parallel_apps.get(a, set())}
        for a in jobs
    }

    # Generate all job instances within the hyperperiod horizon
    horizon = horizon_cycles * H
    instances = []
    for j in jobs:
        T = float(job_periods[j])
        base_r = float(job_rel_times.get(j, 0.0))
        pga = float(durations[j])
        n = max(0, floor(((horizon - base_r) / T) - EPS) + 1)
        for k in range(n):
            rel_k = base_r + k * T
            if rel_k >= horizon - EPS:
                break
            dl_k = rel_k + T
            instances.append((j, k, rel_k, dl_k, pga))

    # Sort instances by (deadline, release time)
    instances.sort(key=lambda x: (x[3], x[2]))

    # Keep track of the last finish time for each job
    last_finish = {j: 0.0 for j in jobs}
    schedule = []
    for j, k, rel, dl, pga in instances:
        block_until = 0.0
        for c in conflicts[j]:
            if last_finish[c] > block_until:
                block_until = last_finish[c]
        if last_finish[j] > block_until:
            block_until = last_finish[j]

        start = max(rel, block_until)
        end = start + pga
        if end > dl + EPS:
            return (
                False,
                "Infeasible schedule -> deadline miss: "
                f"{j}{k} end={end:.9f} > deadline={dl:.9f}",
            )
        schedule.append((f"{j}{k}", start, end, dl))
        last_finish[j] = end

    return True, schedule
