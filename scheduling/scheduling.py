"""
Scheduling
----------
This module implements a feasibility test and scheduling algorithm for
Packet Generation Attempts (PGAs) in a quantum network. It uses an Earliest
Deadline First (EDF) approach to schedule PGAs while considering their
parallelization capabilities. The schedule is constructed over a specified
number of hyperperiod cycles that ensure PGAs are executed within their
deadlines if feasible.
"""

from math import floor
from typing import Dict, List, Set, Tuple

from utils.helper import hyperperiod

EPS = 1e-12


def edf_parallel_static(
    pga_rel_times: Dict[str, float],
    pga_periods: Dict[str, float],
    durations: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
    horizon_cycles: int,
) -> Tuple[bool, List[Tuple[str, float, float, float]] | str]:
    """EDF scheduling with parallelization capabilities. The static schedule is
    constructed over a given number of hyperperiod cycles. It checks if the
    set of PGAs is feasible and returns the schedule if so.

    Args:
        pga_rel_times (Dict[str, float]): PGA relative start times.
        pga_periods (Dict[str, float]): PGA periods.
        durations (Dict[str, float]): PGA execution durations.
        parallel_apps (Dict[str, Set[str]]): Dictionary mapping each PGA to a
        set of PGAs that can run in parallel with it.
        horizon_cycles (int): Number of hyperperiod cycles to consider.

    Returns:
        Tuple[bool, List[Tuple[str, float, float, float]] | str]: A tuple where
        the first element indicates if the schedule is feasible, and the second
        element is either the schedule (pga_name, start, end, deadline) or an
        error message (if not).
    """
    pgas = list(pga_periods.keys())

    U = {p: float(durations[p]) / float(pga_periods[p]) for p in pgas}
    for p in pgas:
        if U[p] > 1.0 + EPS:
            return False, (
                f"Infeasible: task '{p}' has utilization > 1 "
                f"(duration={durations[p]:.9f} > period={pga_periods[p]:.9f})"
            )

    H = float(hyperperiod(pga_periods))
    if H <= 0.0 or horizon_cycles < 1:
        return False, "invalid horizon"

    # Parallelization conflicts
    conflicts = {
        a: {b for b in pgas if b != a and b not in parallel_apps.get(a, set())}
        for a in pgas
    }

    # Generate all PGA instances within the hyperperiod horizon
    horizon = horizon_cycles * H
    instances = []
    for pga in pgas:
        T = float(pga_periods[pga])
        base_r = float(pga_rel_times.get(pga, 0.0))
        pga_duration = float(durations[pga])
        n = max(0, floor(((horizon - base_r) / T) - EPS) + 1)
        for k in range(n):
            rel_k = base_r + k * T
            if rel_k >= horizon - EPS:
                break
            dl_k = rel_k + T
            instances.append((pga, k, rel_k, dl_k, pga_duration))

    # Sort instances by (deadline, release time)
    instances.sort(key=lambda x: (x[3], x[2]))

    # Keep track of the last finish time for each PGA
    last_finish = {p: 0.0 for p in pgas}
    schedule = []
    for pga_name, k, rel, dl, pga_duration in instances:
        block_until = 0.0
        for c in conflicts[pga_name]:
            if last_finish[c] > block_until:
                block_until = last_finish[c]
        if last_finish[pga_name] > block_until:
            block_until = last_finish[pga_name]

        start = max(rel, block_until)
        end = start + pga_duration
        if end > dl + EPS:
            return (
                False,
                "Infeasible schedule -> deadline miss: "
                f"{pga_name}{k} end={end:.9f} > deadline={dl:.9f}",
            )
        schedule.append((f"{pga_name}{k}", start, end, dl))
        last_finish[pga_name] = end

    return True, schedule
