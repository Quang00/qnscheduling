"""
Scheduling
----------
This module implements the Earliest Deadline First (EDF) scheduling-like
algorithm that schedules jobs based on their durations and parallelization
capabilities.
"""

import math
from typing import Dict, List, Set, Tuple


def edf_parallel(
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
    instances: Dict[str, int],
) -> List[Tuple[str, float, float]]:
    """EDF-like scheduling for parallel jobs.

    Args:
        job_rel_times (Dict[str, float]): Jobs' relative release times.
        job_periods (Dict[str, float]): Jobs' periods.
        parallel_apps (Dict[str, Set[str]]): Applications that can run in
        parallel.
        instances (Dict[str, int]): Number of instances for each job.

    Returns:
        List[Tuple[str, float, float]]: Scheduled jobs with their start and
        deadline times.
    """

    schedule = []  # (name, start, deadline)
    jobs = sorted(job_periods, key=job_periods.get)
    max_instances = max((instances.get(job, 0) for job in jobs), default=0)

    for k in range(max_instances):
        if k == 0:
            curr = 0.0
        else:
            prev_deadlines = [
                dl for name, _, dl in schedule if name.endswith(str(k - 1))
            ]
            curr = max(prev_deadlines) if prev_deadlines else 0.0

        for job in jobs:
            if instances.get(job, 0) <= k:
                continue

            P = float(job_periods[job])
            rel0 = float(job_rel_times.get(job, 0.0))
            eff_k = k
            release = rel0 + eff_k * P
            deadline = release + P
            can_parallel = parallel_apps.get(job, set())

            conflicts = [
                dl
                for name, st, dl in schedule
                if name.rstrip("0123456789") not in can_parallel and dl > curr
            ]

            tentative_start = max(curr, max(conflicts, default=curr), release)

            if tentative_start >= deadline:
                skip = math.floor((tentative_start - deadline) / P) + 1
                eff_k += skip
                release += skip * P
                deadline += skip * P
                tentative_start = max(tentative_start, release)

            name_idx = k if eff_k == k else max(eff_k - 1, 0)
            job_name = f"{job}{name_idx}"

            schedule.append((job_name, tentative_start, deadline))

    return schedule
