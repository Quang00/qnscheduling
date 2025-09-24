"""
Scheduling
----------
This module implements the Earliest Deadline First (EDF) scheduling-like
algorithm that schedules jobs based on their durations and parallelization
capabilities.
"""

from typing import Dict, List, Set, Tuple


def edf_parallel(
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
) -> List[Tuple[str, float, float]]:
    """EDF-like scheduling for parallel jobs.

    Args:
        job_rel_times (Dict[str, float]): Jobs' relative release times.
        job_periods (Dict[str, float]): Jobs' periods.
        parallel_apps (Dict[str, Set[str]]): Applications that can run in
        parallel.

    Returns:
        List[Tuple[str, float, float]]: Scheduled jobs with their start and
        deadline times.
    """
    schedule = []  # (job_name, start, deadline)
    jobs = sorted(
        job_periods,
        key=lambda j: job_rel_times.get(j, 0.0) + job_periods[j]
    )
    curr = 0.0

    for job in jobs:
        release = float(job_rel_times.get(job, 0.0))
        period = float(job_periods[job])
        deadline = release + period
        can_parallel = parallel_apps.get(job, set())

        conflicts = [
            dl
            for name, _, dl in schedule
            if name.rstrip("0123456789") not in can_parallel and dl > curr
        ]

        # Find the earliest start time considering conflicts and release time
        start = max(curr, max(conflicts, default=curr), release)
        if start >= deadline:
                skip = (start - deadline) // period + 1
                release += skip * period
                deadline += skip * period
                start = max(start, release)

        job_name = f"{job}0"
        schedule.append((job_name, start, deadline))
        curr = start

    return schedule
