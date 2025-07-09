"""
Scheduling
----------
This module provides simple scheduling algorithms for quantum network jobs.
It includes Earliest Deadline First (EDF), First-Come First-Served (FCFS), and
Priority scheduling methods. Each method takes job durations and parallelizable
applications into account to generate a schedule of jobs with their start and
end times.
It is designed to be used in conjunction with a quantum network simulation
environment, where jobs represent entanglement generation tasks that can be
scheduled based on their durations and parallelization capabilities.
"""

from typing import Dict, List, Set, Tuple


def _simple_schedule(
    jobs: List[str],
    durations: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
) -> List[Tuple[str, float, float]]:
    """Simple scheduling algorithm that schedules jobs based on their durations
    and parallelization capabilities.

    Args:
        jobs (List[str]): List of job names to schedule.
        durations (Dict[str, float]): Mapping from job names to their
        durations.
        parallel_apps (Dict[str, Set[str]]): Mapping from job names to sets of
        jobs they can run in parallel with.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples, each containing
        the job name, start time, and end time of the scheduled jobs.
    """
    schedule = []
    current_time = 0.0

    for job in jobs:
        duration = durations[job]
        parallelizable = parallel_apps.get(job, set())

        # Check for conflicts with already scheduled jobs
        conflicts = [
            end
            for scheduled_job, _, end in schedule
            if scheduled_job not in parallelizable and end > current_time
        ]

        # Wait for the latest conflict to finish
        if conflicts:
            current_time = max(conflicts)

        start = current_time
        end = start + duration
        schedule.append((job, start, end))

        if not parallelizable:
            current_time = end

    return schedule


def simple_edf_schedule(
    durations: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
) -> List[Tuple[str, float, float]]:
    """Earliest Deadline First (EDF): schedule jobs based on their durations.

    Args:
        durations (Dict[str, float]): Mapping from job names to their
        durations.
        parallel_apps (Dict[str, Set[str]]): Mapping from job names to sets of
        jobs they can run in parallel with.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples, each containing
        the job name, start time, and end time of the scheduled jobs.
    """
    jobs = sorted(durations, key=durations.get)
    return _simple_schedule(jobs, durations, parallel_apps)


def simple_fcfs_schedule(
    durations: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
) -> List[Tuple[str, float, float]]:
    """First-Come First-Served (FCFS): schedule jobs in the order they are
    provided.

    Args:
        durations (Dict[str, float]): Mapping from job names to their
        durations.
        parallel_apps (Dict[str, Set[str]]): Mapping from job names to sets of
        jobs they can run in parallel with.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples, each containing
        the job name, start time, and end time of the scheduled jobs.
    """
    jobs = list(durations.keys())
    return _simple_schedule(jobs, durations, parallel_apps)


def simple_priority_schedule(
    durations: Dict[str, float],
    parallel_apps: Dict[str, Set[str]],
    priorities: Dict[str, int],
) -> List[Tuple[str, float, float]]:
    """Priority scheduling: schedule jobs based on their priority levels.

    Args:
        durations (Dict[str, float]): Mapping from job names to their
        durations.
        parallel_apps (Dict[str, Set[str]]): Mapping from job names to sets of
        jobs they can run in parallel with.
        priorities (Dict[str, int]): Mapping from job names to their priority
        levels.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples, each containing
        the job name, start time, and end time of the scheduled jobs.
    """
    # Default priority is 0 if not specified
    jobs = sorted(
        priorities.keys(),
        key=lambda job: priorities.get(job, 0),
        reverse=True,
    )
    return _simple_schedule(jobs, durations, parallel_apps)
