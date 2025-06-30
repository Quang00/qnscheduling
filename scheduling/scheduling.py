def simple_edf_schedule(
    durations: dict[str, float], parallel_apps: dict[str, set[str]]
) -> list[tuple[str, float, float]]:
    """Simple Earliest Deadline First (EDF) schedule for applications based on
    their PGA durations.

    Args:
        durations (dict[str, float]): A dictionary where keys are
        application names and values are their respective PGA durations in
        seconds.
        parallel_apps (dict[str, set[str]]): A dictionary where keys are
        application names and values are sets of application names that can
        run in parallel.

    Returns:
        list[tuple[str, float, float]]: A list of tuples, where each tuple
        contains the application name, start time, and end time of the
        scheduled application.
    """
    schedule = []
    curr = 0.0

    # Sort by earliest durations which are the deadlines
    jobs = sorted(durations, key=durations.get)

    for job in jobs:
        duration = durations[job]
        conflict_finishes = []
        parallelizable = parallel_apps.get(job, set())

        # Check for conflicts with already scheduled jobs
        for scheduled_job, _, end in schedule:
            if scheduled_job not in parallelizable and end > curr:
                conflict_finishes.append(end)

        # Wait for the latest conflicting job to finish
        if conflict_finishes:
            curr = max(conflict_finishes)

        start = curr
        end = start + duration
        schedule.append((job, start, end))

        if not parallelizable:
            curr = end

    return schedule
