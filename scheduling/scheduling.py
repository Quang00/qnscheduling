def simple_edf_schedule(
    app_w_pga_durations: dict[str, float], parallel_apps: dict[str, set[str]]
) -> list[tuple[str, float, float]]:
    """Simple Earliest Deadline First (EDF) schedule for applications based on
    their PGA durations

    Args:
        app_w_pga_durations (dict[str, float]): A dictionary where keys are
        application names and values are their respective PGA durations in
        mircoseconds.
        parallel_apps (dict[str, set[str]]): A dictionary where keys are
        application names and values are sets of application names that can
        run in parallel.

    Returns:
        list[tuple[str, float, float]]: A list of tuples, where each tuple
        contains the application name, start time, and end time of the
        scheduled application.
    """
    groups = []
    seen = set()

    if parallel_apps:
        for app in parallel_apps:
            valid = [t for t in app if t in app_w_pga_durations]
            if not valid:
                continue
            groups.append(valid)
            seen.update(valid)
    print(groups)

    for t in app_w_pga_durations:
        if t not in seen:
            groups.append([t])

    deadlines = [min(app_w_pga_durations[t] for t in grp) for grp in groups]
    order = sorted(range(len(groups)), key=lambda i: deadlines[i])

    schedule = []
    time = 0
    for i in order:
        app = groups[i]
        for t in app:
            schedule.append((t, time, time + app_w_pga_durations[t]))
        time += max(app_w_pga_durations[t] for t in app)
    return schedule
