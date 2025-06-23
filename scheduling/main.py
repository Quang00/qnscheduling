import math

from scheduling.pga import duration_pga
from scheduling.scheduling import simple_edf_schedule
from utils.helper import parallelizable_tasks, shortest_paths


def main():
    # Quantum network (Toy example)
    edges = [
        ("Alice", "Bob"),
        ("Alice", "Charlie"),
        ("Charlie", "David"),
        ("Bob", "David"),
    ]

    # Application requests
    app_requests = {
        "A": ("Alice", "David"),
        "B": ("Alice", "Bob"),
        "C": ("Charlie", "Bob"),
        "D": ("Charlie", "David"),
    }

    # Find shortest paths for each application in the quantum network
    paths = shortest_paths(edges, app_requests)
    print("Shortest paths:", paths)

    # Find parallelizable applications based on shared resources
    parallel_apps = parallelizable_tasks(paths)
    print("Parallelizable applications:", parallel_apps)

    # Calculate PGA durations for each application
    duration_apps = {}
    for app in paths:
        pga_duration = duration_pga(
            p_packet=0.2,
            k=2,
            n_swap=math.ceil(len(paths[app]) / 2),
            memory_lifetime=50,
            p_swap=0.95,
            p_gen=0.001,
            time_slot_duration=100,
        )
        duration_apps[app] = pga_duration
    print("PGA durations for applications:", duration_apps)

    # Simple EDF schedule for the applications based on their PGA durations
    schedule = simple_edf_schedule(duration_apps, parallel_apps)
    print("EDF Schedule:", schedule)


if __name__ == "__main__":
    main()
