"""
QNScheduling
------------
This script simulates the scheduling of probabilistic jobs (entanglement
generation attempts) in a quantum network. It calculates the duration of
applications based on the parameters of the links and the paths taken by the
applications. The script uses EDF, FCFS, or priority scheduling algorithms and
then simulates the execution of these applications in the network. It handles
parallelizable tasks and generates a simulation of the quantum network,
tracking performance metrics.
"""

import argparse
import os

from scheduling.scheduling import (
    simple_edf_schedule,
    simple_fcfs_schedule,
    simple_priority_schedule,
)
from scheduling.simulation import simulate
from utils.helper import (
    app_params_sim,
    compute_durations,
    generate_n_apps,
    gml_data,
    parallelizable_tasks,
    save_results,
    shortest_paths,
    yaml_config,
)


def select_scheduler(scheduler_type: str) -> callable:
    """Selects a scheduling algorithm based on the provided type.

    Args:
        scheduler_type (str): The type of scheduler to use. Options are "edf",
        "fcfs", or "priority".

    Returns:
        callabe: A function that implements the selected scheduling algorithm.
    """
    schedulers = {
        "edf": simple_edf_schedule,
        "fcfs": simple_fcfs_schedule,
        "priority": simple_priority_schedule,
    }
    return schedulers[scheduler_type]


def run_simulation(
    cfg_file,
    scheduler_name: str,
    n_apps: int,
    max_instances: int,
    max_epr_pairs: int,
    seed: int,
    output_dir: str,
):
    """Run the simulation based on the provided configuration file and
    scheduler.

    Args:
        cfg_file (_type_): Configuration file path in YAML format.
        scheduler_name (str): Name of the scheduling algorithm to use.
            Options are "edf", "fcfs", or "priority".
        n_apps (int): Number of applications to generate.
        max_instances (int): Maximum number of instances per application.
        max_epr_pairs (int): Maximum number of EPR pairs to generate per
        application.
        seed (int): Random seed for reproducibility of the simulation.
        output_dir (str): Directory where the results will be saved.
    """
    # Generate network data and applications based on the configuration file
    if cfg_file.endswith(".yaml"):
        edges, apps, instances, epr_pairs, priorities, policies = yaml_config(
            cfg_file
        )
    elif cfg_file.endswith(".gml"):
        nodes, edges, distances = gml_data(cfg_file)
        apps, instances, epr_pairs, priorities, policies = generate_n_apps(
            nodes,
            n_apps=n_apps,
            max_instances=max_instances,
            max_epr_pairs=max_epr_pairs,
            list_policies=["deadline", "best_effort"],
            seed=seed,
        )

    # Compute shortest paths and parallelizable tasks
    paths = shortest_paths(edges, apps)
    parallel_map = parallelizable_tasks(paths)

    # Compute durations for each application
    durations = compute_durations(paths, epr_pairs)

    # Choose scheduler and build schedule
    scheduler = select_scheduler(scheduler_name)
    if scheduler_name == "priority":
        schedule = scheduler(durations, parallel_map, instances, priorities)
    else:
        schedule = scheduler(durations, parallel_map, instances)

    # Run simulation (probabilistic) with optional seed
    os.makedirs(output_dir, exist_ok=True)
    df, job_names, release_times = simulate(
        schedule=schedule,
        job_parameters=app_params_sim(paths, epr_pairs),
        job_rel_times={app: 0.0 for app in durations},
        job_periods=durations.copy(),
        job_network_paths=paths,
        durations=durations,
        policies=policies,
        distances=distances,
        seed=seed,
    )

    # Save results
    save_results(
        df,
        job_names,
        release_times,
        apps,
        instances,
        epr_pairs,
        priorities,
        policies,
        output_dir=output_dir,
    )
    print(f"Results saved to directory {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate scheduling of quantum network applications"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the scheduling simulation")
    run.add_argument(
        "--config",
        required=True,
        help="Path to YAML or GML config"
    )
    run.add_argument(
        "--scheduler",
        choices=["edf", "fcfs", "priority"],
        default="edf",
        help="Scheduling algorithm",
    )
    run.add_argument(
        "--apps",
        type=int,
        default=10,
        help="Number of applications to generate"
    )
    run.add_argument(
        "--inst",
        type=int,
        default=5,
        help="Maximum number of instances per application",
    )
    run.add_argument(
        "--epr",
        type=int,
        default=2,
        help="Maximum number of EPR pairs to generate per application",
    )
    run.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation (optional)",
    )
    run.add_argument(
        "--output",
        default="results",
        help="Directory to save results into",
    )

    args = parser.parse_args()
    if args.command == "run":
        run_simulation(
            args.config,
            args.scheduler,
            args.apps,
            args.inst,
            args.epr,
            args.seed,
            args.output,
        )


if __name__ == "__main__":
    main()
