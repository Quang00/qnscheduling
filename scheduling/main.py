"""
QNScheduling
------------
This script simulates the scheduling of probabilistic jobs (entanglement
generation attempts) in a quantum network. It calculates the duration of
applications based on the parameters of the links and the paths taken by the
applications. It uses an Earliest Deadline First (EDF) scheduling algorithm
that takes into account the parallelization capabilities of the applications.
The simulation can be run with a specified random seed for reproducibility,
and the results are saved to a specified output directory. The network can be
defined using a GML file.
"""

import argparse
import os

import numpy as np

from scheduling.scheduling import edf_parallel
from scheduling.simulation import simulate_periodicity
from utils.helper import (
    app_params_sim,
    compute_durations,
    generate_n_apps,
    gml_data,
    parallelizable_tasks,
    save_results,
    shortest_paths,
)


def run_simulation(
    config: str,
    n_apps: int,
    inst_range: tuple[int, int],
    epr_range: tuple[int, int],
    period_range: tuple[float, float],
    hyperperiod_cycles: int,
    seed: int,
    output_dir: str,
):
    """Run the quantum network scheduling simulation.

    Args:
        config (yaml or gml): Configuration file path in YAML or GML format.
        n_apps (int): Number of applications to generate.
        inst_range (tuple[int, int]): Range (min, max) for the number of
        instances per application.
        epr_range (tuple[int, int]): Range (min, max) for the number of EPR
        pairs to generate per application.
        period_range (tuple[float, float]): Range (min, max) for the period of
        each application.
        hyperperiod_cycles (int): Number of hyperperiod cycles to simulate.
        seed (int): Random seed for reproducibility of the simulation.
        output_dir (str): Directory where the results will be saved.
    """
    rng = np.random.default_rng(seed)

    # Generate network data and applications based on the configuration file
    if config.endswith(".gml"):
        nodes, edges, distances = gml_data(config)
        apps, instances, epr_pairs, periods, policies = generate_n_apps(
            nodes,
            n_apps=n_apps,
            inst_range=inst_range,
            epr_range=epr_range,
            period_range=period_range,
            list_policies=["deadline"],
            rng=rng,
        )

    # Compute shortest paths and parallelizable tasks
    paths = shortest_paths(edges, apps)
    print("Paths:", paths)
    parallel_map = parallelizable_tasks(paths)
    print("Parallelizable tasks:", parallel_map)

    # Compute durations for each application
    durations = compute_durations(paths, epr_pairs)
    print("Durations:", durations)

    job_rel_times = {app: 0.0 for app in apps}
    print("Release times:", job_rel_times)

    job_periods = periods
    print("Periods:", job_periods)

    # Compute initial schedule
    schedule = edf_parallel(job_rel_times, job_periods, parallel_map)
    print("Initial Schedule:", schedule)

    # Run simulation (probabilistic) with optional seed
    os.makedirs(output_dir, exist_ok=True)
    df, job_names, release_times = simulate_periodicity(
        schedule=schedule,
        job_parameters=app_params_sim(paths, epr_pairs),
        job_rel_times=job_rel_times,
        job_periods=job_periods,
        policies=policies,
        job_network_paths=paths,
        distances=distances,
        instances=instances,
        hyperperiod_cycles=hyperperiod_cycles,
        rng=rng,
    )

    # Save results
    save_results(
        df,
        job_names,
        release_times,
        apps,
        instances,
        epr_pairs,
        policies,
        output_dir=output_dir,
    )
    print(f"Results saved to directory {output_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Simulate scheduling of quantum network applications"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configurations/network/Dumbbell.gml",
        help="Path to YAML or GML config"
    )
    parser.add_argument(
        "--apps",
        "-a",
        type=int,
        default=10,
        help="Number of applications to generate"
    )
    parser.add_argument(
        "--inst",
        "-i",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[2, 2],
        help="Number of instances to generate (e.g., --inst 1 5)",
    )
    parser.add_argument(
        "--epr",
        "-e",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[2, 2],
        help="Maximum number of EPR pairs to generate per application"
        "(e.g., --epr 1 5)",
    )
    parser.add_argument(
        "--period",
        "-p",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[3.0, 3.0],
        help="Period of the application (e.g., --period 1.0 5.0)",
    )
    parser.add_argument(
        "--hyperperiod",
        "-hp",
        type=int,
        default=float("inf"),
        help="Number of hyperperiods cycle: horizon (e.g., --hyperperiod 2)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for simulation (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    run_simulation(
        config=args.config,
        n_apps=args.apps,
        inst_range=args.inst,
        epr_range=args.epr,
        period_range=args.period,
        hyperperiod_cycles=args.hyperperiod,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
