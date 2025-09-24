"""
QNScheduling
------------
This script simulates the scheduling of probabilistic jobs (entanglement
generation attempts) in a quantum network. It calculates the duration of
applications based on the parameters of the links and the paths taken by the
applications. It uses an Earliest Deadline First (EDF) scheduling algorithm
that takes into account the parallelization capabilities of the applications.
The simulation can be run with a specified configuration file in YAML or GML
format, and the results are saved to a specified output directory.
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
    yaml_config,
)


def run_simulation(
    cfg_file,
    n_apps: int,
    max_instances: int,
    max_epr_pairs: int,
    seed: int,
    output_dir: str,
):
    """Run the quantum network scheduling simulation.

    Args:
        cfg_file (yaml or gml): Configuration file path in YAML or GML format.
        n_apps (int): Number of applications to generate.
        max_instances (int): Maximum number of instances per application.
        max_epr_pairs (int): Maximum number of EPR pairs to generate per
        application.
        seed (int): Random seed for reproducibility of the simulation.
        output_dir (str): Directory where the results will be saved.
    """
    rng = np.random.default_rng(seed)

    # Generate network data and applications based on the configuration file
    if cfg_file.endswith(".yaml"):
        edges, apps, instances, epr_pairs, policies = yaml_config(cfg_file)
    elif cfg_file.endswith(".gml"):
        nodes, edges, distances = gml_data(cfg_file)
        apps, instances, epr_pairs, policies = generate_n_apps(
            nodes,
            n_apps=n_apps,
            max_instances=max_instances,
            max_epr_pairs=max_epr_pairs,
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

    job_periods = {job: 2 * durations[job] for job in durations}
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the scheduling simulation")
    run.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML or GML config"
    )
    run.add_argument(
        "--apps",
        "-a",
        type=int,
        default=10,
        help="Number of applications to generate"
    )
    run.add_argument(
        "--inst",
        "-i",
        type=int,
        default=5,
        help="Maximum number of instances per application",
    )
    run.add_argument(
        "--epr",
        "-e",
        type=int,
        default=2,
        help="Maximum number of EPR pairs to generate per application",
    )
    run.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for simulation (optional)",
    )
    run.add_argument(
        "--output",
        "-o",
        default="results",
        help="Directory to save results into",
    )

    args = parser.parse_args()
    if args.command == "run":
        run_simulation(
            args.config,
            args.apps,
            args.inst,
            args.epr,
            args.seed,
            args.output,
        )


if __name__ == "__main__":
    main()
