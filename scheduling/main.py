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
    p_packet: float,
    memory_lifetime: float,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
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
        p_packet (float): Probability of a packet being generated.
        memory_lifetime (float): Memory lifetime in number of time slot units.
        p_swap (float): Probability of swapping an EPR pair in a single trial.
        p_gen (float): Probability of generating an EPR pair in a single trial.
        time_slot_duration (float): Duration of a time slot in seconds.
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
    durations = compute_durations(
        paths,
        epr_pairs,
        p_packet,
        memory_lifetime,
        p_swap,
        p_gen,
        time_slot_duration
    )
    print("Durations:", durations)

    job_rel_times = {app: 0.0 for app in apps}
    print("Release times:", job_rel_times)

    job_periods = periods
    print("Periods:", job_periods)

    print("Hyperperiod cycles:", hyperperiod_cycles)

    # Compute static schedule
    feasible, schedule = edf_parallel(
        job_rel_times, job_periods, durations, parallel_map, hyperperiod_cycles
    )
    print("Schedule:", schedule)

    if not feasible:
        return None

    job_parameters = app_params_sim(
        paths,
        epr_pairs,
        p_packet,
        memory_lifetime,
        p_swap,
        p_gen,
        time_slot_duration
    )

    # Run simulation (probabilistic) with optional seed
    os.makedirs(output_dir, exist_ok=True)
    df, job_names, release_times = simulate_periodicity(
        schedule=schedule,
        job_parameters=job_parameters,
        job_rel_times=job_rel_times,
        job_periods=job_periods,
        policies=policies,
        job_network_paths=paths,
        distances=distances,
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
        help="Path to YAML or GML config",
    )
    parser.add_argument(
        "--apps",
        "-a",
        type=int,
        default=10,
        help="Number of applications to generate",
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
        default=10,
        help="Number of hyperperiods cycle: horizon (e.g., --hyperperiod 2)",
    )
    parser.add_argument(
        "--ppacket",
        "-pp",
        type=float,
        default=0.05,
        help="Probability of a packet being generated",
    )
    parser.add_argument(
        "--memory",
        "-m",
        type=int,
        default=50,
        help="Memory lifetime in number of time slot units",
    )
    parser.add_argument(
        "--pswap",
        "-ps",
        type=float,
        default=0.95,
        help="Probability of swapping an EPR pair in a single trial",
    )
    parser.add_argument(
        "--pgen",
        "-pg",
        type=float,
        default=1e-3,
        help="Probability of generating an EPR pair in a single trial",
    )
    parser.add_argument(
        "--slot-duration",
        "-sd",
        type=float,
        default=1e-4,
        help="Duration of a time slot in seconds",
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
        p_packet=args.ppacket,
        memory_lifetime=args.memory,
        p_swap=args.pswap,
        p_gen=args.pgen,
        time_slot_duration=args.slot_duration,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
