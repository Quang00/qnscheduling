"""
QNScheduling
============

Overview:
---------
This script simulates the scheduling of quantum network applications. It
generates a set of applications based on a given network configuration,
computes their durations using Packet Generation Attempt (PGA) models, and
applies an Earliest Deadline First (EDF) scheduling algorithm to create a
static schedule. The simulation then runs over a specified number of
hyperperiod cycles, tracking PGA performance and link utilization, and saves
the results to an output directory. Each PGA duration is calculated based on
the end-to-end probability of generating EPR pairs, considering factors such
as memory lifetime, swap probabilities, and generation probabilities. The
scheduling process accounts for parallelizable tasks to optimize resource
usage.

Process:
--------
1. Parse command-line arguments for configuration, application parameters,
   scheduling parameters, and output settings.
2. Generate network data and applications based on the provided configuration.
3. Compute shortest paths for each application and identify parallelizable
   tasks.
4. Calculate the duration of each application using the PGA model.
5. Create a static schedule using the EDF scheduling algorithm.
6. Run a probabilistic simulation of the scheduled PGAs over the defined
   hyperperiod cycles.
7. Save the simulation results, including PGA performance metrics and link
   utilization, to the specified output directory.

Usage:
------
Run the script from the command line with appropriate arguments:
    --config: Path to the network configuration file (YAML or GML).
    --apps: Number of applications to generate.
    --inst: Range for the number of instances per application.
    --epr: Range for the number of EPR pairs per application.
    --period: Range for the period of each application.
    --hyperperiod: Number of hyperperiod cycles to simulate.
    --ppacket: Probability of a packet being generated.
    --memory: Memory lifetime in number of time slot units.
    --pswap: Probability of swapping an EPR pair in a single trial.
    --pgen: Probability of generating an EPR pair in a single trial.
    --slot-duration: Duration of a time slot in seconds.
    --seed: Random seed for reproducibility.
    --output: Directory to save the simulation results.
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd

from scheduling.pga import compute_durations
from scheduling.scheduling import edf_parallel
from scheduling.simulation import simulate_dynamic, simulate_static
from utils.helper import (
    app_params_sim,
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
    scheduler: str = "static",
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
    Returns:
    tuple[bool, pd.DataFrame | None, dict[str, float]]: A tuple indicating
    whether the schedule is feasible, the resulting PGA DataFrame when
    feasible, and the PGA durations per application.
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
        time_slot_duration,
    )
    print("Durations:", durations)

    pga_rel_times = {app: 0.0 for app in apps}
    print("Release times:", pga_rel_times)

    pga_periods = periods
    print("Periods:", pga_periods)

    print("Hyperperiod cycles:", hyperperiod_cycles)

    pga_parameters = app_params_sim(
        paths,
        epr_pairs,
        p_packet,
        memory_lifetime,
        p_swap,
        p_gen,
        time_slot_duration
    )

    # Run simulation
    os.makedirs(output_dir, exist_ok=True)
    if scheduler == "dynamic":
        simulate_dynamic()
        feasible = True
    else:
        feasible, schedule = edf_parallel(
            pga_rel_times,
            pga_periods,
            durations,
            parallel_map,
            hyperperiod_cycles,
        )

        if not feasible:
            print("Schedule", schedule)
            return False, None, durations

        print("Preview Schedule:", schedule[: n_apps * 2])

        df, pga_names, pga_release_times, link_utilization = simulate_static(
            schedule=schedule,
            pga_parameters=pga_parameters,
            pga_rel_times=pga_rel_times,
            pga_periods=pga_periods,
            policies=policies,
            pga_network_paths=paths,
            rng=rng,
        )

    # Save results
    save_results(
        df,
        pga_names,
        pga_release_times,
        apps,
        instances,
        epr_pairs,
        policies,
        durations=durations,
        n_edges=len(edges),
        link_utilization=link_utilization,
        output_dir=output_dir,
    )
    return feasible, df, durations


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
        type=float,
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
        "--scheduler",
        "-sch",
        type=str,
        choices=["static", "dynamic"],
        default="dynamic",
        help="Scheduling strategy: 'static' uses the precomputed EDF table,"
        " 'dynamic' schedules online",
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

    if args.scheduler == "dynamic":
        print("Dynamic scheduling is not yet implemented.")
        return

    seed_dir = os.path.join(args.output, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)

    run_number = 1
    pattern = re.compile(r"run(\d+)$")
    for name in os.listdir(seed_dir):
        m = pattern.match(name)
        if m:
            run_number = max(run_number, int(m.group(1)) + 1)

    run_dir = os.path.join(seed_dir, f"run{run_number}")
    os.makedirs(run_dir, exist_ok=False)

    t0 = time.perf_counter()
    feasible, _, _ = run_simulation(
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
        output_dir=run_dir,
        scheduler=args.scheduler,
    )
    t1 = time.perf_counter()

    if not feasible:
        return

    runtime = t1 - t0
    print(f"Run time: {runtime:.3f} seconds\n")

    params = {
        "config": args.config,
        "n_apps": args.apps,
        "inst_min": args.inst[0],
        "inst_max": args.inst[1],
        "epr_min": args.epr[0],
        "epr_max": args.epr[1],
        "period_min": args.period[0],
        "period_max": args.period[1],
        "hyperperiod_cycles": args.hyperperiod,
        "p_packet": args.ppacket,
        "memory_lifetime": args.memory,
        "p_swap": args.pswap,
        "p_gen": args.pgen,
        "time_slot_duration": args.slot_duration,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "run_number": run_number,
    }
    pd.DataFrame([params]).to_csv(os.path.join(run_dir, "params.csv"))

    path_results = os.path.join(run_dir, "pga_results.csv")
    path_params = os.path.join(run_dir, "params.csv")
    path_link_util = os.path.join(run_dir, "link_utilization.csv")
    print(f"Saved results to: {path_results}")
    print(f"Saved parameters to: {path_params}")
    print(f"Saved link utilization to: {path_link_util}")


if __name__ == "__main__":
    main()
