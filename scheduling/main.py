"""
QNScheduling
============

Overview:
---------
This script simulates the scheduling of quantum network applications. It
generates a set of applications based on a given network configuration,
computes their durations using Packet Generation Attempt (PGA). There are two
scheduling strategies available: static scheduling using Earliest Deadline
First (EDF) that precomputes a schedule, and dynamic scheduling that makes
decisions online based on application arrivals. The simulation runs until all
applications are processed, collecting performance metrics such as completion
status, delays, and link utilization.

Process:
--------
1. Parse command-line arguments for configuration, application parameters,
   scheduling parameters, and output settings.
2. Generate network data and applications based on the provided configuration.
3. Compute shortest paths for each application and identify parallelizable
   tasks.
4. Calculate the duration of each application using PGA parameters.
5. Depending on the chosen scheduling strategy (static or dynamic):

   - For static scheduling, compute a feasible schedule using EDF with
      parallelization capabilities.
   - For dynamic scheduling, prepare for online scheduling based on arrivals.
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
    --memory: Memory: number of independent link-generation trials per slot.
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
from scheduling.routing import find_feasible_path, shortest_paths
from scheduling.simulation import simulate_dynamic, simulate_static
from scheduling.static import edf_parallel_static
from utils.helper import (
    app_params_sim,
    generate_n_apps,
    gml_data,
    parallelizable_tasks,
    save_results,
)


def run_simulation(
    config: str,
    n_apps: int,
    inst_range: tuple[int, int],
    epr_range: tuple[int, int],
    period_range: tuple[float, float],
    hyperperiod_cycles: int,
    p_packet: float,
    memory: float,
    p_swap: float,
    p_gen: float,
    fidelity_range: tuple[float, float],
    time_slot_duration: float,
    seed: int,
    output_dir: str,
    scheduler: str = "static",
    arrival_rate: float | None = None,
    routing: str = "shortest",
    capacity_threshold: float = 0.8,
    save_csv: bool = True,
    verbose: bool = True,
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
        memory (float): Memory: number of independent link-generation trials
            per slot.
        p_swap (float): Probability of swapping an EPR pair in a single trial.
        p_gen (float): Probability of generating an EPR pair in a single trial.
        fidelity_range (tuple[float, float]): Range (min, max) for the minimum
            fidelity of each application.
        time_slot_duration (float): Duration of a time slot in seconds.
        seed (int): Random seed for reproducibility of the simulation.
        output_dir (str): Directory where the results will be saved.
        scheduler (str): Either "static" or "dynamic".
        arrival_rate (float | None): Mean rate lambda for Poisson arrivals.
            When None, releases remain periodic.
    Returns:
        tuple[bool, dict]:
            A tuple containing:
            - bool: whether the schedule is feasible
            - dict: summary metrics dictionary
    """
    rng = np.random.default_rng(seed)

    # Generate network data and applications based on the configuration file
    if config.endswith(".gml"):
        nodes, edges, distances, fidelities, end_nodes = gml_data(config)
        app_specs = generate_n_apps(
            nodes,
            end_nodes,
            n_apps=n_apps,
            inst_range=inst_range,
            epr_range=epr_range,
            period_range=period_range,
            fid_range=fidelity_range,
            list_policies=["deadline"],
            rng=rng,
        )

    # Arrival times for each application
    poisson_enabled = (
        scheduler == "dynamic"
        and arrival_rate is not None
        and float(arrival_rate) > 0.0
    )
    if poisson_enabled:
        mean_interarrival = 1.0 / float(arrival_rate)
        pga_rel_times = {
            app: float(rng.exponential(mean_interarrival))
            for app in app_specs.keys()
        }
    else:
        pga_rel_times = {app: 0.0 for app in app_specs.keys()}

    # Find feasible paths for each application based on fidelity/routing mode
    app_requests = {
        name: {
            "src": spec["src"],
            "dst": spec["dst"],
            "min_fidelity": spec.get("min_fidelity", 0.0),
            "epr": spec.get("epr", 0),
            "period": spec.get("period", 0.0),
            "arrival_time": pga_rel_times[name],
        }
        for name, spec in app_specs.items()
    }
    fidelity_enabled = not (
        len(fidelity_range) == 2
        and float(fidelity_range[0]) == 0.0
        and float(fidelity_range[1]) == 0.0
    )
    total_apps = len(app_specs)
    routing_mode = str(routing)
    admitted_specs = {}

    if not fidelity_enabled:
        paths = shortest_paths(edges, app_requests)
    else:
        paths = find_feasible_path(
            edges,
            app_requests,
            fidelities if fidelity_enabled else None,
            pga_rel_times=pga_rel_times,
            routing_mode=routing_mode,
            threshold=capacity_threshold,
            p_packet=p_packet,
            memory=memory,
            p_swap=p_swap,
            p_gen=p_gen,
            time_slot_duration=time_slot_duration,
            rng=rng,
        )

    admitted_paths = {
        app: path for app, path in paths.items() if path is not None
    }
    admitted_specs = {app: app_specs[app] for app in admitted_paths.keys()}
    admitted_apps = len(admitted_specs)

    if admitted_apps == 0:
        return False, {}

    app_specs = admitted_specs
    paths = admitted_paths

    parallel_map = parallelizable_tasks(paths)
    epr_pairs = {name: spec["epr"] for name, spec in app_specs.items()}

    # Compute durations for each application
    durations = compute_durations(
        paths,
        epr_pairs,
        p_packet,
        memory,
        p_swap,
        p_gen,
        time_slot_duration,
    )

    pga_periods = {name: spec["period"] for name, spec in app_specs.items()}

    pga_parameters = app_params_sim(
        paths,
        app_specs,
        p_packet,
        memory,
        p_swap,
        p_gen,
        time_slot_duration
    )

    policies = {name: spec["policy"] for name, spec in app_specs.items()}

    # Run simulation
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)

    app_requests_df = (
        pd.DataFrame.from_dict(app_requests, orient="index")
        .reset_index()
        .rename(columns={"index": "app"})
    )
    hops_map = {
        app: (len(path) - 1) if path is not None else np.nan
        for app, path in paths.items()
    }
    app_requests_df["hops"] = app_requests_df["app"].map(hops_map)
    app_requests_df["admitted"] = app_requests_df["app"].isin(app_specs)
    if save_csv:
        app_requests_df.to_csv(
            os.path.join(output_dir, "app_requests.csv"), index=False
        )

    if scheduler == "dynamic":
        (
            df,
            pga_names,
            pga_release_times,
            link_utilization,
            link_waiting,
        ) = simulate_dynamic(
            app_specs,
            durations,
            pga_parameters,
            pga_rel_times,
            paths,
            rng,
            arrival_rate,
        )
        feasible = True
    else:
        feasible, schedule = edf_parallel_static(
            pga_rel_times,
            pga_periods,
            durations,
            parallel_map,
            hyperperiod_cycles,
        )

        if not feasible:
            if verbose:
                print("Schedule", schedule)
            return False, {}

        if verbose:
            print("Preview Schedule:", schedule[: n_apps * 2])

        (
            df,
            pga_names,
            pga_release_times,
            link_utilization,
            link_waiting,
        ) = simulate_static(
            schedule=schedule,
            app_specs=app_specs,
            pga_parameters=pga_parameters,
            pga_rel_times=pga_rel_times,
            pga_periods=pga_periods,
            policies=policies,
            pga_network_paths=paths,
            rng=rng,
        )

    # Save results
    summary = save_results(
        df,
        pga_names,
        pga_release_times,
        app_specs,
        durations=durations,
        pga_network_paths=paths,
        n_edges=len(edges),
        link_utilization=link_utilization,
        link_waiting=link_waiting,
        admitted_apps=admitted_apps,
        total_apps=total_apps,
        output_dir=output_dir,
        save_csv=save_csv,
        verbose=verbose,
    )
    return feasible, summary


def main():
    parser = argparse.ArgumentParser(
        description="Simulate scheduling of quantum network applications"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configurations/network/basic/Dumbbell.gml",
        help="Path to YAML or GML config",
    )
    parser.add_argument(
        "--apps",
        "-a",
        type=int,
        default=2,
        help="Number of applications to generate",
    )
    parser.add_argument(
        "--inst",
        "-i",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[2, 2],
        help="Number of entanglement packets to generate per application"
        "(e.g., --inst 1 5)",
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
        default=[1.0, 1.0],
        help="Period of the application (e.g., --period 1.0 5.0)",
    )
    parser.add_argument(
        "--hyperperiod",
        "-hp",
        type=float,
        default=1000,
        help="Number of hyperperiods cycle: horizon (e.g., --hyperperiod 2)",
    )
    parser.add_argument(
        "--ppacket",
        "-pp",
        type=float,
        default=0.2,
        help="Probability of a packet being generated",
    )
    parser.add_argument(
        "--memory",
        "-m",
        type=int,
        default=1000,
        help="Number of independent link-generation trials per slot in the"
        "multiplexed memory",
    )
    parser.add_argument(
        "--pswap",
        "-ps",
        type=float,
        default=0.6,
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
        "--min-fidelity",
        "-f",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[0.0, 0.0],
        help="Minimum fidelity per application (e.g., --min-fidelity 0.6 0.7)",
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
        "--arrival-rate",
        "-ar",
        type=float,
        default=None,
        help="Mean arrival rate (lambda) for Poisson process",
    )
    parser.add_argument(
        "--routing",
        "-r",
        type=str,
        choices=["shortest", "capacity", "random",
                 "degree", "smallest", "least"],
        default="shortest",
        help=(
            "Routing: 'shortest' (Dijkstra), 'capacity' (capacity-aware),"
            "'smallest' (smallest bottleneck), 'least' (least total capacity),"
            "'random' (random path), or 'degree' (degree-aware)."
        ),
    )
    parser.add_argument(
        "--capacity-threshold",
        "-ct",
        type=float,
        default=0.8,
        help="Capacity threshold for routing capacity (sum(PGA/T) per link)",
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
    seed_dir = os.path.join(args.output, f"seed_{args.seed}")

    run_number = 1
    pattern = re.compile(r"run(\d+)$")
    if os.path.isdir(seed_dir):
        for name in os.listdir(seed_dir):
            m = pattern.match(name)
            if m:
                run_number = max(run_number, int(m.group(1)) + 1)

    run_dir = os.path.join(seed_dir, f"run{run_number}")

    t0 = time.perf_counter()
    feasible, _ = run_simulation(
        config=args.config,
        n_apps=args.apps,
        inst_range=args.inst,
        epr_range=args.epr,
        period_range=args.period,
        hyperperiod_cycles=args.hyperperiod,
        p_packet=args.ppacket,
        memory=args.memory,
        p_swap=args.pswap,
        p_gen=args.pgen,
        fidelity_range=args.min_fidelity,
        time_slot_duration=args.slot_duration,
        seed=args.seed,
        output_dir=run_dir,
        scheduler=args.scheduler,
        arrival_rate=args.arrival_rate,
        routing=args.routing,
        capacity_threshold=args.capacity_threshold,
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
        "memory": args.memory,
        "p_swap": args.pswap,
        "p_gen": args.pgen,
        "fidelity_min": args.min_fidelity[0],
        "fidelity_max": args.min_fidelity[1],
        "time_slot_duration": args.slot_duration,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "run_number": run_number,
        "routing": args.routing,
        "capacity_threshold": args.capacity_threshold,
    }
    pd.DataFrame([params]).to_csv(os.path.join(run_dir, "params.csv"))

    path_results = os.path.join(run_dir, "pga_results.csv")
    path_params = os.path.join(run_dir, "params.csv")
    path_link_util = os.path.join(run_dir, "link_utilization.csv")
    path_link_wait = os.path.join(run_dir, "link_waiting.csv")
    print(f"Saved results to: {path_results}")
    print(f"Saved parameters to: {path_params}")
    print(f"Saved link utilization to: {path_link_util}")
    print(f"Saved link waiting metrics to: {path_link_wait}")


if __name__ == "__main__":
    main()
