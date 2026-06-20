"""
QNScheduling
============

Overview:
---------
This script simulates the scheduling of quantum network applications. It
generates a set of applications based on a given network configuration,
computes their durations using Packet Generation Attempt (PGA). There are two
scheduling strategies: static scheduling (deprecated) using Earliest Deadline
First (EDF) that precomputes a schedule, and dynamic scheduling that makes
decisions online based on application arrivals. Application releases are drawn
from a Poisson process with rate ``--arrival-rate`` over the observation
horizon. The simulation runs for a fixed time horizon; any pending releases
past the horizon are dropped. Metrics collected over the window [warmup,
horizon].
"""

import argparse
import json
import os
import re
import time

import numpy as np

from scheduling.fidelity import fidelity_bounds_and_paths
from scheduling.pga import compute_durations
from scheduling.routing import (
    compute_path_durations,
    find_feasible_path,
    static_routing,
)
from scheduling.simulation import simulate_dynamic
from utils.graph_generator import clos, fat_tree, generate_waxman_graph
from utils.helper import (
    all_simple_paths,
    app_params_sim,
    count_edge_disjoint_paths,
    generate_n_apps,
    gml_data,
    save_results,
)


def run_simulation(
    config: str,
    arrival_rate: float,
    inst_range: int,
    epr_range: tuple[int, int],
    deadline_range: tuple[float, float],
    p_packet: float,
    memory: float,
    p_swap: float,
    time_slot_duration: float,
    seed: int,
    output_dir: str,
    instance_arrival_rate: float = 10.0,
    routing: str = "shortest",
    save_csv: bool = True,
    verbose: bool = True,
    graph: str | None = None,
    provisioning: bool = True,
    full_dynamic: bool = True,
    static_routing_mode: bool = False,
    nwc_mode: bool = False,
    windows: tuple[float, float] | None = None,
):
    """Run the quantum network scheduling simulation.

    Args:
        config (yaml or gml): Configuration file path in YAML or GML format.
        arrival_rate (float): Mean rate lambda for the Poisson arrival
            process. The number of applications is drawn from this rate over
            the observation horizon.
        inst_range (tuple[int, int]): Range (min, max) for the number of
            instances per application.
        epr_range (tuple[int, int]): Range (min, max) for the number of EPR
            pairs to generate per application.
        deadline_range (tuple[float, float]): Range (min, max) for the relative
            deadline budget of each application (deadline = release +
            deadline_budget).
        p_packet (float): Probability of a packet being generated.
        memory (float): Memory: number of independent link-generation trials
            per slot.
        p_swap (float): Probability of swapping an EPR pair in a single trial.
        time_slot_duration (float): Duration of a time slot in seconds.
        seed (int): Random seed for reproducibility of the simulation.
        output_dir (str): Directory where the results will be saved.
        windows (tuple[float, float] | None): Post-warm-up observation
            window as (min_time, max_time). In dynamic mode, max_time is used
            as the simulation horizon.
    Returns:
        tuple[bool, dict]:
            A tuple containing:
            - bool: whether the schedule is feasible
            - dict: summary metrics dictionary
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(4)
    rng, rng_arrivals, rng_routing = (
        np.random.default_rng(s) for s in child_seeds[:3]
    )
    ss_app_arrivals = child_seeds[3]

    # Generate network data and applications based on the configuration file
    fidelities = {}
    simple_paths = {}
    avg_deg = float("nan")
    diameter = float("nan")
    rates = None
    if graph == "waxman":
        nodes, edges, fidelities, rates, avg_deg, diameter = (
            generate_waxman_graph(rng=rng)
        )
        if not nodes or not edges:
            print("Failed to generate a connected Waxman graph.")
            return False, {}
    elif graph == "fat":
        nodes, edges, fidelities, rates, qpus, diameter = fat_tree()
        nodes = qpus
    elif graph == "clos":
        nodes, edges, fidelities, rates, qpus, diameter = clos()
        nodes = qpus
    elif graph == "gml":
        nodes, edges, distances, fidelities, rates, diameter = gml_data(config)
    bounds, simple_paths = fidelity_bounds_and_paths(
        nodes, fidelities, diameter + 2
    )
    all_links = {tuple(sorted((u, v))) for u, v in edges}

    # Poisson arrival times for each application over the observation horizon
    if float(arrival_rate) <= 0.0:
        raise ValueError("arrival_rate must be positive")
    windows_max = windows[1] if windows is not None else float("inf")
    mean_interarrival = 1.0 / float(arrival_rate)
    arrival_times = []
    t = 0.0
    while True:
        t += float(rng_arrivals.exponential(mean_interarrival))
        if t > windows_max:
            break
        arrival_times.append(t)
    app_specs = generate_n_apps(
        nodes,
        bounds,
        n_apps=len(arrival_times),
        inst_range=inst_range,
        epr_range=epr_range,
        deadline_range=deadline_range,
        rng=rng,
    )
    pga_rel_times = {
        app: arrival_times[i] for i, app in enumerate(app_specs.keys())
    }

    rng_arrivals_per_app = {
        app: np.random.default_rng(s)
        for app, s in zip(
            app_specs.keys(),
            ss_app_arrivals.spawn(len(app_specs)),
            strict=True,
        )
    }

    # Find feasible paths for each application based on fidelity/routing mode
    app_requests = {
        name: {
            "src": spec["src"],
            "dst": spec["dst"],
            "min_fidelity": spec.get("min_fidelity", 0.0),
            "instances": spec.get("instances", 0),
            "epr": spec.get("epr", 0),
            "deadline_budget": spec.get("deadline_budget", 0.0),
            "arrival_time": pga_rel_times[name],
        }
        for name, spec in app_specs.items()
    }
    total_apps = len(app_specs)
    routing_mode = str(routing)
    admitted_specs = {}
    static_routing_time = 0.0

    if static_routing_mode:
        _t0 = time.perf_counter()
        paths, app_e2e_fidelities = static_routing(app_requests, simple_paths)
        static_routing_time = time.perf_counter() - _t0
    elif full_dynamic:
        paths = {
            app: [[req["src"], req["dst"]]]
            for app, req in app_requests.items()
        }
        app_e2e_fidelities = {app: float("nan") for app in app_requests}
        static_routing_time = 0.0
    else:
        _t0 = time.perf_counter()
        paths, app_e2e_fidelities = find_feasible_path(
            edges,
            simple_paths,
            app_requests,
            fidelities,
            pga_rel_times=pga_rel_times,
            routing_mode=routing_mode,
            p_packet=p_packet,
            memory=memory,
            p_swap=p_swap,
            rates=rates,
            time_slot_duration=time_slot_duration,
            rng=rng_routing,
            provisioning=provisioning,
        )
        static_routing_time = time.perf_counter() - _t0

    admitted_paths = {
        app: path_list for app, path_list in paths.items() if path_list
    }
    admitted_specs = {app: app_specs[app] for app in admitted_paths.keys()}
    admitted_apps = len(admitted_specs)

    if admitted_apps == 0:
        return False, {}

    app_e2e_fidelities = {
        app: app_e2e_fidelities[app] for app in admitted_paths
    }

    app_specs = admitted_specs
    paths = admitted_paths

    single_path_cpt = 0
    two_path_cpt = 0
    app_request_rows = []
    for app, spec in app_specs.items():
        src, dst = spec["src"], spec["dst"]
        min_fid = spec.get("min_fidelity", 0.0)
        feasible_paths = [
            (fid, p)
            for fid, p in all_simple_paths(simple_paths, src, dst)
            if fid >= min_fid
        ]
        if len(feasible_paths) == 1:
            single_path_cpt += 1
        if len(feasible_paths) <= 2:
            two_path_cpt += 1

        if not verbose:
            continue
        n_disjoint = count_edge_disjoint_paths(feasible_paths)
        pga_params = {
            "p_packet": p_packet,
            "epr_pairs": int(spec["epr"]),
            "memory": memory,
            "p_swap": p_swap,
            "slot_duration": time_slot_duration,
        }
        feasible_durations = sorted(
            (
                (fid, path, duration)
                for fid, path, _, duration in compute_path_durations(
                    pga_params, rates, simple_paths=simple_paths,
                    src=src, dst=dst,
                )
                if fid >= min_fid
            ),
            key=lambda item: item[2],
        )
        shortest_duration = (
            feasible_durations[0][2] if feasible_durations else float("nan")
        )
        for fid, path, duration in feasible_durations:
            overhead_pct = (
                (duration - shortest_duration) / shortest_duration * 100.0
                if shortest_duration and np.isfinite(shortest_duration)
                else float("nan")
            )
            app_request_rows.append(
                {
                    "task": app,
                    "src_node": src,
                    "dst_node": dst,
                    "min_fidelity": float(min_fid),
                    "instances": int(spec.get("instances", 0)),
                    "epr": int(spec.get("epr", 0)),
                    "deadline_budget": float(spec.get("deadline_budget", 0.0)),
                    "arrival_time": float(pga_rel_times[app]),
                    "edge_disjoint_paths": n_disjoint,
                    "path": "-".join(map(str, path)),
                    "e2e_fidelity": float(fid),
                    "pga_duration": float(duration),
                    "overhead_pct": float(overhead_pct),
                }
            )
    single_path_share = single_path_cpt / total_apps if total_apps > 0 else 0.0
    two_path_share = two_path_cpt / total_apps if total_apps > 0 else 0.0

    initial_paths = {app: path_list[0] for app, path_list in paths.items()}
    epr_pairs = {name: spec["epr"] for name, spec in app_specs.items()}

    # Compute durations for each application
    durations = (
        {}
        if full_dynamic
        else compute_durations(
            initial_paths,
            epr_pairs,
            p_packet,
            memory,
            p_swap,
            time_slot_duration,
            rates=rates,
        )
    )

    pga_parameters = app_params_sim(
        initial_paths,
        app_specs,
        p_packet,
        memory,
        p_swap,
        time_slot_duration,
    )

    # Run simulation
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)

    routing_decision_cpt = None
    routing_decision_runtime = None
    defer_counts = None
    (
        df,
        pga_names,
        pga_release_times,
        link_utilization,
        link_waiting,
        routing_decision_cpt,
        routing_decision_runtime,
        defer_counts,
    ) = simulate_dynamic(
        app_specs,
        durations,
        pga_parameters,
        pga_rel_times,
        paths,
        rng,
        full_dynamic,
        provisioning,
        all_links,
        simple_paths,
        static_routing_mode,
        nwc_mode=nwc_mode,
        horizon_time=windows[1] if windows is not None else None,
        warmup_time=windows[0] if windows is not None else 0.0,
        rng_arrivals=rng_arrivals_per_app,
        instance_arrival_rate=instance_arrival_rate,
        rates=rates,
        app_e2e_fidelities=app_e2e_fidelities,
    )
    feasible = True
    if not full_dynamic:
        if static_routing_mode:
            routing_decision_cpt += 1
        else:
            routing_decision_cpt += len(app_specs)

    # Save results
    routing_decision_runtime = (
        routing_decision_runtime or 0.0
    ) + static_routing_time

    if link_utilization is None:
        link_utilization = {}
    if link_waiting is None:
        link_waiting = {}

    for link in all_links:
        link_utilization.setdefault(
            link,
            {
                "busy_time": 0.0,
                "utilization": 0.0,
            },
        )
        link_waiting.setdefault(
            link,
            {
                "total_waiting_time": 0.0,
                "block_events": 0,
                "acquisitions": 0,
            },
        )

    summary = save_results(
        df,
        pga_names,
        pga_release_times,
        app_specs,
        durations=durations,
        pga_network_paths=initial_paths,
        n_edges=len(edges),
        link_utilization=link_utilization,
        link_waiting=link_waiting,
        admitted_apps=admitted_apps,
        total_apps=total_apps,
        app_e2e_fidelities=app_e2e_fidelities,
        single_path_share=single_path_share,
        two_path_share=two_path_share,
        app_request_rows=app_request_rows,
        avg_deg=avg_deg,
        output_dir=output_dir,
        save_csv=save_csv,
        verbose=verbose,
        routing_decision_cpt=routing_decision_cpt,
        routing_decision_runtime=routing_decision_runtime,
        warmup=windows[0] if windows is not None else None,
        end_time=windows[1] if windows is not None else None,
        defer_counts=defer_counts,
    )
    del df
    return feasible, summary


def main():
    parser = argparse.ArgumentParser(
        description="Simulate scheduling of quantum network applications"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configurations/network/basic/2_equal_paths.gml",
        help="Path to YAML or GML config",
    )
    parser.add_argument(
        "--inst",
        "-i",
        type=int,
        default=100,
        help="Number of instances to generate per application",
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
        "--deadline",
        "-d",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[1.0, 1.0],
        help="Relative deadline budget per application: deadline = release"
        " + budget (e.g., --deadline 1.0 5.0)",
    )
    parser.add_argument(
        "--ppacket",
        "-pp",
        type=float,
        default=0.9,
        help="Probability of a packet being generated",
    )
    parser.add_argument(
        "--memory",
        "-m",
        type=int,
        default=1,
        help="Number of independent link-generation trials per slot in the"
        "multiplexed memory",
    )
    parser.add_argument(
        "--pswap",
        "-ps",
        type=float,
        default=0.5,
        help="Probability of swapping an EPR pair in a single trial",
    )
    parser.add_argument(
        "--slot-duration",
        "-sd",
        type=float,
        default=1e-4,
        help="Duration of a time slot in seconds",
    )
    parser.add_argument(
        "--arrival-rate",
        "-ar",
        type=float,
        default=1.0,
        help="Mean arrival rate (lambda) for the Poisson arrival process",
    )
    parser.add_argument(
        "--routing",
        "-r",
        type=str,
        choices=["shortest", "smallest", "least", "highest"],
        default="highest",
        help=(
            "Routing: 'shortest' (Dijkstra),'smallest' (smallest bottleneck),"
            "'least' (least total capacity), or 'highest' (highest "
            "E2E fidelity)."
        ),
    )
    parser.add_argument(
        "--graph",
        "-g",
        type=str,
        choices=["waxman", "fat", "clos", "gml"],
        default=None,
        help="Graph generator (e.g., 'waxman', 'fat', 'clos', 'gml')",
    )
    parser.add_argument(
        "--routing-strategy",
        "-rs",
        type=str,
        choices=["static", "hybrid", "rerouting", "dynamic", "nwc"],
        default=None,
        help="Routing strategy: 'static' (fixed static paths), 'hybrid' ("
        "static no rerouting), 'rerouting' (static rerouting),"
        "'dynamic' (work-conserving dynamic routing), or 'nwc' "
        "(non-work-conserving dynamic routing)",
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

    max_observation_window = 3 * args.inst * 10 * (1.0 / 10.0)
    warmup = 0.15 * max_observation_window
    windows = (warmup, max_observation_window)

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
        arrival_rate=args.arrival_rate,
        inst_range=args.inst,
        epr_range=args.epr,
        deadline_range=args.deadline,
        windows=windows,
        p_packet=args.ppacket,
        memory=args.memory,
        p_swap=args.pswap,
        time_slot_duration=args.slot_duration,
        seed=args.seed,
        output_dir=run_dir,
        routing=args.routing,
        graph=args.graph,
        provisioning=args.routing_strategy == "rerouting",
        full_dynamic=args.routing_strategy in ("dynamic", "nwc"),
        static_routing_mode=args.routing_strategy == "static",
        nwc_mode=args.routing_strategy == "nwc",
    )
    t1 = time.perf_counter()

    if not feasible:
        return

    runtime = t1 - t0
    print(f"Run time: {runtime:.3f} seconds\n")

    params = {
        "config": args.config,
        "arrival_rate": args.arrival_rate,
        "inst_range": args.inst,
        "epr_min": args.epr[0],
        "epr_max": args.epr[1],
        "deadline_min": args.deadline[0],
        "deadline_max": args.deadline[1],
        "windows_min": windows[0],
        "windows_max": windows[1],
        "p_packet": args.ppacket,
        "memory": args.memory,
        "p_swap": args.pswap,
        "time_slot_duration": args.slot_duration,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "run_number": run_number,
        "routing": args.routing,
        "routing_strategy": args.routing_strategy,
    }
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    path_results = os.path.join(run_dir, "pga_results.parquet")
    path_params = os.path.join(run_dir, "params.json")
    path_link_util = os.path.join(run_dir, "link_utilization.csv")
    path_link_wait = os.path.join(run_dir, "link_waiting.csv")
    print(f"Saved results to: {path_results}")
    print(f"Saved parameters to: {path_params}")
    print(f"Saved link utilization to: {path_link_util}")
    print(f"Saved link waiting metrics to: {path_link_wait}")


if __name__ == "__main__":
    main()
