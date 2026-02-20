import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter


# =============================================================================
# Parallelization
# =============================================================================
def parallelizable_tasks(
    paths_for_each_apps: dict[str, List[str] | None],
) -> dict[str, set[str]]:
    """Find parallelizable applications based on shared links of a
    quantum network.

    Args:
        paths_for_each_apps (dict[str, List[str]]): A dictionary where keys are
            application names and values are list of nodes describing the route
            used to run the applications, e.g. on a linear chain Alice-Rob-Bob:

                {
                    'A': ['Alice', 'Rob'],
                    'B': ['Rob', 'Bob'],
                }

    Returns:
        dict[str, set[str]]: A dictionary where keys are application names and
            values are sets of applications that can run in parallel with key
            application, based on shared links. From the example above, the
            output would be A and B can run in parallel since they do not share
            any links:

                {
                    'A': {'B'},
                    'B': {'A'},
                }
    """
    G = nx.Graph()
    conflicts = defaultdict(set)

    # Build conflict graph using undirected links along each path
    for app, v in paths_for_each_apps.items():
        G.add_node(app)
        if not v or len(v) < 2:
            continue
        edges_on_path = {
            tuple(sorted((u, v))) for u, v in zip(v[:-1], v[1:], strict=False)
        }
        for edge in edges_on_path:
            for other_app in conflicts[edge]:
                G.add_edge(app, other_app)
            conflicts[edge].add(app)

    g_complement = nx.complement(G)
    parallelizable_applications = {
        app: set(g_complement.neighbors(app)) for app in g_complement.nodes()
    }

    return parallelizable_applications


# =============================================================================
# Helper simulation functions
# =============================================================================
def app_params_sim(
    paths: dict[str, list[str]],
    app_specs: dict[str, dict[str, Any]],
    p_packet: float,
    memory: int,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
) -> dict[str, dict[str, float | int]]:
    """Prepare application parameters for simulation.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        app_specs (dict[str, dict[str, Any]]): Application metadata produced by
        ``generate_n_apps`` containing network endpoints and requirements.
        p_packet (float): Probability of a packet being generated.
        memory (int): Number of independent link-generation trials per slot.
        p_swap (float): Probability of swapping an EPR pair in a single trial.
        p_gen (float): Probability of generating an EPR pair in a single trial.
        time_slot_duration (float): Duration of a time slot in seconds.

    Returns:
        dict[str, dict[str, float | int]]: Mapping of application name to the
        parameters required by the simulator when instantiating PGAs.
    """
    sim_params = {}
    for key in paths.keys():
        spec = app_specs[key]
        sim_params[key] = {
            "p_packet": p_packet,
            "memory": memory,
            "p_swap": p_swap,
            "p_gen": p_gen,
            "epr_pairs": int(spec["epr"]),
            "slot_duration": time_slot_duration,
        }
    return sim_params


def build_default_sim_args(config: str, args: dict | None) -> dict:
    default_args = {
        "config": config,
        "inst_range": (100, 100),
        "epr_range": (2, 2),
        "period_range": (1, 1),
        "hyperperiod_cycles": 1000,
        "memory": 1000,
        "p_swap": 0.6,
        "p_gen": 1e-3,
        "fidelity_range": (0.6, 0.9),
        "routing": "capacity",
        "capacity_threshold": 0.8,
        "time_slot_duration": 1e-4,
    }
    if args:
        default_args.update(args)
    return default_args


def build_tasks(
    ppacket_values: Iterable[float],
    sim_per_point: int,
    seed_start: int,
    run_dir: str,
    default_kwargs: dict,
    n_apps_values: Iterable[int],
    keep_seed_outputs: bool = True,
) -> list[tuple[Any, ...]]:
    tasks = []
    seed_pool = [seed_start + offset for offset in range(sim_per_point)]
    n_apps_list = list(n_apps_values)
    if not n_apps_list:
        raise ValueError("n_apps_values must contain at least one value")
    for n_apps in n_apps_list:
        for p_packet in ppacket_values:
            for run_seed in seed_pool:
                tasks.append(
                    (
                        p_packet,
                        run_seed,
                        run_dir,
                        default_kwargs,
                        int(n_apps),
                        keep_seed_outputs,
                    )
                )
    return tasks


# =============================================================================
# Tracking
# =============================================================================
def compute_link_utilization(
    link_busy: Dict[Tuple[str, str], float],
    min_start: float,
    max_completion: float,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    if not link_busy:
        return {}

    horizon = 0.0
    if (
        np.isfinite(min_start)
        and np.isfinite(max_completion)
        and max_completion > min_start
    ):
        horizon = max_completion - min_start

    if horizon > 0.0:
        return {
            link: {
                "busy_time": busy,
                "utilization": busy / horizon,
            }
            for link, busy in link_busy.items()
        }

    return {
        link: {"busy_time": busy, "utilization": 0.0}
        for link, busy in link_busy.items()
    }


def track_link_waiting(
    waiting_time: float,
    wait_acc: Dict[Tuple[str, str], Dict[str, float]],
    blocking_links: List[Tuple[str, str]] | None = None,
) -> None:
    """Track waiting time statistics per link.

    Args:
        waiting_time (float): Waiting time per PGA.
        wait_acc (Dict[Tuple[str, str], Dict[str, float]]): Accumulator
        for waiting time statistics per link.
        blocking_links (List[Tuple[str, str]] | None): The specific link(s)
        that caused the waiting (with maximum busy time). If provided, waiting
        time is distributed equally among these links.
    """
    wait = max(0.0, float(waiting_time))
    if wait <= 0.0:
        return
    if blocking_links is None or len(blocking_links) == 0:
        return

    links_to_update = blocking_links
    w = wait / len(blocking_links)

    for link in links_to_update:
        pga_wait = wait_acc.setdefault(
            link,
            {
                "total_waiting_time": 0.0,
                "pga_waited": 0,
            },
        )
        pga_wait["total_waiting_time"] = pga_wait["total_waiting_time"] + w
        pga_wait["pga_waited"] = pga_wait["pga_waited"] + 1


# =============================================================================
# Results saving and summary
# =============================================================================
def save_results(
    df: pd.DataFrame,
    pga_names: List[str],
    pga_release_times: Dict[str, float],
    app_specs: Dict[str, Dict[str, Any]],
    n_edges: int,
    durations: Dict[str, float] | None = None,
    pga_network_paths: Dict[str, List[str]] | None = None,
    link_utilization: Dict[Tuple[str, str], Dict[str, float]] | None = None,
    link_waiting: Dict[Tuple[str, str], Dict[str, float | int]] | None = None,
    admitted_apps: int | None = None,
    total_apps: int | None = None,
    output_dir: str = "results",
    save_csv: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Save the results of PGA scheduling and execution to a CSV file and print
    a summary of the results.

    Args:
        df (DataFrame): DataFrame containing PGA results with columns:
            - pga: PGA identifier
            - arrival_time: Time when the PGA arrived
            - start_time: Time when the PGA started execution
            - burst_time: Total time required for the PGA to complete
            - completion_time: Time when the PGA completed execution
            - turnaround_time: Total time from arrival to completion
            - waiting_time: Total time the PGA waited before execution
            - status: Status of the PGA (e.g., "completed", "failed")
            - deadline: Deadline for the PGA (if applicable)
            - src_node: Source node of the PGA
            - dst_node: Destination node of the PGA
            - instances: Number of instances for the PGA
            - epr_pairs: Number of EPR pairs for the PGA
            - policy: Scheduling policy used for the PGA
        pga_names (List): List of all PGA names that should be present in the
            results.
        pga_release_times (Dict): Dictionary mapping PGA names to their
            relative release times, used to fill in missing PGAs.
        app_specs (Dict): Metadata for each application including endpoints,
            instances, requested EPR pairs, period, and policy.
        n_edges (int): Number of edges in the network graph.
        durations (Dict | None): Optional mapping of deterministic PGA
            durations per application.
        pga_network_paths (Dict | None): Length of network paths per
            application.
        link_utilization (Dict): Dictionary mapping links to busy time and
            utilization metrics.
        link_waiting (Dict | None): Dictionary mapping links to waiting
            metrics (total waiting time and number of PGAs that waited).
        output_dir (str): Directory where the results CSV file will be saved.
        save_csv (bool): Whether to save results to CSV files.
        verbose (bool): Whether to print summary statistics to stdout.

    Returns:
        Dict[str, float]: Dictionary containing summary metrics including
            admission_rate, makespan, throughput, completion ratios, and
            utilization statistics.
    """
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)

    missing = set(pga_names) - set(df["pga"])
    if missing:
        filler_rows = []
        for pga in missing:
            task = re.sub(r"\d+$", "", pga)
            filler_rows.append(
                {
                    "pga": pga,
                    "arrival_time": pga_release_times.get(pga, np.nan),
                    "start_time": np.nan,
                    "burst_time": np.nan,
                    "completion_time": np.nan,
                    "turnaround_time": np.nan,
                    "waiting_time": np.nan,
                    "status": "missing",
                }
            )
        df = pd.concat([df, pd.DataFrame(filler_rows)], ignore_index=True)

    df["task"] = df["pga"].astype(str).str.replace(r"\d+$", "", regex=True)
    app_names = list(app_specs.keys())
    if pga_network_paths:
        path_length = {
            app: max(0, len(path) - 1)
            for app, path in pga_network_paths.items()
            if path is not None
        }
    else:
        path_length = {}
    params = pd.DataFrame(
        {
            "task": app_names,
            "src_node": [app_specs[a]["src"] for a in app_names],
            "dst_node": [app_specs[a]["dst"] for a in app_names],
            "instances": [int(app_specs[a]["instances"]) for a in app_names],
            "pairs_requested": [int(app_specs[a]["epr"]) for a in app_names],
            "policy": [app_specs[a]["policy"] for a in app_names],
            "hops": [path_length.get(a, np.nan) for a in app_names],
            "pga_duration": [
                float(durations[a]) if durations and a in durations else np.nan
                for a in app_names
            ],
        }
    )

    df = df.merge(params, on="task", how="left").drop(columns="task")
    df = df.sort_values(by="completion_time").reset_index(drop=True)

    tot_reqs = df["pga"].nunique()
    df_ord = df.copy()
    df_ord["_order_time"] = df_ord["completion_time"].fillna(-np.inf)
    final = (
        df_ord.sort_values(["pga", "_order_time"])
        .groupby("pga", as_index=False)
        .tail(1)
        .drop(columns="_order_time")
    )
    completed_total = int((final["status"] == "completed").sum())
    drop_total = int((final["status"] == "drop").sum())
    failed_total = int((final["status"] == "failed").sum())

    arrival_min = df["arrival_time"].min()
    completion_max = df["completion_time"].max()
    makespan = completion_max - arrival_min

    if save_csv:
        csv_path = os.path.join(output_dir, "pga_results.csv")
        df.to_csv(csv_path, index=False)

        if verbose:
            print("\n=== Preview PGA Results ===")
            print(df.head(20).to_string(index=False))

    avg_link_utilization = float("nan")
    p90_link_utilization = float("nan")
    p95_link_utilization = float("nan")
    total_busy_time = float("nan")
    p90_link_avg_wait = float("nan")
    p95_link_avg_wait = float("nan")
    avg_queue_length = float("nan")
    p90_avg_queue_length = float("nan")
    p95_avg_queue_length = float("nan")
    link_waiting_path = None
    if link_utilization:
        link_util_rows = [
            {
                "link": f"{min(a, b)}-{max(a, b)}",
                "busy_time": metrics.get("busy_time", float("nan")),
                "utilization": metrics.get("utilization", float("nan")),
            }
            for (a, b), metrics in link_utilization.items()
        ]
        lk_ut_df = (
            pd.DataFrame(link_util_rows)
            .sort_values("utilization", ascending=False)
            .reset_index(drop=True)
        )

        busy_time_sum = lk_ut_df["busy_time"].sum()
        avg_link_utilization = float((busy_time_sum / makespan) / n_edges)
        p90_link_utilization = float(lk_ut_df["utilization"].quantile(0.9))
        p95_link_utilization = float(lk_ut_df["utilization"].quantile(0.95))
        total_busy_time = busy_time_sum

        if save_csv:
            link_util_path = os.path.join(output_dir, "link_utilization.csv")
            lk_ut_df.to_csv(link_util_path, index=False)

            if verbose:
                print("\n=== Link Utilization ===")
                print(lk_ut_df.to_string(index=False))

    if link_waiting:
        waiting_rows = [
            {
                "link": f"{min(a, b)}-{max(a, b)}",
                "total_waiting_time": waiting.get(
                    "total_waiting_time", float("nan")
                ),
                "pga_waited": waiting.get("pga_waited", 0),
            }
            for (a, b), waiting in link_waiting.items()
        ]
        for row in waiting_rows:
            w = row["pga_waited"]
            total_wait = row["total_waiting_time"]
            row["avg_wait"] = total_wait / w if w and w > 0 else 0.0
            row["avg_queue_length"] = (
                total_wait / makespan if makespan and makespan > 0 else 0.0
            )
        waiting_df = (
            pd.DataFrame(waiting_rows)
            .sort_values("total_waiting_time", ascending=False)
            .reset_index(drop=True)
        )

        avg_wait_series = pd.to_numeric(
            waiting_df.get("avg_wait"), errors="coerce"
        ).dropna()
        if not avg_wait_series.empty:
            p90_link_avg_wait = float(avg_wait_series.quantile(0.9))
            p95_link_avg_wait = float(avg_wait_series.quantile(0.95))

        avg_queue_series = pd.to_numeric(
            waiting_df.get("avg_queue_length"), errors="coerce"
        ).dropna()
        if not avg_queue_series.empty:
            avg_queue_length = float(avg_queue_series.mean())
            p90_avg_queue_length = float(avg_queue_series.quantile(0.9))
            p95_avg_queue_length = float(avg_queue_series.quantile(0.95))
        if save_csv:
            link_waiting_path = os.path.join(output_dir, "link_waiting.csv")
            waiting_df.to_csv(link_waiting_path, index=False)

            if verbose:
                print("\n=== Link Waiting ===")
                print(waiting_df.to_string(index=False))

    total = len(df)
    admission_rate = float("nan")
    if admitted_apps is not None and total_apps is not None and total_apps > 0:
        admission_rate = float(admitted_apps) / float(total_apps)
    completed_final = final.loc[final["status"] == "completed"]
    throughput = completed_total / makespan
    pga_durations = (
        np.array(list(durations.values()), dtype=float)
        if durations
        else np.array([], dtype=float)
    )
    avg_wait = (
        completed_final["waiting_time"].mean()
        if not completed_final.empty
        else float("nan")
    )
    max_wait = (
        completed_final["waiting_time"].max()
        if not completed_final.empty
        else float("nan")
    )
    avg_turnaround = (
        completed_final["turnaround_time"].mean()
        if not completed_final.empty
        else float("nan")
    )
    max_turnaround = (
        completed_final["turnaround_time"].max()
        if not completed_final.empty
        else float("nan")
    )
    avg_pga_duration = float(np.sum(pga_durations)) / tot_reqs
    completed_ratio = completed_total / tot_reqs if tot_reqs else float("nan")
    drop_ratio = drop_total / tot_reqs if tot_reqs else float("nan")
    failed_ratio = failed_total / tot_reqs if tot_reqs else float("nan")
    defer_count = len(df.loc[df["status"] == "defer"])
    retry_count = len(df.loc[df["status"] == "retry"])
    avg_defer_per_pga = defer_count / tot_reqs if tot_reqs else float("nan")
    avg_retry_per_pga = retry_count / tot_reqs if tot_reqs else float("nan")
    avg_hops = params["hops"].mean() if not params.empty else float("nan")
    admitted_min_fidelities = [
        app_specs[app].get("min_fidelity", float("nan"))
        for app in app_specs.keys()
    ]
    avg_min_fidelity = (
        float(np.mean(admitted_min_fidelities))
        if admitted_min_fidelities
        else float("nan")
    )

    if verbose:
        print("\n=== Summary ===")
        print(f"Total PGAs       : {total}")

    tmp = df.copy()
    tmp["task"] = tmp["pga"].astype(str).str.replace(r"\d+$", "", regex=True)
    expected_tasks = sorted({re.sub(r"\d+$", "", j) for j in pga_names})
    per_task = (
        tmp.groupby(["task", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(expected_tasks, fill_value=0)
    )

    for col in ["completed", "failed"]:
        if col not in per_task.columns:
            per_task[col] = 0

    tasks_sorted = sorted(per_task.index, key=lambda x: (len(x), x))
    if verbose:
        for task in tasks_sorted:
            row = per_task.loc[task]
            completed = int(row.get("completed", 0))
            failed = int(row.get("failed", 0))
            print(f"    {task:<4} completed: {completed}, failed: {failed}")

        print(f"Admission rate   : {admission_rate:.4f}")
        print(f"Completion time  : {makespan:.4f}")
        print(f"Throughput       : {throughput:.4f} completed PGAs/s")
        print(f"Completion ratio : {completed_ratio:.4f}")
        print(f"Failed ratio     : {failed_ratio:.4f}")
        print(f"Drop ratio       : {drop_ratio:.4f}")
        print(f"Avg defer per PGA: {avg_defer_per_pga:.4f}")
        print(f"Avg retry per PGA: {avg_retry_per_pga:.4f}")
        print(f"Avg waiting time : {avg_wait:.4f}")
        print(f"Max waiting time : {max_wait:.4f}")
        print(f"P90 link avg_wait : {p90_link_avg_wait:.4f}")
        print(f"P95 link avg_wait : {p95_link_avg_wait:.4f}")
        print(f"Avg queue length : {avg_queue_length:.4f}")
        print(f"P90 avg_queue_length : {p90_avg_queue_length:.4f}")
        print(f"P95 avg_queue_length : {p95_avg_queue_length:.4f}")
        print(f"Avg turnaround   : {avg_turnaround:.4f}")
        print(f"Max turnaround   : {max_turnaround:.4f}")
        print(f"Avg hops         : {avg_hops:.4f}")
        print(f"Avg min fidelity : {avg_min_fidelity:.4f}")
        print(f"Avg PGA duration : {avg_pga_duration:.4f}")
        print(f"Total busy time  : {total_busy_time:.4f}")
        print(f"Avg link utilization : {avg_link_utilization:.4f}")
        print(f"P90 link utilization : {p90_link_utilization:.4f}")
        print(f"P95 link utilization : {p95_link_utilization:.4f}")

    summary_metrics = {
        "admission_rate": float(admission_rate),
        "makespan": float(makespan),
        "throughput": float(throughput),
        "completed_ratio": float(completed_ratio),
        "failed_ratio": float(failed_ratio),
        "drop_ratio": float(drop_ratio),
        "avg_defer_per_pga": float(avg_defer_per_pga),
        "avg_retry_per_pga": float(avg_retry_per_pga),
        "avg_waiting_time": float(avg_wait),
        "max_waiting_time": float(max_wait),
        "avg_turnaround_time": float(avg_turnaround),
        "max_turnaround_time": float(max_turnaround),
        "avg_hops": float(avg_hops),
        "avg_min_fidelity": float(avg_min_fidelity),
        "avg_pga_duration": float(avg_pga_duration),
        "total_busy_time": float(total_busy_time),
        "avg_link_utilization": float(avg_link_utilization),
        "p90_link_utilization": float(p90_link_utilization),
        "p95_link_utilization": float(p95_link_utilization),
        "p90_link_avg_wait": float(p90_link_avg_wait),
        "p95_link_avg_wait": float(p95_link_avg_wait),
        "avg_queue_length": float(avg_queue_length),
        "p90_avg_queue_length": float(p90_avg_queue_length),
        "p95_avg_queue_length": float(p95_avg_queue_length),
    }

    if save_csv:
        overall_df = pd.DataFrame([summary_metrics])
        overall_path = os.path.join(output_dir, "summary.csv")
        overall_df.to_csv(overall_path, index=False)

    for col in ["defer", "retry", "drop"]:
        if col not in per_task.columns:
            per_task[col] = 0

    per_task_cols = ["completed", "failed", "defer", "retry", "drop"]
    per_task_df = (
        per_task.loc[tasks_sorted, per_task_cols]
        .reset_index()
        .rename(columns={"task": "task_name"})
    )
    per_task_df = per_task_df.merge(
        params.rename(columns={"task": "task_name"})[
            ["task_name", "pga_duration"]
        ],
        on="task_name",
        how="left",
    )
    if save_csv:
        per_task_path = os.path.join(output_dir, "summary_per_app.csv")
        per_task_df.to_csv(per_task_path, index=False)

    return summary_metrics


# =============================================================================
# Application and network generation
# =============================================================================
def gml_data(
    gml_file: str,
) -> Tuple[list, list, dict[tuple, float], dict[tuple, float]]:
    """Extracts nodes, edges, and distances from a GML file.

    Args:
        gml_file (str): Path to the GML file.

    Returns:
        nodes (list): List of nodes.
        edges (list): List of edges (source, target).
        distances (dict[tuple, float]): Dict mapping edges
        to distances.
        fidelity (dict[tuple, float]): Dict mapping directed edges to
        fidelities.
    """
    G = nx.read_gml(gml_file)

    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda edge: (str(edge[0]), str(edge[1])))
    distances = {
        (u, v): float(data.get("dist", 0.0))
        for u, v, data in G.edges(data=True)
    }
    end_nodes = sorted(nx.k_core(G).nodes(), key=str)
    fidelities = {}
    F_min = 0.5
    L_max = max(distances.values(), default=0.0)
    L_dep = (
        -L_max / np.log((4 * F_min - 1) / 3)
        if L_max > 0.0
        else float("inf")
    )

    for u, v, data in G.edges(data=True):
        L = float(data.get("dist", 0.0))
        f = (1 + 3 * np.exp(-L / L_dep)) / 4
        data["fidelity"] = f
        fidelities[(u, v)] = f

    return nodes, edges, distances, fidelities, end_nodes


def generate_n_apps(
    end_nodes: list,
    bounds: dict[tuple, tuple],
    n_apps: int,
    inst_range: tuple[int, int],
    epr_range: tuple[int, int],
    period_range: tuple[float, float],
    list_policies: list[str],
    rng: np.random.Generator,
) -> Dict[str, Dict[str, Any]]:
    """Generates a specified number of applications with random parameters.

    Args:
        nodes (list): List of available nodes in the network.
        n_apps (int): Number of applications to generate.
        inst_range (tuple[int, int]): Range (min, max) for the number of
        instances for each application.
        epr_range (tuple[int, int]): Range (min, max) for the number of EPR
        pairs for each application.
        period_range (tuple[float, float]): Range (min, max) for the period
        of each application.
        fidelity_range (tuple[float, float]): Range (min, max) for the minimum
        fidelity of each application.
        list_policies (list[str], optional): List of policies to assign to
        each application.
        rng (np.random.Generator): Random number generator for reproducibility.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of application name to its metadata,
        including endpoints, number of instances, requested EPR pairs, period,
        and policy.
    """
    apps = {}

    while len(apps) < n_apps:
        src, dst = rng.choice(end_nodes, 2, replace=False).tolist()
        min_fidelity, max_fidelity = fidelity_bounds(bounds, src, dst)
        min_fidelity = max(min_fidelity, 0.5)
        if max_fidelity <= min_fidelity:
            continue
        rand_min_fidelity = float(rng.uniform(min_fidelity, max_fidelity))
        name_app = get_column_letter(len(apps) + 1)
        rand_instance = int(rng.integers(inst_range[0], inst_range[1] + 1))
        rand_epr_pairs = int(rng.integers(epr_range[0], epr_range[1] + 1))
        rand_period = float(rng.uniform(period_range[0], period_range[1]))
        rand_policy = rng.choice(list_policies, 1, replace=False).item()

        apps[name_app] = {
            "src": src,
            "dst": dst,
            "instances": rand_instance,
            "epr": rand_epr_pairs,
            "period": rand_period,
            "min_fidelity": rand_min_fidelity,
            "policy": rand_policy,
        }

    return apps


# =============================================================================
# Directory management
# =============================================================================
def ppacket_dirname(value: float) -> str:
    label = f"{value:.6f}".rstrip("0").rstrip(".")
    if not label:
        label = "0"
    return f"ppacket_{label.replace('.', '_')}"


def prepare_run_dir(
    output_dir: str,
    ppacket_values: Iterable[float],
    keep_seed_outputs: bool = True,
) -> tuple[str, str]:
    base_output = output_dir or "results"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_output, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    if keep_seed_outputs:
        for p_val in ppacket_values:
            subdir = os.path.join(run_dir, ppacket_dirname(p_val))
            os.makedirs(subdir, exist_ok=True)
    return run_dir, timestamp


# =============================================================================
# Retrievial
# =============================================================================
def fidelity_bounds(
    bounds: Dict[Tuple[str, str], Tuple[float, float]], src: str, dst: str
) -> Tuple[float, float]:
    return bounds[(src, dst) if src < dst else (dst, src)]


def all_simple_paths(
    paths: Dict[Tuple[str, str], List[Tuple[float, Tuple[str, ...]]]],
    src: str,
    dst: str,
) -> List[Tuple[float, Tuple[str, ...]]]:
    return paths.get((src, dst) if src < dst else (dst, src), [])
