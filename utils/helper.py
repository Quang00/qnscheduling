import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

from scheduling.pga import duration_pga


def shortest_paths(
    edges: List[Tuple[str, str]], app_requests: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """Find shortest paths for each applications in a quantum network graph
    represented by edges.

    Args:
        edges (List[Tuple[str, str]]): List of edges in the quantum network,
            where each edge is a tuple of nodes (src, dst).
        app_requests (Dict[str, Dict[str, Any]]): A dictionary where keys are
            application names and values are dictionaries containing source and
            destination nodes (src, dst) for the application. For example:

                {
                    'A': {'src': 'Alice', 'dst': 'Bob'},
                    'B': {'src': 'Alice', 'dst': 'Charlie'},
                    'C': {'src': 'Charlie', 'dst': 'David'},
                    'D': {'src': 'Bob', 'dst': 'David'}
                }

    Returns:
        Dict[str, List[str]]: A dictionary where keys are application names and
        values are lists of nodes representing the shortest path from source to
        destination for that application.
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    return {
        application: nx.shortest_path(G, req['src'], req['dst'])
        for application, req in app_requests.items()
    }


def find_feasible_path(
    edges: List[Tuple[str, str]],
    app_requests: Dict[str, Dict[str, Any]],
    fidelities: Dict[Tuple[str, str], float] | None,
    pga_rel_times: Dict[str, float] | None = None,
    routing_mode: str = "shortest",
    capacity_threshold: float = 0.8,
    p_packet: float | None = None,
    memory: int = 1,
    p_swap: float = 0.6,
    p_gen: float = 0.001,
    time_slot_duration: float = 1e-4,
) -> Dict[str, List[str] | None]:
    """Assign a feasible path for each application in a quantum network graph
    based on minimum fidelity threshold. The fidelity constraint is transformed
    into a maximum path length (L_max) from the paper Chakraborty et al.,
    "Entanglement Distribution in a Quantum Network: A Multicommodity
    Flow-Based Approach", (2020). The path with the fewest hops that meets the
    fidelity requirement is selected. The routing mode can be set to "shortest"
    or "capacity". In "capacity" mode, edges that have reached the capacity
    threshold are excluded from path selection.

    Args:
        edges (List[Tuple[str, str]]): List of edges in the quantum network,
            where each edge is a tuple of nodes (src, dst).
        app_requests (Dict[str, Dict[str, Any]]): A dictionary where keys are
            application names and values are dictionaries containing source and
            destination nodes (src, dst) and minimum fidelity (min_fidelity)
            for the application.
        fidelities (Dict[Tuple[str, str], float]): Per-edge fidelities in
            the quantum network, where keys are directed edges (src, dst) and
            values are the fidelities of those edges.
        routing_mode (str, optional): Routing mode, either "shortest" or
            "capacity". In "capacity" mode, edges that have reached the
            capacity threshold are excluded from path selection.
        capacity_threshold (float, optional): Capacity threshold for edges in
            "capacity" routing mode. Edges with utilization above this
            threshold are excluded from path selection.
        p_packet (float, optional): Probability of a packet being generated.
        memory (int, optional): Number of independent link-generation trials
            per slot.
        p_swap (float, optional): Probability of swapping an EPR pair in a
            single trial.
        p_gen (float, optional): Probability of generating an EPR pair in a
            single trial.
        time_slot_duration (float, optional): Duration of a time slot in
            seconds.

    Returns:
        Dict[str, List[str] | None]: A dictionary where keys are application
        names and values are lists of nodes representing the feasible path from
        source to destination for that application, or None if no feasible
        path exists.
    """
    initial_fidelity = min(fidelities.values())
    base_graph = nx.Graph()
    base_graph.add_edges_from(edges)
    ret = {}
    capacity = defaultdict(float)

    apps_ordered = list(app_requests.keys())
    if pga_rel_times is not None:
        apps_ordered.sort(
            key=lambda app: (float(pga_rel_times.get(app, 0.0)), str(app))
        )

    for app in apps_ordered:
        req = app_requests[app]
        src = req['src']
        dst = req['dst']
        min_fidelity = req['min_fidelity']

        if min_fidelity <= 0.25:
            ret[app] = None
            continue

        if routing_mode == "capacity":
            usable_edges = [
                (u, v)
                for (u, v) in edges
                if capacity[tuple(sorted((u, v)))] < capacity_threshold
            ]
            G = nx.Graph()
            G.add_edges_from(usable_edges)
        else:
            G = base_graph

        if not nx.has_path(G, src, dst):
            ret[app] = None
            continue

        selected_path = None
        selected_delta = 0.0
        L_max = math.floor(
            math.log((4 * min_fidelity - 1) / 3)
            / math.log((4 * initial_fidelity - 1) / 3)
        )
        for path in nx.shortest_simple_paths(G, src, dst):
            L = len(path) - 1
            if L > L_max:
                continue
            if routing_mode == "capacity":
                n_swaps = max(0, len(path) - 2)
                pga_duration = duration_pga(
                    p_packet=p_packet,
                    epr_pairs=req['epr'],
                    n_swap=n_swaps,
                    memory=memory,
                    p_swap=p_swap,
                    p_gen=p_gen,
                    time_slot_duration=time_slot_duration,
                )
                delta = float(pga_duration) / float(req['period'])
                links = [
                    tuple(sorted((u, v)))
                    for u, v in zip(path[:-1], path[1:], strict=False)
                ]
                if any(
                    capacity[lk] + delta > capacity_threshold for lk in links
                ):
                    continue
                selected_delta = delta
            selected_path = path
            break

        if selected_path is None:
            ret[app] = None
            continue

        if routing_mode == "capacity":
            for u, v in zip(
                selected_path[:-1], selected_path[1:], strict=False
            ):
                link = tuple(sorted((u, v)))
                capacity[link] += selected_delta
        ret[app] = selected_path
    return ret


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
    for app, nodes in paths_for_each_apps.items():
        G.add_node(app)
        if not nodes or len(nodes) < 2:
            continue
        edges_on_path = {
            tuple(sorted((u, v)))
            for u, v in zip(nodes[:-1], nodes[1:], strict=False)
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


def track_link_waiting(
    route_links: List[Tuple[str, str]] | None,
    waiting_time: float,
    wait_acc: Dict[Tuple[str, str], Dict[str, float]],
) -> None:
    """Track waiting time statistics per link.

    Args:
        route_links (List[Tuple[str, str]] | None): List of links.
        waiting_time (float): Waiting time per PGA.
        wait_acc (Dict[Tuple[str, str], Dict[str, float]]): Accumulator
        for waiting time statistics per link.
    """
    wait = max(0.0, float(waiting_time))
    for link in route_links:
        pga_wait = wait_acc.setdefault(
            link,
            {
                "total_waiting_time": 0.0,
                "pga_waited": 0,
            },
        )
        pga_wait["total_waiting_time"] = pga_wait["total_waiting_time"] + wait
        pga_wait["pga_waited"] = pga_wait["pga_waited"] + 1


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
) -> None:
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
    """
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
    deadline_mask = (
        final.get("deadline").notna() if "deadline" in final else None
    )
    if deadline_mask is None:
        miss_total = 0
        tot_deadline_reqs = 0
    else:
        tot_deadline_reqs = int(deadline_mask.sum())
        miss_total = int(
            ((final["status"] != "completed") & deadline_mask).sum()
        )
    failed_total = int((final["status"] == "failed").sum())

    arrival_min = df["arrival_time"].min()
    completion_max = df["completion_time"].max()
    makespan = completion_max - arrival_min

    csv_path = os.path.join(output_dir, "pga_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Preview PGA Results ===")
    print(df.head(20).to_string(index=False))

    avg_link_utilization = float("nan")
    p90_link_utilization = float("nan")
    p95_link_utilization = float("nan")
    total_busy_time = float("nan")
    p90_link_avg_wait = float("nan")
    p95_link_avg_wait = float("nan")
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
        link_util_df = (
            pd.DataFrame(link_util_rows)
            .sort_values("utilization", ascending=False)
            .reset_index(drop=True)
        )

        busy_time_sum = link_util_df["busy_time"].sum()
        avg_link_utilization = float((busy_time_sum/makespan) / n_edges)
        p90_link_utilization = float(
            link_util_df["utilization"].quantile(0.9)
        )
        p95_link_utilization = float(
            link_util_df["utilization"].quantile(0.95)
        )
        total_busy_time = busy_time_sum

        link_util_path = os.path.join(output_dir, "link_utilization.csv")
        link_util_df.to_csv(link_util_path, index=False)

        print("\n=== Link Utilization ===")
        print(link_util_df.to_string(index=False))

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
            waited = row["pga_waited"]
            total_wait = row["total_waiting_time"]
            row["avg_wait"] = (
                total_wait / waited if waited and waited > 0 else 0.0
            )
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
        link_waiting_path = os.path.join(output_dir, "link_waiting.csv")
        waiting_df.to_csv(link_waiting_path, index=False)

        print("\n=== Link Waiting ===")
        print(waiting_df.to_string(index=False))

    total = len(df)
    admission_rate = float("nan")
    if admitted_apps is not None and total_apps is not None and total_apps > 0:
        admission_rate = float(admitted_apps) / float(total_apps)
    completed_final = final.loc[final["status"] == "completed"]
    completed_burst = completed_final["burst_time"].sum()
    useful_util = completed_burst / makespan if makespan > 0 else float("nan")
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
    total_pga_duration = (
        float(np.sum(pga_durations)) if pga_durations.size else float("nan")
    )
    completed_ratio = completed_total / tot_reqs if tot_reqs else float("nan")
    drop_ratio = drop_total / tot_reqs if tot_reqs else float("nan")
    deadline_miss_rate = (
        miss_total / tot_deadline_reqs if tot_deadline_reqs else float("nan")
    )
    failed_ratio = failed_total / tot_reqs if tot_reqs else float("nan")
    defer_prob = (
        df.loc[df["status"] == "defer", "pga"].nunique() / tot_reqs
        if tot_reqs
        else float("nan")
    )
    retry_prob = (
        df.loc[df["status"] == "retry", "pga"].nunique() / tot_reqs
        if tot_reqs
        else float("nan")
    )
    admitted_apps = completed_final["pga"].astype(str).str.replace(
        r"\d+$", "", regex=True
    ).unique()
    admitted_hops = params.loc[params["task"].isin(admitted_apps), "hops"]
    avg_hops = (
        admitted_hops.mean()
        if not admitted_hops.empty
        else float("nan")
    )
    admitted_min_fidelities = [
        app_specs[app].get("min_fidelity", float("nan"))
        for app in admitted_apps
        if app in app_specs
    ]
    avg_min_fidelity = (
        float(np.mean(admitted_min_fidelities))
        if admitted_min_fidelities
        else float("nan")
    )

    print("\n=== Summary ===")
    print(f"Admission rate   : {admission_rate:.4f}")
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
    for task in tasks_sorted:
        row = per_task.loc[task]
        n_completed = int(row.get("completed", 0))
        n_failed = int(row.get("failed", 0))
        print(f"    {task:<4} completed: {n_completed}, failed: {n_failed}")

    print(f"Completion time  : {makespan:.4f}")
    print(f"Throughput       : {throughput:.4f} completed PGAs/s")
    print(f"Completed ratio  : {completed_ratio:.4f}")
    print(f"Failed ratio     : {failed_ratio:.4f}")
    print(f"Deadline miss rate : {deadline_miss_rate:.4f}")
    print(f"Drop ratio       : {drop_ratio:.4f}")
    print(f"Deferral prob    : {defer_prob:.4f}")
    print(f"Retry prob       : {retry_prob:.4f}")
    print(f"Avg waiting time : {avg_wait:.4f}")
    print(f"Max waiting time : {max_wait:.4f}")
    print(f"P90 link avg_wait : {p90_link_avg_wait:.4f}")
    print(f"P95 link avg_wait : {p95_link_avg_wait:.4f}")
    print(f"Avg turnaround   : {avg_turnaround:.4f}")
    print(f"Max turnaround   : {max_turnaround:.4f}")
    print(f"Avg hops         : {avg_hops:.4f}")
    print(f"Avg min fidelity : {avg_min_fidelity:.4f}")
    print(f"Total PGA duration : {total_pga_duration:.4f}")
    print(f"Total busy time  : {total_busy_time:.4f}")
    print(f"Avg link utilization : {avg_link_utilization:.4f}")
    print(f"P90 link utilization : {p90_link_utilization:.4f}")
    print(f"P95 link utilization : {p95_link_utilization:.4f}")
    print(f"Useful utilization : {useful_util:.4f}")

    overall_df = pd.DataFrame(
        [
            {
                "admission_rate": float(admission_rate),
                "makespan": float(makespan),
                "throughput": float(throughput),
                "completed_ratio": float(completed_ratio),
                "failed_ratio": float(failed_ratio),
                "deadline_miss_rate": float(deadline_miss_rate),
                "drop_ratio": float(drop_ratio),
                "defer_prob": float(defer_prob),
                "retry_prob": float(retry_prob),
                "defer_ratio": float(defer_prob),
                "retry_ratio": float(retry_prob),
                "deadline_miss": int(miss_total),
                "avg_waiting_time": float(avg_wait),
                "max_waiting_time": float(max_wait),
                "avg_turnaround_time": float(avg_turnaround),
                "max_turnaround_time": float(max_turnaround),
                "avg_hops": float(avg_hops),
                "avg_min_fidelity": float(avg_min_fidelity),
                "total_pga_duration": float(total_pga_duration),
                "total_busy_time": float(total_busy_time),
                "avg_link_utilization": float(avg_link_utilization),
                "p90_link_utilization": float(p90_link_utilization),
                "p95_link_utilization": float(p95_link_utilization),
                "p90_link_avg_wait": float(p90_link_avg_wait),
                "p95_link_avg_wait": float(p95_link_avg_wait),
                "useful_utilization": float(useful_util),
            }
        ]
    )
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
    per_task_path = os.path.join(output_dir, "summary_per_app.csv")
    per_task_df.to_csv(per_task_path, index=False)


def gml_data(
    gml_file: str,
    rng: np.random.Generator,
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

    for _, _, data in G.edges(data=True):
        data["fidelity"] = float(rng.random())
        data["fidelity"] = 0.95

    nodes = list(G.nodes())
    edges = list(G.edges())
    distances = {(u, v): data.get("dist") for u, v, data in G.edges(data=True)}
    fidelities: dict[tuple, float] = {}
    for u, v, data in G.edges(data=True):
        f = float(data["fidelity"])
        fidelities[(u, v)] = f

    return nodes, edges, distances, fidelities


def generate_n_apps(
    nodes: list,
    n_apps: int,
    inst_range: tuple[int, int],
    epr_range: tuple[int, int],
    period_range: tuple[float, float],
    fidelity_range: tuple[float, float],
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

    for i in range(n_apps):
        name_app = get_column_letter(i + 1)
        src, dst = rng.choice(nodes, 2, replace=False).tolist()
        rand_instance = int(rng.integers(inst_range[0], inst_range[1] + 1))
        rand_epr_pairs = int(rng.integers(epr_range[0], epr_range[1] + 1))
        rand_period = float(rng.uniform(period_range[0], period_range[1]))
        rand_min_fidelity = float(
            rng.uniform(fidelity_range[0], fidelity_range[1])
        )
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


def edges_delay(
    distances: dict[tuple, float], c_fiber: float = 200_000.0
) -> dict[tuple, float]:
    """Compute the delay for each edge based on its distance.

    Args:
        distances (dict[tuple, float]): A dictionary mapping edges (as tuples)
        to their distances.
        c_fiber (float, optional): Speed of light in fiber (in km/s).

    Returns:
        dict[tuple, float]: A dictionary mapping edges (as tuples) to their
        delays.
    """
    delay_map = {(a, b): dist / c_fiber for (a, b), dist in distances.items()}
    for (a, b), delay in list(delay_map.items()):
        delay_map[(b, a)] = delay
    return delay_map


def sum_path_delay(route: list[str], delay_map: dict[tuple, float]) -> float:
    """Summing the total delay along a given route based on a delay map.

    Args:
        route (list[str]): A list of nodes representing the route.
        delay_map (dict[tuple, float]): A mapping of edges to their delays.

    Returns:
        float: The total delay along the route.
    """
    total = 0.0
    for u, v in zip(route[:-1], route[1:], strict=False):
        total += max(0.0, delay_map.get((u, v), delay_map.get((v, u), 0.0)))
    return total


def _lcm(a: int, b: int) -> int:
    """Compute the least common multiple (LCM) of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The least common multiple of a and b.
    """
    return 0 if (a == 0 or b == 0) else abs(a // gcd(a, b) * b)


def hyperperiod(periods: dict[str, float]) -> float:
    """Compute the hyperperiod of a set of periods.

    Args:
        periods (dict[str, float]): A dictionary mapping PGA names to their
        periods.

    Returns:
        float: The hyperperiod of the given periods.
    """
    positive_periods = [
        float(period) for period in periods.values() if float(period) > 0.0
    ]
    if not positive_periods:
        return 0.0

    fracs = [
        Fraction(period).limit_denominator()
        for period in positive_periods
    ]
    if not fracs:
        return 0.0

    D = reduce(_lcm, (frac.denominator for frac in fracs))
    nums = [frac.numerator * (D // frac.denominator) for frac in fracs]
    hyperperiod_ticks = reduce(_lcm, nums)

    try:
        value = hyperperiod_ticks / D
    except OverflowError:
        return max(positive_periods)

    if not math.isfinite(value):
        return max(positive_periods)

    MAX_HYPERPERIOD_SECONDS = 1e6
    if value > MAX_HYPERPERIOD_SECONDS:
        return max(positive_periods)

    return value


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


def build_default_sim_args(config: str, args: dict | None) -> dict:
    default_args = {
        "config": config,
        "inst_range": (100, 100),
        "epr_range": (2, 2),
        "period_range": (1, 1),
        "hyperperiod_cycles": 2000,
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
    simulations_per_point: int,
    seed_start: int,
    run_dir: str,
    default_kwargs: dict,
    n_apps_values: Iterable[int],
    keep_seed_outputs: bool = True,
) -> list[tuple[Any, ...]]:
    tasks = []
    seed_pool = [
        seed_start + offset for offset in range(simulations_per_point)
    ]
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


def aggregate_metric(
    data: pd.DataFrame,
    column: str,
    prefix: str,
    clip: tuple[float | None, float | None] | None = None,
    prefixed_columns: bool = True,
) -> pd.DataFrame:
    mean_col = f"mean_{prefix}"
    std_col = f"std_{prefix}"
    if prefixed_columns:
        sem_col, ci_col, lower_col, upper_col = (
            f"sem_{prefix}",
            f"ci95_{prefix}",
            f"lower_{prefix}",
            f"upper_{prefix}",
        )
    else:
        sem_col, ci_col, lower_col, upper_col = "sem", "ci95", "lower", "upper"

    ordered_cols = [
        "p_packet",
        "total",
        mean_col,
        std_col,
        "count",
        sem_col,
        ci_col,
        lower_col,
        upper_col,
    ]
    metric_df = data.dropna(subset=[column])
    if metric_df.empty:
        return pd.DataFrame(columns=ordered_cols)

    grouped = (
        metric_df.groupby("p_packet", as_index=False)
        .agg(
            total=(column, "sum"),
            mean=(column, "mean"),
            std=(column, "std"),
            count=(column, "count"),
        )
        .assign(
            sem=lambda df: df["std"] / np.sqrt(df["count"].clip(lower=1)),
        )
    )

    grouped["total"] = grouped["total"].astype(float)
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["sem"] = grouped["sem"].fillna(0.0)

    grouped["ci95"] = grouped["sem"] * 1.96
    grouped["lower"] = grouped["mean"] - grouped["ci95"]
    grouped["upper"] = grouped["mean"] + grouped["ci95"]

    if clip is not None:
        lo, hi = clip
        grouped["lower"] = grouped["lower"].clip(lower=lo, upper=hi)
        grouped["upper"] = grouped["upper"].clip(lower=lo, upper=hi)

    grouped["count"] = grouped["count"].astype(int)
    grouped = grouped.rename(
        columns={
            "mean": mean_col,
            "std": std_col,
            "sem": sem_col,
            "ci95": ci_col,
            "lower": lower_col,
            "upper": upper_col,
        }
    )
    return grouped[ordered_cols]


def generate_metadata(
    run_dir: str,
    timestamp: str,
    ppacket_values: Iterable[float],
    simulations_per_point: int,
    seed_start: int,
    config: str,
    save_path: str,
    raw_csv_path: str,
    default_kwargs: dict,
    metrics_to_plot: list[dict[str, Any]],
    n_apps_values: Iterable[int] | None = None,
    keep_seed_outputs: bool = True,
    scheduler: str | None = None,
) -> None:
    metrics_metadata = {}
    for spec in metrics_to_plot:
        entry = {
            "base_label": spec.get("base_label"),
            "plot": (
                os.path.basename(spec.get("plot_path", ""))
                if spec.get("plot_path")
                else None
            ),
            "plot_type": spec.get("plot_type"),
        }
        metrics_metadata[spec["key"]] = entry
    metadata = {
        "timestamp": timestamp,
        "output_dir": run_dir,
        "ppacket_values": [float(v) for v in ppacket_values],
        "simulations_per_point": simulations_per_point,
        "seed_start": seed_start,
        "config": config,
        "save_path": save_path,
        "raw_csv": os.path.basename(raw_csv_path),
        "metrics": metrics_metadata,
        "parameters": {k: default_kwargs[k] for k in default_kwargs},
        "keep_seed_outputs": bool(keep_seed_outputs),
    }
    if n_apps_values is not None:
        metadata["n_apps_values"] = [int(val) for val in n_apps_values]
    if scheduler is not None:
        metadata["scheduler"] = str(scheduler)
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


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
