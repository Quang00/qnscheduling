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


def shortest_paths(
    edges: List[Tuple[str, str]], app_requests: Dict[str, Tuple[str, str]]
) -> Dict[str, List[str]]:
    """Find shortest paths for each applications in a quantum network graph
    represented by edges.

    Args:
        edges (List[Tuple[str, str]]): List of edges in the quantum network,
        where each edge is a tuple of nodes (src, dst).
        app_requests (Dict[str, Tuple[str, str]]): A dictionary where keys are
        application names and values are tuples of source and destination nodes
        (src, dst) for the application. For example:
            {
                'A': ('Alice', 'Bob'),
                'B': ('Alice', 'Charlie'),
                'C': ('Charlie', 'David'),
                'D': ('Bob', 'David')
            }

    Returns:
        Dict[str, List[str]]: A dictionary where keys are application names and
        values are lists of nodes representing the shortest path from source to
        destination for that application.
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    return {
        application: nx.shortest_path(G, src, dst)
        for application, (src, dst) in app_requests.items()
    }


def parallelizable_tasks(
    paths_for_each_apps: dict[str, List[str]],
) -> dict[str, set[str]]:
    """Find parallelizable applications based on shared resources (nodes) of a
    quantum network.

    Args:
        paths_for_each_apps (dict[str, List[str]]): A dictionary where keys are
        application names and values are lists of resources (network nodes)
        used to run the applications, e.g.:
            {
                'A': ['Alice', 'Bob', 'David'],
                'B': ['Alice', 'Bob'],
                'C': ['Charlie', 'Alice', 'Bob'],
                'D': ['Charlie', 'David']
            }

    Returns:
        dict[str, set[str]]: A dictionary where keys are application names and
        values are sets of applications that can run in parallel with the key
        application, based on shared resources. For example:
            {
                'A': set(),
                'B': {'D'},
                'C': set(),
                'D': {'B'}
            }
    """
    G = nx.Graph()
    conflicts = defaultdict(set)

    # Build conflict graph
    for app, resources in paths_for_each_apps.items():
        G.add_node(app)
        for r in resources:
            for other_app in conflicts[r]:
                G.add_edge(app, other_app)
            conflicts[r].add(app)

    g_complement = nx.complement(G)
    parallelizable_applications = {
        app: set(g_complement.neighbors(app)) for app in g_complement.nodes()
    }

    return parallelizable_applications


def app_params_sim(
    paths: dict[str, list[str]],
    epr_pairs: dict[str, int],
    p_packet: float,
    memory_lifetime: int,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
) -> dict[str, dict[str, int], float, int, float, float, float]:
    """Prepare application parameters for simulation.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        epr_pairs (dict[str, int]): Entanglement generation pairs for each
        application, indicating how many EPR pairs are to be generated.
        p_packet (float): Probability of a packet being generated.
        memory_lifetime (int): Memory lifetime in number of time
        slot units.
        p_swap (float): Probability of swapping an EPR pair in a
        single trial.
        p_gen (float): Probability of generating an EPR pair in a
        single trial.
        time_slot_duration (float): Duration of a time slot in
        seconds.

    Returns:
        dict[str, dict[str, float]]: A dictionary mapping each application to
        its parameters for simulation, including the probability of generating
        an EPR pair, the number of EPR pairs to generate, and the time slot
        duration.
    """
    sim_params = {}
    for key in paths.keys():
        sim_params[key] = {
            "p_packet": p_packet,
            "memory_lifetime": memory_lifetime,
            "p_swap": p_swap,
            "p_gen": p_gen,
            "epr_pairs": epr_pairs[key],
            "slot_duration": time_slot_duration,
        }
    return sim_params


def save_results(
    df: pd.DataFrame,
    pga_names: List[str],
    pga_release_times: Dict[str, float],
    apps: Dict[str, Tuple[str, str]],
    instances: Dict[str, int],
    epr_pairs: Dict[str, int],
    policies: Dict[str, str],
    durations: Dict[str, float] | None = None,
    link_utilization: Dict[Tuple[str, str], Dict[str, float]] = None,
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
        apps (Dict): Dictionary mapping application names to their source and
        destination nodes.
        instances (Dict): Dictionary mapping application names to the number of
        instances.
        epr_pairs (Dict): Dictionary mapping application names to the number of
        EPR pairs.
        policies (Dict): Dictionary mapping application names to their
        scheduling policies.
        durations (Dict | None): Optional mapping of deterministic PGA
        durations per application.
        link_utilization (Dict): Dictionary mapping links to busy time and
        utilization metrics.
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
    params = pd.DataFrame(
        {
            "task": list(apps.keys()),
            "src_node": [apps[a][0] for a in apps],
            "dst_node": [apps[a][1] for a in apps],
            "instances": [instances[a] for a in apps],
            "pairs_requested": [epr_pairs[a] for a in apps],
            "policy": [policies[a] for a in apps],
            "pga_duration": [
                float(durations[a]) if durations and a in durations else np.nan
                for a in apps
            ],
        }
    )

    df = df.merge(params, on="task", how="left").drop(columns="task")
    df = df.sort_values(by="completion_time").reset_index(drop=True)

    csv_path = os.path.join(output_dir, "pga_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Preview PGA Results ===")
    print(df.head(20).to_string(index=False))

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
        link_util_path = os.path.join(output_dir, "link_utilization.csv")
        link_util_df.to_csv(link_util_path, index=False)

        print("\n=== Link Utilization ===")
        print(link_util_df.to_string(index=False))

    makespan = df["completion_time"].max() - df["arrival_time"].min()
    total = len(df)
    completed = (df["status"] == "completed").sum()
    throughput = completed / makespan if makespan > 0 else float("inf")
    waits = df.loc[
        (df["status"] == "completed") | (df["status"] == "failed"),
        "waiting_time"
    ]
    turnarounds = df.loc[
        (df["status"] == "completed") | (df["status"] == "failed"),
        "turnaround_time"
    ]
    pga_durations = (
        np.array(list(durations.values()), dtype=float)
        if durations
        else np.array([], dtype=float)
    )
    avg_wait = waits.mean() if not waits.empty else float("nan")
    max_wait = waits.max() if not waits.empty else float("nan")
    avg_turnaround = (
        turnarounds.mean() if not turnarounds.empty else float("nan")
    )
    max_turnaround = (
        turnarounds.max() if not turnarounds.empty else float("nan")
    )
    total_pga_duration = (
        float(np.sum(pga_durations)) if pga_durations.size else float("nan")
    )
    completed_ratio = (completed / total) if total > 0 else float("nan")

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
    for task in tasks_sorted:
        row = per_task.loc[task]
        n_completed = int(row.get("completed", 0))
        n_failed = int(row.get("failed", 0))
        print(f"    {task:<4} completed: {n_completed}, failed: {n_failed}")

    print(f"Makespan         : {makespan:.4f}")
    print(f"Throughput       : {throughput:.4f} completed PGAs/s")
    print(f"Completed ratio  : {completed_ratio:.4f}")
    print(f"Avg waiting time : {avg_wait:.4f}")
    print(f"Max waiting time : {max_wait:.4f}")
    print(f"Avg turnaround   : {avg_turnaround:.4f}")
    print(f"Max turnaround   : {max_turnaround:.4f}")
    if pga_durations.size:
        print(f"Total PGA duration : {total_pga_duration:.4f}")

    overall_df = pd.DataFrame(
        [
            {
                "makespan": float(makespan),
                "throughput": float(throughput),
                "completed_ratio": float(completed_ratio),
                "avg_waiting_time": float(avg_wait),
                "max_waiting_time": float(max_wait),
                "avg_turnaround_time": float(avg_turnaround),
                "max_turnaround_time": float(max_turnaround),
                "total_pga_duration": float(total_pga_duration),
            }
        ]
    )
    overall_path = os.path.join(output_dir, "summary.csv")
    overall_df.to_csv(overall_path, index=False)

    per_task_df = (
        per_task.loc[tasks_sorted, ["completed", "failed"]]
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
    per_task_path = os.path.join(output_dir, "summary_per_task.csv")
    per_task_df.to_csv(per_task_path, index=False)


def gml_data(gml_file: str) -> Tuple[list, list, dict[tuple, float]]:
    """Extracts nodes, edges, and distances from a GML file.

    Args:
        gml_file (str): Path to the GML file.

    Returns:
        nodes (list): List of nodes.
        edges (list): List of edges (source, target).
        distances (dict[tuple, float]): Dict mapping edges
        to distances.
    """
    G = nx.read_gml(gml_file)

    nodes = list(G.nodes())
    edges = list(G.edges())
    distances = {(u, v): data.get("dist") for u, v, data in G.edges(data=True)}

    return nodes, edges, distances


def generate_n_apps(
    nodes: list,
    n_apps: int,
    inst_range: tuple[int, int],
    epr_range: tuple[int, int],
    period_range: tuple[float, float],
    list_policies: list[str],
    rng: np.random.Generator,
) -> Tuple[
    Dict[str, Tuple[str, str]],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
]:
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
        list_policies (list[str], optional): List of policies to assign to
        each application.
        rng (np.random.Generator): Random number generator for reproducibility.

    Returns:
        A tuple containing the generated applications and their parameters.
    """
    apps = {}
    instances = {}
    epr_pairs = {}
    policies = {}
    periods = {}

    for i in range(n_apps):
        name_app = get_column_letter(i + 1)
        rand_app = tuple(rng.choice(nodes, 2, replace=False).tolist())
        rand_instance = rng.integers(inst_range[0], inst_range[1] + 1)
        rand_epr_pairs = rng.integers(epr_range[0], epr_range[1] + 1)
        rand_period = rng.uniform(period_range[0], period_range[1])
        rand_policy = rng.choice(list_policies, 1, replace=False).item()

        apps[name_app] = rand_app
        instances[name_app] = rand_instance
        epr_pairs[name_app] = rand_epr_pairs
        periods[name_app] = rand_period
        policies[name_app] = rand_policy

    return apps, instances, epr_pairs, periods, policies


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
    output_dir: str, ppacket_values: Iterable[float]
) -> tuple[str, str]:
    base_output = output_dir or "results"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_output, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    for p_val in ppacket_values:
        subdir = os.path.join(run_dir, ppacket_dirname(p_val))
        os.makedirs(subdir, exist_ok=True)
    return run_dir, timestamp


def build_default_sim_args(config: str, args: dict | None) -> dict:
    default_args = {
        "config": config,
        "n_apps": 100,
        "inst_range": (100, 100),
        "epr_range": (2, 2),
        "period_range": (70, 70),
        "hyperperiod_cycles": 100,
        "memory_lifetime": 50,
        "p_swap": 0.95,
        "p_gen": 1e-3,
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
    n_apps: int,
) -> list[tuple[Any, ...]]:
    tasks = []
    seed_pool = [
        seed_start + offset for offset in range(simulations_per_point)
    ]
    for p_packet in ppacket_values:
        for run_seed in seed_pool:
            tasks.append((p_packet, run_seed, run_dir, default_kwargs, n_apps))
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
            mean=(column, "mean"),
            std=(column, "std"),
            count=(column, "count"),
        )
        .assign(
            sem=lambda df: df["std"] / np.sqrt(df["count"].clip(lower=1)),
        )
    )

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
) -> None:
    metrics_metadata = {
        spec["key"]: {
            "base_label": spec.get("base_label"),
            "plot": (
                os.path.basename(spec.get("plot_path", ""))
                if spec.get("plot_path")
                else None
            ),
            "summary_csv": os.path.basename(spec["csv_path"]),
        }
        for spec in metrics_to_plot
    }
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
    }
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
