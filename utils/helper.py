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
    app_specs: dict[str, dict[str, Any]],
    p_packet: float,
    memory_lifetime: int,
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
        memory_lifetime (int): Memory lifetime in number of time slot units.
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
            "memory_lifetime": memory_lifetime,
            "p_swap": p_swap,
            "p_gen": p_gen,
            "epr_pairs": int(spec["epr"]),
            "slot_duration": time_slot_duration,
        }
    return sim_params


def save_results(
    df: pd.DataFrame,
    pga_names: List[str],
    pga_release_times: Dict[str, float],
    app_specs: Dict[str, Dict[str, Any]],
    n_edges: int,
    durations: Dict[str, float] | None = None,
    link_utilization: Dict[Tuple[str, str], Dict[str, float]] | None = None,
    scheduler: str = 'dynamic',
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
    app_names = list(app_specs.keys())
    params = pd.DataFrame(
        {
            "task": app_names,
            "src_node": [app_specs[a]["src"] for a in app_names],
            "dst_node": [app_specs[a]["dst"] for a in app_names],
            "instances": [int(app_specs[a]["instances"]) for a in app_names],
            "pairs_requested": [int(app_specs[a]["epr"]) for a in app_names],
            "policy": [app_specs[a]["policy"] for a in app_names],
            "pga_duration": [
                float(durations[a]) if durations and a in durations else np.nan
                for a in app_names
            ],
        }
    )

    df = df.merge(params, on="task", how="left").drop(columns="task")
    df = df.sort_values(by="completion_time").reset_index(drop=True)

    csv_path = os.path.join(output_dir, "pga_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Preview PGA Results ===")
    print(df.head(20).to_string(index=False))

    avg_link_utilization = float("nan")
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
        makespan = df["completion_time"].max() - df["arrival_time"].min()
        avg_link_utilization = busy_time_sum / makespan / n_edges

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
    if scheduler == 'dynamic':
        total = df['pga'].nunique()
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
    print(f"Total PGA duration : {total_pga_duration:.4f}")
    print(f"Avg link utilization : {avg_link_utilization:.4f}")

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
                "avg_link_utilization": float(avg_link_utilization),
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
        rand_policy = rng.choice(list_policies, 1, replace=False).item()

        apps[name_app] = {
            "src": src,
            "dst": dst,
            "instances": rand_instance,
            "epr": rand_epr_pairs,
            "period": rand_period,
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
