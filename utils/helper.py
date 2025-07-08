import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import yaml


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


def parse_yaml_config(
    file_path: str,
) -> Tuple[
    List[Tuple[str, str]],
    Dict[frozenset, dict],
    Dict[str, Tuple[str, ...]],
    Dict[str, int],
    Dict[str, int],
]:
    """Parse a YAML configuration file to extract quantum network
    configuration, including links and applications.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        Tuple containing:
            - List of edges as tuples (src, dst).
            - Dictionary mapping frozenset of nodes to link parameters.
            - Dictionary mapping application names to tuples of peer nodes.
            - Dictionary mapping application names to the number of instances.
            - Dictionary mapping application names to the number of EPR pairs.
    """
    with open(file_path) as f:
        network = yaml.safe_load(f)

    links = network.get("links", [])
    edges = [tuple(link["nodes"]) for link in links]
    link_params = {frozenset(link["nodes"]): link for link in links}
    apps = network.get("apps", {})
    peers = {app: tuple(cfg["peers"]) for app, cfg in apps.items()}
    instances = {app: cfg.get("N", 1) for app, cfg in apps.items()}
    e_pairs = {app: cfg.get("E_pairs", 1) for app, cfg in apps.items()}

    return edges, link_params, peers, instances, e_pairs


def save_results(
    df: pd.DataFrame,
    job_names: List,
    release_times: Dict,
    total_jobs: int,
    output_dir: str = "results",
) -> None:
    """Save the results of job scheduling and execution to a CSV file and print
    a summary of the results.

    Args:
        df (DataFrame): DataFrame containing job results with columns:
            - job: Job identifier
            - arrival_time: Time when the job arrived
            - start_time: Time when the job started execution
            - burst_time: Total time required for the job to complete
            - completion_time: Time when the job completed execution
            - turnaround_time: Total time from arrival to completion
            - waiting_time: Total time the job waited before execution
        job_names (List): List of all job names that should be present in the
        results.
        release_times (Dict): Dictionary mapping application names to their
        relative release times, used to fill in missing jobs.
        total_jobs (int): Total number of jobs that were expected to be
        processed.
        output_dir (str): Directory where the results CSV file will be saved.
    """
    finished_jobs = set(df["job"])
    uncompleted_jobs = [job for job in job_names if job not in finished_jobs]

    if uncompleted_jobs:
        supplements = []
        for uncompleted_job in uncompleted_jobs:
            app = uncompleted_job.split("_")[0]
            supplements.append(
                {
                    "job": uncompleted_job,
                    "arrival_time": release_times.get(app, float("nan")),
                    "start_time": math.nan,
                    "burst_time": math.nan,
                    "completion_time": math.nan,
                    "turnaround_time": math.nan,
                    "waiting_time": math.nan,
                }
            )
        df = pd.concat([df, pd.DataFrame(supplements)], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    df = df.sort_values(by=["completion_time"]).reset_index(drop=True)
    df.to_csv(f"{output_dir}/job_results.csv", index=False)

    print("\n=== Job Results ===")
    print(
        df.to_string(
            index=False,
            columns=[
                "job",
                "arrival_time",
                "start_time",
                "burst_time",
                "completion_time",
                "turnaround_time",
                "waiting_time",
            ],
        )
    )

    makespan = df["completion_time"].max() - df["arrival_time"].min()
    completed_count = df["completion_time"].count()
    throughput = completed_count / makespan if makespan > 0 else float("inf")
    waits = df["waiting_time"].dropna()

    print("\n=== Summary ===")
    print(f"makespan         : {makespan:.4f}")
    print(f"throughput       : {throughput:.4f} jobs/s")
    print(f"avg_waiting_time : {waits.mean():.4f}")
    print(f"max_waiting_time : {waits.max():.4f}")
    print(f"total_instances  : {total_jobs}")
    print(f"completed_jobs   : {completed_count}")
    print(f"unfinished_jobs  : {len(uncompleted_jobs)}")
