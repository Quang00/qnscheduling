import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from openpyxl.utils import get_column_letter

from scheduling.pga import duration_pga


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


def compute_durations(
    paths: dict[str, list[str]], epr_pairs: dict[str, int]
) -> dict[str, float]:
    """Compute the duration of each application based on the paths and
    link parameters.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        epr_pairs (dict[str, int]): Entanglement generation pairs for each
        application, indicating how many EPR pairs are to be generated.

    Returns:
        dict[str, float]: A dictionary mapping each application to its total
        duration, which includes the time taken for probabilistic generation
        of EPR pairs and the latency based on the distance of the path.
    """
    durations = {}
    for app, route in paths.items():
        pga_time = duration_pga(
            p_packet=0.2,
            epr_pairs=epr_pairs[app],
            n_swap=math.ceil(len(route) / 2),
            memory_lifetime=50,
            p_swap=0.95,
            p_gen=0.001,
            time_slot_duration=1e-4,
        )
        durations[app] = pga_time
    return durations


def app_params_sim(
    paths: dict[str, list[str]], epr_pairs: dict[str, int]
) -> dict[str, dict[str, float]]:
    """Prepare application parameters for simulation.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        epr_pairs (dict[str, int]): Entanglement generation pairs for each
        application, indicating how many EPR pairs are to be generated.

    Returns:
        dict[str, dict[str, float]]: A dictionary mapping each application to
        its parameters for simulation, including the probability of generating
        an EPR pair, the number of EPR pairs to generate, and the time slot
        duration.
    """
    sim_params = {}
    for key in paths.keys():
        sim_params[key] = {
            "p_gen": 0.001,
            "epr_pairs": epr_pairs[key],
            "slot_duration": 1e-4,
        }
    return sim_params


def yaml_config(
    file_path: str,
) -> Tuple[
    List[Tuple[str, str]],
    Dict[frozenset, dict],
    Dict[str, Tuple[str, ...]],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, str],
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
            - Dictionary mapping application names to their priorities.
            - Dictionary mapping application names to their scheduling
            policies.
    """
    with open(file_path) as f:
        network = yaml.safe_load(f)

    # --- Links ---
    links = network.get("links", [])
    edges = [tuple(link["nodes"]) for link in links]

    # --- Applications ---
    apps = network.get("apps", {})
    peers = {app: tuple(cfg["peers"]) for app, cfg in apps.items()}
    instances = {app: cfg.get("N", 1) for app, cfg in apps.items()}
    e_pairs = {app: cfg.get("E_pairs", 1) for app, cfg in apps.items()}
    priorities = {app: cfg.get("priority", 0) for app, cfg in apps.items()}
    policies = {
        app: cfg.get("policy", "best_effort")
        for app, cfg in apps.items()
    }

    return edges, peers, instances, e_pairs, priorities, policies


def save_results(
    df: pd.DataFrame,
    job_names: List[str],
    release_times: Dict[str, float],
    apps: Dict[str, Tuple[str, str]],
    instances: Dict[str, int],
    epr_pairs: Dict[str, int],
    priorities: Dict[str, int],
    policies: Dict[str, str],
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
            - status: Status of the job (e.g., "completed", "failed")
        job_names (List): List of all job names that should be present in the
        results.
        release_times (Dict): Dictionary mapping application names to their
        relative release times, used to fill in missing jobs.
        apps (Dict): Dictionary mapping application names to their source and
        destination nodes.
        instances (Dict): Dictionary mapping application names to the number of
        instances.
        epr_pairs (Dict): Dictionary mapping application names to the number of
        EPR pairs.
        priorities (Dict): Dictionary mapping application names to their
        priorities.
        policies (Dict): Dictionary mapping application names to their
        scheduling policies.
        output_dir (str): Directory where the results CSV file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    missing = set(job_names) - set(df["job"])
    if missing:
        filler_rows = []
        for job in missing:
            app = re.sub(r"\d+$", "", job)
            filler_rows.append(
                {
                    "job": job,
                    "arrival_time": release_times.get(app, np.nan),
                    "start_time": np.nan,
                    "burst_time": np.nan,
                    "completion_time": np.nan,
                    "turnaround_time": np.nan,
                    "waiting_time": np.nan,
                    "status": "missing",
                }
            )
        df = pd.concat([df, pd.DataFrame(filler_rows)], ignore_index=True)

    df["app"] = df["job"].astype(str).str.replace(r"\d+$", "", regex=True)
    params = pd.DataFrame(
        {
            "app": list(apps.keys()),
            "src_node": [apps[a][0] for a in apps],
            "dst_node": [apps[a][1] for a in apps],
            "instances": [instances[a] for a in apps],
            "epr_pairs": [epr_pairs[a] for a in apps],
            "priority": [priorities[a] for a in apps],
            "policy": [policies[a] for a in apps],
        }
    )

    df = df.merge(params, on="app", how="left")
    df = df.drop(columns="app")
    df = df.sort_values(by="completion_time").reset_index(drop=True)

    csv_path = os.path.join(output_dir, "job_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Job Results ===")
    print(df.to_string(index=False))

    makespan = df["completion_time"].max() - df["arrival_time"].min()
    total = len(job_names)
    completed = (df["status"] == "completed").sum()
    failed = (df["status"] == "failed").sum()
    throughput = completed / makespan if makespan > 0 else float("inf")
    waits = df.loc[df["status"] == "completed", "waiting_time"]
    avg_wait = waits.mean() if not waits.empty else float("nan")
    max_wait = waits.max() if not waits.empty else float("nan")

    print("\n=== Summary ===")
    print(f"Total jobs       : {total}")
    print(f"  - completed    : {completed}")
    print(f"  - failed       : {failed}")
    print(f"Makespan         : {makespan:.4f}")
    print(f"Throughput       : {throughput:.4f} jobs/s")
    print(f"Avg waiting time : {avg_wait:.4f}")
    print(f"Max waiting time : {max_wait:.4f}")


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
    max_instances: int,
    max_epr_pairs: int,
    list_policies: list[str],
    max_priority: int = 1,
    seed: int = 42,
) -> Tuple[
    Dict[str, Tuple[str, str]],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, str],
]:
    """Generates a specified number of applications with random parameters.

    Args:
        nodes (list): List of available nodes in the network.
        n_apps (int): Number of applications to generate.
        max_instances (int): Maximum number of instances for each application.
        max_epr_pairs (int): Maximum number of EPR pairs for each application.
        list_policies (list[str], optional): List of policies to assign to
        each application.
        max_priority (int, optional): Maximum priority level for each app.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        A tuple containing the generated applications and their parameters.
    """
    np.random.seed(seed)
    apps = {}
    instances = {}
    epr_pairs = {}
    priorities = {}
    policies = {}

    for i in range(n_apps):
        name_app = get_column_letter(i + 1)
        rand_app = tuple(np.random.choice(nodes, 2, replace=False).tolist())
        rand_instance = np.random.randint(1, max_instances + 1)
        rand_epr_pairs = np.random.randint(1, max_epr_pairs + 1)
        rand_policy = np.random.choice(list_policies, 1, replace=False).item()

        apps[name_app] = rand_app
        instances[name_app] = rand_instance
        epr_pairs[name_app] = rand_epr_pairs
        priorities[name_app] = max_priority
        policies[name_app] = rand_policy

    return apps, instances, epr_pairs, priorities, policies


def total_distances(
    distances: dict[tuple, float], paths: dict[str, list[str]]
) -> dict[str, float]:
    """Compute the total distance for each path in the given paths.

    Args:
        distances (dict[tuple, float]): A dictionary mapping edges (as tuples)
        to their distances.
        paths (dict[str, list[str]]): A dictionary mapping application names
        to their network paths.

    Returns:
        dict[str, float]: A dictionary mapping application names to their total
        distances.
    """
    total_distances = {}
    for name, path in paths.items():
        total = 0.0
        for start, end in zip(path, path[1:]):
            if (start, end) in distances:
                total += distances[(start, end)]
            else:
                total += distances[(end, start)]
        total_distances[name] = total
    return total_distances


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
