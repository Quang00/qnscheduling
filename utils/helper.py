import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import yaml

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
    paths: dict[str, list[str]], link_params: dict, epr_pairs: dict[str, int]
) -> dict[str, float]:
    """Compute the duration of each application based on the paths and
    link parameters.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        link_params (dict): Parameters for each link in the network, including
        packet generation probability, memory lifetime, swap probability,
        and time slot duration.
        epr_pairs (dict[str, int]): Entanglement generation pairs for each
        application, indicating how many EPR pairs are to be generated.

    Returns:
        dict[str, float]: A dictionary mapping each application to its total
        duration, which includes the time taken for probabilistic generation
        of EPR pairs and the latency based on the distance of the path.
    """
    durations = {}
    for app, route in paths.items():
        first_link = frozenset((route[0], route[1]))
        link_parameters = link_params[first_link]

        pga_time = duration_pga(
            p_packet=link_parameters["p_packet"],
            epr_pairs=epr_pairs[app],
            n_swap=math.ceil(len(route) / 2),
            memory_lifetime=link_parameters["memory_lifetime"],
            p_swap=link_parameters["p_swap"],
            p_gen=link_parameters["p_gen"],
            time_slot_duration=link_parameters["time_slot_duration"],
        )

        distance = 0.0
        for i in range(len(route) - 1):
            link = frozenset((route[i], route[i + 1]))
            if link in link_params:
                distance += link_params[link].get("distance", 0.0)
            else:
                distance += 0.0
        latency = distance / 200_000

        durations[app] = pga_time + latency
    return durations


def app_params_sim(
    paths: dict[str, list[str]], link_params: dict, epr_pairs: dict[str, int]
) -> dict[str, dict[str, float]]:
    """Prepare application parameters for simulation.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        link_params (dict): Parameters for each link in the network, including
        packet generation probability, memory lifetime, swap probability,
        and time slot duration.
        epr_pairs (dict[str, int]): Entanglement generation pairs for each
        application, indicating how many EPR pairs are to be generated.

    Returns:
        dict[str, dict[str, float]]: A dictionary mapping each application to
        its parameters for simulation, including the probability of generating
        an EPR pair, the number of EPR pairs to generate, and the time slot
        duration.
    """
    sim_params = {}
    for app, route in paths.items():
        lk = link_params[frozenset((route[0], route[1]))]
        sim_params[app] = {
            "p_gen": lk["p_gen"],
            "epr_pairs": epr_pairs[app],
            "slot_duration": lk["time_slot_duration"],
        }
    return sim_params


def parse_yaml_config(
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
    link_params = {frozenset(link["nodes"]): link for link in links}

    # --- Applications ---
    apps = network.get("apps", {})
    peers = {app: tuple(cfg["peers"]) for app, cfg in apps.items()}
    instances = {app: cfg.get("N", 1) for app, cfg in apps.items()}
    e_pairs = {app: cfg.get("E_pairs", 1) for app, cfg in apps.items()}
    priorities = {app: cfg.get("priority", 0) for app, cfg in apps.items()}
    policies = {app: cfg.get("policy", "best_effort")
                for app, cfg in apps.items()}

    return edges, link_params, peers, instances, e_pairs, priorities, policies


def save_results(
    df: pd.DataFrame,
    job_names: List,
    release_times: Dict,
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
        output_dir (str): Directory where the results CSV file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = df.sort_values(by="completion_time").reset_index(drop=True)
    csv_path = os.path.join(output_dir, "job_results.csv")
    df.to_csv(csv_path, index=False)

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
                "status",
            ],
        )
    )

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


def extract_gml_data(gml_file: str):
    """Extracts nodes, edges, and distances from a GML file.

    Args:
        gml_file: Path to the GML file.

    Returns:
        nodes: List of nodes.
        edges: List of edges (source, target).
        distances: Dict mapping (source, target): distance.
    """
    G = nx.read_gml(gml_file)

    nodes = list(G.nodes())
    edges = list(G.edges())
    distances = {(u, v): data.get("dist") for u, v, data in G.edges(data=True)}

    return nodes, edges, distances
