from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
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
        paths_for_each_apps: dict[str, List[str]]
) -> List[set]:
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
        List[set]: A list of sets, where each set contains application names
        that can run in parallel without conflicting resource usage.
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

    # Find maximal cliques in the complement conflict graph
    g_complement = nx.complement(G)
    parallelizable_applications = list(nx.find_cliques(g_complement))

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
