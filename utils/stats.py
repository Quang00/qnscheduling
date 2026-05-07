from itertools import combinations

import networkx as nx


def topology_stats(gml_path: str) -> dict[str, float]:
    """Compute statistics for a network topology represented by a GML file.
    Args:
        gml_path (str): GML file path representing the network topology.

    Returns:
        dict[str, float]: avg_disjoint_paths, avg_hops_all, avg_hops_shortest.
    """
    G = nx.read_gml(gml_path)
    disjoint_path_counts = []
    hops_count_all_paths = []
    hops_count_shortest_paths = []

    for u, v in combinations(G.nodes(), 2):
        paths = list(nx.edge_disjoint_paths(G, u, v))
        disjoint_path_counts.append(len(paths))
        hops_count_all_paths.extend(len(path) - 1 for path in paths)
        hops_count_shortest_paths.append(nx.shortest_path_length(G, u, v))

    avg_disjoint_paths = (
        sum(disjoint_path_counts) / len(disjoint_path_counts)
        if disjoint_path_counts
        else 0
    )
    avg_hops_all = (
        sum(hops_count_all_paths) / len(hops_count_all_paths)
        if hops_count_all_paths
        else 0
    )
    avg_hops_shortest = (
        sum(hops_count_shortest_paths) / len(hops_count_shortest_paths)
        if hops_count_shortest_paths
        else 0
    )

    stats = {
        "avg_disjoint_paths": avg_disjoint_paths,
        "avg_hops_all": avg_hops_all,
        "avg_hops_shortest": avg_hops_shortest,
    }
    return stats
