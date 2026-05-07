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


def predictor(gml_path: str) -> dict[str, float | str]:
    """Predict whether a network topology is more suitable for static or
    dynamic routing based on its statistics.

    Args:
        gml_path (str): GML file path representing the network topology.

    Returns:
        dict[str, float | str]: path_redundancy, path_stretch, score,
        prediction.
    """
    stats = topology_stats(gml_path)
    path_redundancy = stats["avg_disjoint_paths"] - 1
    path_stretch = stats["avg_hops_all"] / stats["avg_hops_shortest"] - 1
    if path_stretch == 0:
        score = 0 if path_redundancy == 0 else float("inf")
    else:
        score = path_redundancy / path_stretch

    if score == 0:
        prediction = "equal"
    elif path_redundancy < 0.3:
        prediction = "static"
    elif score >= 3.3:
        prediction = "dynamic"
    else:
        prediction = "static"

    topology_prediction = {
        "path_redundancy": path_redundancy,
        "path_stretch": path_stretch,
        "score": score,
        "prediction": prediction,
    }
    return topology_prediction
