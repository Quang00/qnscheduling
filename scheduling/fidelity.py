from collections import defaultdict
from typing import Dict, List, Tuple


def werner_adj_list(
    fidelities: Dict[Tuple[str, str], float],
) -> Dict[str, List[Tuple[str, float]]]:
    """Convert directed edge fidelities to an undirected adjacency list with
    Werner parameters as weights.

    Args:
        fidelities (Dict[Tuple[str, str], float]): Mapping of directed edges
        (u, v) to their fidelities.

    Returns:
        Dict[str, List[Tuple[str, float]]]: Adjacency list where each key is a
        node and the value is a list of tuples (neighbor, weight) representing
        the undirected edges and their corresponding Werner parameters.
    """
    werner_adjacency = defaultdict(list)
    for (u, v), f in fidelities.items():
        w = (4.0 * f - 1.0) / 3.0
        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0
        werner_adjacency[u].append((v, w))
        werner_adjacency[v].append((u, w))
    return werner_adjacency


def fidelity_bounds_and_paths(
    end_nodes: list[str], fidelities: dict[tuple[str, str], float], K: int = 8
) -> Tuple[Dict, Dict]:
    """Compute E2E fidelity bounds and simple paths between end nodes based on
    the given edge fidelities. The function performs a DFS from each end node
    to find all simple paths up to K hops and calculates the corresponding E2E
    fidelities.

    Args:
        end_nodes (list[str]): List of end nodes for which to compute fidelity
        bounds and paths.
        fidelities (dict[tuple[str, str], float]): Mapping of directed edges
        (u, v) to their fidelities.
        K (int, optional): Maximum number of hops.

    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - bounds: A dictionary mapping pairs of end nodes to their minimum
            and maximum fidelity bounds.
            - paths: A dictionary mapping pairs of end nodes to a list of
            tuples where each tuple contains the fidelity and the corresponding
            simple path between the nodes.
    """
    werner_adjacency = werner_adj_list(fidelities)
    end_nodes = sorted(end_nodes)
    endpoints = set(end_nodes)
    bounds = {}
    paths = defaultdict(list)

    for i, source in enumerate(end_nodes):
        best_max, best_min = {}, {}
        stack = [(source, 0, 1.0, (source,))]

        while stack:
            cur_node, hop_cpt, prod, path = stack.pop()
            if hop_cpt == K:
                continue

            for v, w in werner_adjacency.get(cur_node, ()):
                if v in path:
                    continue
                acc = prod * w
                next_path = path + (v,)

                if v in endpoints and v != source:
                    if source < v:
                        F = (3.0 * acc + 1.0) / 4.0
                        paths[(source, v)].append((F, next_path))

                    if (v not in best_max) or (acc > best_max[v]):
                        best_max[v] = acc
                    if (v not in best_min) or (acc < best_min[v]):
                        best_min[v] = acc

                stack.append((v, hop_cpt + 1, acc, next_path))

        for destination in end_nodes[i + 1:]:
            if destination in best_max:
                bounds[(source, destination)] = (
                    (3.0 * best_min[destination] + 1.0) / 4.0,
                    (3.0 * best_max[destination] + 1.0) / 4.0,
                )
    for _, path in paths.items():
        path.sort(key=lambda x: len(x[1]) - 1)
    return bounds, dict(paths)
