from collections import defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from scheduling.pga import duration_pga
from scheduling.fidelity import is_e2e_fidelity_feasible

from utils.helper import all_simple_paths


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
        application: nx.shortest_path(G, req["src"], req["dst"])
        for application, req in app_requests.items()
    }


def _build_graph(
    routing_mode: str,
    edges: List[Tuple[str, str]],
    base_graph: nx.Graph,
    cap: Dict[Tuple[str, str], float],
    capacity_threshold: float,
) -> nx.Graph:
    if routing_mode == "capacity":
        usable_edges = [
            (u, v)
            for (u, v) in edges
            if cap[tuple(sorted((u, v)))] < capacity_threshold
        ]
        G = nx.Graph()
        G.add_edges_from(usable_edges)
        return G
    return base_graph


def yen_random(
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    src: str,
    dst: str,
    rng: np.random.Generator,
    min_fidelity: float,
) -> List[str] | None:
    seen = 0
    selected_path = None
    all_paths = all_simple_paths(simple_paths, src, dst)
    for path in all_paths:
        if path[0] < min_fidelity:
            continue
        seen += 1
        if rng.integers(seen) == 0:
            selected_path = path[1]
    return selected_path


def smallest_bottleneck(
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    src: str,
    dst: str,
    req: Dict[str, Any],
    cap: Dict[Tuple[str, str], float],
    p_packet: float | None,
    memory: int,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
    rng: np.random.Generator | None = None,
) -> Tuple[List[str] | None, float]:
    """Select the path with the smallest bottleneck capacity among all paths
    that meet the fidelity requirement. The bottleneck capacity of a path is
    defined as the maximum capacity utilization of any edge along the path
    after accounting for the additional load (delta) introduced by the new
    request.
    """
    selected_path = None
    selected_delta = 0.0
    smallest_bottleneck = None
    tied_count = 0
    all_paths = all_simple_paths(simple_paths, src, dst)

    for path in all_paths:
        e2e_fid, path = path[0], path[1]
        if e2e_fid < req["min_fidelity"]:
            continue

        n_swaps = max(0, len(path) - 2)
        pga_duration = duration_pga(
            p_packet=p_packet,
            epr_pairs=req["epr"],
            n_swap=n_swaps,
            memory=memory,
            p_swap=p_swap,
            p_gen=p_gen,
            time_slot_duration=time_slot_duration,
        )
        delta = float(pga_duration) / float(req["period"])

        links = [
            tuple(sorted((u, v)))
            for u, v in zip(path[:-1], path[1:], strict=False)
        ]
        max_cap = max(cap[lk] + delta for lk in links)

        if smallest_bottleneck is None or max_cap < smallest_bottleneck:
            smallest_bottleneck = max_cap
            selected_path = path
            selected_delta = delta
            tied_count = 1
        elif max_cap == smallest_bottleneck:
            tied_count += 1
            if rng.integers(tied_count) == 0:
                selected_path = path
                selected_delta = delta
    return selected_path, selected_delta


def least_capacity(
    G: nx.Graph,
    src: str,
    dst: str,
    req: Dict[str, Any],
    cap: Dict[Tuple[str, str], float],
    fidelities: Dict[Tuple[str, str], float],
    p_packet: float | None,
    memory: int,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
    rng: np.random.Generator | None = None,
) -> Tuple[List[str] | None, float]:
    """Select the path with the least total capacity utilization among all
    paths that meet the fidelity requirement. The total capacity utilization of
    a path is defined as the sum of the capacity utilizations of all edges
    along the path after accounting for the additional load (delta) introduced
    by the new request.
    """
    selected_path = None
    selected_delta = 0.0
    least_cap = None
    tied_count = 0

    for path in nx.shortest_simple_paths(G, src, dst):
        if not is_e2e_fidelity_feasible(
            path, req["min_fidelity"], fidelities
        ):
            continue

        n_swaps = max(0, len(path) - 2)
        pga_duration = duration_pga(
            p_packet=p_packet,
            epr_pairs=req["epr"],
            n_swap=n_swaps,
            memory=memory,
            p_swap=p_swap,
            p_gen=p_gen,
            time_slot_duration=time_slot_duration,
        )
        delta = float(pga_duration) / float(req["period"])

        links = [
            tuple(sorted((u, v)))
            for u, v in zip(path[:-1], path[1:], strict=False)
        ]
        sum_cap = sum(cap[lk] + delta for lk in links)
        if least_cap is None or sum_cap < least_cap:
            least_cap = sum_cap
            selected_path = path
            selected_delta = delta
            tied_count = 1
        elif sum_cap == least_cap:
            tied_count += 1
            if rng.integers(tied_count) == 0:
                selected_path = path
                selected_delta = delta
    return selected_path, selected_delta


def capacity_threshold(
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    src: str,
    dst: str,
    req: Dict[str, Any],
    cap: Dict[Tuple[str, str], float],
    threshold: float,
    p_packet: float | None,
    memory: int,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
) -> Tuple[List[str] | None, float]:
    selected_path = None
    selected_delta = 0.0
    all_paths = all_simple_paths(simple_paths, src, dst)

    for path in all_paths:
        e2e_fid, path = path[0], path[1]
        if e2e_fid < req["min_fidelity"]:
            continue

        n_swaps = max(0, len(path) - 2)
        pga_duration = duration_pga(
            p_packet=p_packet,
            epr_pairs=req["epr"],
            n_swap=n_swaps,
            memory=memory,
            p_swap=p_swap,
            p_gen=p_gen,
            time_slot_duration=time_slot_duration,
        )
        delta = float(pga_duration) / float(req["period"])
        links = [
            tuple(sorted((u, v)))
            for u, v in zip(path[:-1], path[1:], strict=False)
        ]
        if any(cap[lk] + delta > threshold for lk in links):
            continue
        selected_delta = delta
        selected_path = path
        break

    return selected_path, selected_delta


def fidelity_shortest(
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    src: str,
    dst: str,
    min_fidelity: float,
) -> List[str] | None:
    all_paths = all_simple_paths(simple_paths, src, dst)
    for path in all_paths:
        e2e_fid = path[0]
        if e2e_fid < min_fidelity:
            continue
        else:
            return path[1]
    return None


def _update_capacity(
    path: List[str],
    delta: float,
    cap: Dict[Tuple[str, str], float],
) -> None:
    for u, v in zip(path[:-1], path[1:], strict=False):
        link = tuple(sorted((u, v)))
        cap[link] += delta


def find_feasible_path(
    edges: List[Tuple[str, str]],
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    app_requests: Dict[str, Dict[str, Any]],
    fidelities: Dict[Tuple[str, str], float] | None,
    pga_rel_times: Dict[str, float] | None = None,
    routing_mode: str = "shortest",
    threshold: float = 0.8,
    p_packet: float | None = None,
    memory: int = 1,
    p_swap: float = 0.6,
    p_gen: float = 0.001,
    time_slot_duration: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> Dict[str, List[str] | None]:
    """Assign a feasible path for each application in a quantum network graph
    based on minimum fidelity threshold.

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
        routing_mode (str, optional): Routing mode, either "shortest", "random"
            , "degree", or "capacity". In "capacity" mode, edges that have
            reached the capacity threshold are excluded from path selection. In
            "random" mode, a random path is selected among all shortest paths
            that meet the fidelity requirement. In "degree" mode, the path with
            the lowest maximum degree of internal nodes is selected among all
            shortest paths that meet the fidelity requirement.
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
    if fidelities is None or not fidelities:
        print(
            "Please provide per-edge fidelities "
            "with CLI (e.g., \"-f 0.6 0.85\")"
        )
        return {app: None for app in app_requests.keys()}

    base_graph = nx.Graph()
    base_graph.add_edges_from(edges)
    ret = {}
    cap = defaultdict(float)

    apps_ordered = list(app_requests.keys())
    if pga_rel_times is not None:
        apps_ordered.sort(
            key=lambda app: (float(pga_rel_times.get(app, 0.0)), str(app))
        )

    for app in apps_ordered:
        req = app_requests[app]
        src = req["src"]
        dst = req["dst"]
        min_fidelity = req["min_fidelity"]

        if min_fidelity <= 0.5:
            ret[app] = None
            continue

        G = _build_graph(
            routing_mode, edges, base_graph, cap, threshold
        )

        if not nx.has_path(G, src, dst):
            ret[app] = None
            continue

        if routing_mode == "random":
            selected_path = yen_random(
                simple_paths,
                src,
                dst,
                rng,
                min_fidelity,
            )
        elif routing_mode == "smallest":
            selected_path, selected_delta = smallest_bottleneck(
                simple_paths,
                src,
                dst,
                req,
                cap,
                p_packet,
                memory,
                p_swap,
                p_gen,
                time_slot_duration,
                rng,
            )
            if selected_path is None:
                ret[app] = None
                continue
            _update_capacity(selected_path, selected_delta, cap)
        elif routing_mode == "capacity":
            selected_path, selected_delta = capacity_threshold(
                simple_paths,
                src,
                dst,
                req,
                cap,
                threshold,
                p_packet,
                memory,
                p_swap,
                p_gen,
                time_slot_duration,
            )
            if selected_path is None:
                ret[app] = None
                continue
            _update_capacity(selected_path, selected_delta, cap)
        elif routing_mode == "least":
            selected_path, selected_delta = least_capacity(
                G,
                src,
                dst,
                req,
                cap,
                fidelities,
                p_packet,
                memory,
                p_swap,
                p_gen,
                time_slot_duration,
                rng,
            )
            if selected_path is None:
                ret[app] = None
                continue
            _update_capacity(selected_path, selected_delta, cap)
        else:
            selected_path = fidelity_shortest(
                simple_paths,
                src,
                dst,
                min_fidelity,
            )

        ret[app] = selected_path
    return ret
