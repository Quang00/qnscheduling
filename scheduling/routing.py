from collections import defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from scheduling.pga import duration_pga
from utils.helper import all_simple_paths

EPS = 1e-12


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
        application: [nx.shortest_path(G, req["src"], req["dst"])]
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


def _compute_delta_and_links(
    path: List[str],
    req: Dict[str, Any],
    p_packet: float | None,
    memory: int,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
) -> Tuple[float, List[Tuple[str, str]]]:
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
    return delta, links


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
    provisioning: bool = True,
) -> Tuple[List[List[str]], float, float]:
    """Select the path with the smallest bottleneck capacity among all paths
    that meet the fidelity requirement. The bottleneck capacity of a path is
    defined as the maximum capacity utilization of any edge along the path
    after accounting for the additional load (delta) introduced by the new
    request.
    """
    min_val = None
    tied_candidates = []
    all_paths = all_simple_paths(simple_paths, src, dst)

    for path in all_paths:
        e2e_fid, path = path[0], path[1]
        if e2e_fid < req["min_fidelity"]:
            continue
        delta, links = _compute_delta_and_links(
            path, req, p_packet, memory, p_swap, p_gen, time_slot_duration
        )

        max_cap = max(cap[lk] + delta for lk in links)
        if min_val is None or max_cap < min_val:
            min_val = max_cap
            tied_candidates = [(path, delta, e2e_fid)]
        elif np.isclose(max_cap, min_val):
            tied_candidates.append((path, delta, e2e_fid))

    if not tied_candidates:
        return [], 0.0, float("nan")

    initial_idx = (
        0
        if rng is None or len(tied_candidates) == 1
        else int(rng.integers(len(tied_candidates)))
    )
    selected_path, selected_delta, selected_e2e_fid = (
        tied_candidates[initial_idx]
    )

    if provisioning:
        result = [list(selected_path)] + [
            list(p)
            for i, (p, _, _) in enumerate(tied_candidates)
            if i != initial_idx
        ]
    else:
        result = [list(selected_path)]
    return result, selected_delta, selected_e2e_fid


def least_capacity(
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
    provisioning: bool = True,
) -> Tuple[List[List[str]], float, float]:
    """Select the path with the least total capacity utilization among all
    paths that meet the fidelity requirement. The total capacity utilization of
    a path is defined as the sum of the capacity utilizations of all edges
    along the path after accounting for the additional load (delta) introduced
    by the new request.
    """
    min_val = None
    tied_candidates = []
    all_paths = all_simple_paths(simple_paths, src, dst)

    for path in all_paths:
        e2e_fid, path = path[0], path[1]
        if e2e_fid < req["min_fidelity"]:
            continue
        delta, links = _compute_delta_and_links(
            path, req, p_packet, memory, p_swap, p_gen, time_slot_duration
        )

        sum_cap = sum(cap[lk] + delta for lk in links)
        if min_val is None or sum_cap < min_val:
            min_val = sum_cap
            tied_candidates = [(path, delta, e2e_fid)]
        elif np.isclose(sum_cap, min_val):
            tied_candidates.append((path, delta, e2e_fid))

    if not tied_candidates:
        return [], 0.0, float("nan")

    initial_idx = (
        0
        if rng is None or len(tied_candidates) == 1
        else int(rng.integers(len(tied_candidates)))
    )
    selected_path, selected_delta, selected_e2e_fid = (
        tied_candidates[initial_idx]
    )

    if provisioning:
        result = [list(selected_path)] + [
            list(p)
            for i, (p, _, _) in enumerate(tied_candidates)
            if i != initial_idx
        ]
    else:
        result = [list(selected_path)]
    return result, selected_delta, selected_e2e_fid


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
    rng: np.random.Generator | None = None,
    provisioning: bool = True,
) -> Tuple[List[List[str]], float, float]:
    candidate_paths = []
    shortest_length = None
    selected_delta = 0.0
    selected_e2e_fid = float("nan")
    all_paths = all_simple_paths(simple_paths, src, dst)

    for path in all_paths:
        e2e_fid, path = path[0], path[1]
        if e2e_fid < req["min_fidelity"]:
            continue
        delta, links = _compute_delta_and_links(
            path, req, p_packet, memory, p_swap, p_gen, time_slot_duration
        )
        if any(cap[lk] + delta > threshold for lk in links):
            continue
        path_len = len(path)
        if shortest_length is None:
            shortest_length = path_len
            selected_delta = delta
            selected_e2e_fid = e2e_fid
        elif path_len > shortest_length:
            break
        candidate_paths.append(list(path))

    if not candidate_paths:
        return [], 0.0, float("nan")

    initial_idx = (
        0
        if rng is None or len(candidate_paths) == 1
        else int(rng.integers(len(candidate_paths)))
    )
    if provisioning:
        result = [candidate_paths[initial_idx]] + [
            p for i, p in enumerate(candidate_paths) if i != initial_idx
        ]
    else:
        result = [candidate_paths[initial_idx]]
    return result, selected_delta, selected_e2e_fid


def fidelity_shortest(
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    src: str,
    dst: str,
    min_fidelity: float,
    rng: np.random.Generator,
    provisioning: bool = True,
) -> Tuple[List[List[str]], float]:
    candidate_paths = []
    shortest_length = None

    for path in all_simple_paths(simple_paths, src, dst):
        e2e_fid, path_nodes = path[0], path[1]
        if e2e_fid < min_fidelity:
            continue
        path_len = len(path_nodes)
        if shortest_length is None:
            shortest_length = path_len
        elif path_len > shortest_length:
            break
        candidate_paths.append((e2e_fid, list(path_nodes)))

    if not candidate_paths:
        return [], float("nan")

    initial_idx = int(rng.integers(len(candidate_paths)))
    initial_fid, initial_path = candidate_paths[initial_idx]
    if provisioning:
        result = [initial_path] + [
            p for i, (_, p) in enumerate(candidate_paths) if i != initial_idx
        ]
    else:
        result = [initial_path]
    return result, initial_fid


def highest_fidelity(
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    src: str,
    dst: str,
    min_fidelity: float,
    rng: np.random.Generator,
    provisioning: bool = True,
) -> Tuple[List[List[str]], float]:
    """Select the path with the highest E2E fidelity among all paths that
    meet the minimum fidelity requirement. Ties are broken randomly.

    Args:
        simple_paths: Pre-computed (fidelity, path) pairs keyed by node pair.
        src: Source node.
        dst: Destination node.
        min_fidelity: Minimum acceptable E2E fidelity.
        rng: Random number generator for tie-breaking.
        provisioning: If True, return selected path plus all tied candidates;
            otherwise return only the selected path.

    Returns:
        Tuple of (list of paths with selected first, e2e fidelity or nan).
    """
    best_fid = float("nan")
    tied_candidates = []
    all_paths = all_simple_paths(simple_paths, src, dst)
    for path in all_paths:
        e2e_fid = path[0]
        if e2e_fid < min_fidelity:
            continue
        if not tied_candidates or e2e_fid > best_fid:
            best_fid = e2e_fid
            tied_candidates = [list(path[1])]
        elif np.isclose(e2e_fid, best_fid):
            tied_candidates.append(list(path[1]))

    if not tied_candidates:
        return [], float("nan")

    initial_idx = (
        0
        if len(tied_candidates) == 1
        else int(rng.integers(len(tied_candidates)))
    )
    if provisioning:
        result = [tied_candidates[initial_idx]] + [
            p for i, p in enumerate(tied_candidates) if i != initial_idx
        ]
    else:
        result = [tied_candidates[initial_idx]]
    return result, best_fid


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
    provisioning: bool = True,
) -> Dict[str, List[List[str]]]:
    """Find feasible paths for each application request based on the specified
    routing and the fidelity threshold.

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
        provisioning (bool, optional): Whether to enable provisioning for
            routing.

    Returns:
        Tuple[
            Dict[str, List[List[str]]],
            Dict[str, float],
        ]: A tuple of:
            - paths: dict mapping application names to lists of feasible paths
              fidelity of the selected path, or nan if none was found.
            - e2e_fids: dict mapping application names to the end-to-end
              fidelity of the selected path, or nan if none was found.
    """
    if fidelities is None or not fidelities:
        return (
            {app: [] for app in app_requests.keys()},
            {app: float("nan") for app in app_requests.keys()},
        )

    base_graph = nx.Graph()
    base_graph.add_edges_from(edges)
    ret = {}
    e2e_fids = {}
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
            ret[app] = []
            e2e_fids[app] = float("nan")
            continue

        G = _build_graph(routing_mode, edges, base_graph, cap, threshold)

        if not nx.has_path(G, src, dst):
            ret[app] = []
            e2e_fids[app] = float("nan")
            continue

        if routing_mode == "smallest":
            path_list, selected_delta, selected_e2e_fid = smallest_bottleneck(
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
                provisioning,
            )
            if not path_list:
                ret[app] = []
                e2e_fids[app] = float("nan")
                continue
            _update_capacity(path_list[0], selected_delta, cap)
            ret[app] = path_list
            e2e_fids[app] = selected_e2e_fid
            continue
        elif routing_mode == "capacity":
            path_list, selected_delta, selected_e2e_fid = capacity_threshold(
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
                rng,
                provisioning,
            )
            if not path_list:
                ret[app] = []
                e2e_fids[app] = float("nan")
                continue
            _update_capacity(path_list[0], selected_delta, cap)
            ret[app] = path_list
            e2e_fids[app] = selected_e2e_fid
            continue
        elif routing_mode == "least":
            path_list, selected_delta, selected_e2e_fid = least_capacity(
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
                provisioning,
            )
            if not path_list:
                ret[app] = []
                e2e_fids[app] = float("nan")
                continue
            _update_capacity(path_list[0], selected_delta, cap)
            ret[app] = path_list
            e2e_fids[app] = selected_e2e_fid
            continue
        elif routing_mode == "highest":
            path_list, selected_e2e_fid = highest_fidelity(
                simple_paths,
                src,
                dst,
                min_fidelity,
                rng,
                provisioning,
            )
            ret[app] = path_list
            e2e_fids[app] = selected_e2e_fid
            continue
        else:
            path_list, selected_e2e_fid = fidelity_shortest(
                simple_paths,
                src,
                dst,
                min_fidelity,
                rng,
                provisioning,
            )

            ret[app] = path_list
            e2e_fids[app] = selected_e2e_fid
    return ret, e2e_fids


def dynamic_routing(
    simple_paths: Dict[Tuple[str, str], List[List[str]]],
    resources: Dict[Tuple[str, str], float],
    current_time: float,
    src: str,
    dst: str,
    req: Dict[str, Any],
    pga_params: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[List[str], float, float]:
    """Select a path for a new request that meets the fidelity requirement and
    has the least PGA duration among all simple paths that are available at
    current_time. This function is intended to be used for dynamic routing when
    a request arrives and we need to select a path for it based on the current
    state of the network.

    Args:
        simple_paths (Dict[Tuple[str, str], List[List[str]]]): Pre-computed
            simple paths and their E2E fidelities.
        resources (Dict[Tuple[str, str], float]): Current busy-until times for
        each link.
        current_time (float): The current time in the simulation.
        src (str): The source node of the request.
        dst (str): The destination node of the request.
        req (Dict[str, Any]): The application request parameters.
        pga_params (Dict[str, float]): Parameters for the PGA duration
        calculation.
        rng (np.random.Generator): Random number generator.

    Returns:
        Tuple[List[str], float, float]: A tuple containing the selected path,
        the PGA duration of the selected path, and the E2E fidelity of the
        selected path.
    """
    selected_path = None
    least_duration = float("inf")
    selected_e2e_fid = float("nan")
    tied_count = 0

    for path in all_simple_paths(simple_paths, src, dst):
        e2e_fid, path = path[0], path[1]
        if e2e_fid < req["min_fidelity"]:
            continue
        if any(resources.get(link, 0.0) > current_time + EPS for link in path):
            continue

        n_swap = max(0, len(path) - 2)
        pga_duration = duration_pga(
            p_packet=pga_params["p_packet"],
            epr_pairs=int(pga_params["epr_pairs"]),
            n_swap=n_swap,
            memory=pga_params["memory"],
            p_swap=pga_params["p_swap"],
            p_gen=pga_params["p_gen"],
            time_slot_duration=pga_params["slot_duration"],
        )
        if pga_duration < least_duration - EPS:
            selected_path = path
            least_duration = pga_duration
            selected_e2e_fid = e2e_fid
            tied_count = 1
        elif np.isclose(pga_duration, least_duration):
            tied_count += 1
            if rng.integers(tied_count) == 0:
                selected_path = path
                least_duration = pga_duration
                selected_e2e_fid = e2e_fid
    return selected_path, least_duration, selected_e2e_fid
