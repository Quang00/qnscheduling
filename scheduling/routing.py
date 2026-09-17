from collections import defaultdict
from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np

from scheduling.pga import duration_pga
from utils.helper import all_simple_paths

EPS = 1e-12


def shortest_paths(
    edges: list[tuple[str, str]], app_requests: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
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


def _compute_delta_and_links(
    path: list[str],
    req: dict[str, Any],
    p_packet: float | None,
    memory: int,
    p_swap: float,
    rates: dict[tuple[str, str], float],
    time_slot_duration: float,
    t_cut: float = 0.001,
) -> tuple[float, list[tuple[str, str]]]:
    n_swaps = max(0, len(path) - 2)
    links = [
        (min(u, v), max(u, v))
        for u, v in pairwise(path)
    ]
    effective_p_gen = min(rates[lk] for lk in links)
    pga_duration = duration_pga(
        p_packet=p_packet,
        epr_pairs=req["epr"],
        n_swap=n_swaps,
        memory=memory,
        p_swap=p_swap,
        p_gen=effective_p_gen,
        time_slot_duration=time_slot_duration,
        t_cut=t_cut,
    )
    delta = float(pga_duration) / float(req["deadline_budget"])
    return delta, links


def smallest_bottleneck(
    simple_paths: dict[tuple[str, str], list[list[str]]],
    src: str,
    dst: str,
    req: dict[str, Any],
    cap: dict[tuple[str, str], float],
    p_packet: float | None,
    memory: int,
    p_swap: float,
    rates: dict[tuple[str, str], float],
    time_slot_duration: float,
    k: int | None = None,
    t_cut: float = 0.001,
) -> tuple[list[list[str]], float, float]:
    candidates = []
    for e2e_fid, path in all_simple_paths(simple_paths, src, dst):
        if e2e_fid < req["min_fidelity"]:
            continue
        delta, links = _compute_delta_and_links(
            path, req, p_packet, memory, p_swap, rates, time_slot_duration,
            t_cut=t_cut,
        )
        post_loads = [cap[lk] + delta for lk in links]
        bottleneck = max(post_loads)
        candidates.append((bottleneck, path, delta, e2e_fid))

    if not candidates:
        return [], 0.0, float("nan")

    candidates.sort(key=lambda x: x[0])
    if k is not None:
        candidates = candidates[:k]

    selected = candidates[0]
    result = [list(c[1]) for c in candidates]
    return result, selected[2], selected[3]


def least_capacity(
    simple_paths: dict[tuple[str, str], list[list[str]]],
    src: str,
    dst: str,
    req: dict[str, Any],
    cap: dict[tuple[str, str], float],
    p_packet: float | None,
    memory: int,
    p_swap: float,
    rates: dict[tuple[str, str], float],
    time_slot_duration: float,
    rng: np.random.Generator | None = None,
    provisioning: bool = True,
    t_cut: float = 0.001,
) -> tuple[list[list[str]], float, float]:
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
            path, req, p_packet, memory, p_swap, rates, time_slot_duration,
            t_cut=t_cut,
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


def fidelity_shortest(
    simple_paths: dict[tuple[str, str], list[list[str]]],
    src: str,
    dst: str,
    min_fidelity: float,
    rng: np.random.Generator,
    provisioning: bool = True,
) -> tuple[list[list[str]], float]:
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
    simple_paths: dict[tuple[str, str], list[list[str]]],
    src: str,
    dst: str,
    min_fidelity: float,
    rng: np.random.Generator,
    provisioning: bool = True,
) -> tuple[list[list[str]], float]:
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
    path: list[str],
    delta: float,
    cap: dict[tuple[str, str], float],
) -> None:
    for u, v in pairwise(path):
        link = tuple(sorted((u, v)))
        cap[link] += delta


def find_feasible_path(
    edges: list[tuple[str, str]],
    simple_paths: dict[tuple[str, str], list[list[str]]],
    app_requests: dict[str, dict[str, Any]],
    fidelities: dict[tuple[str, str], float] | None,
    pga_rel_times: dict[str, float] | None = None,
    routing_mode: str = "shortest",
    p_packet: float | None = None,
    memory: int = 1,
    p_swap: float = 0.6,
    rates: dict[tuple[str, str], float] | None = None,
    time_slot_duration: float = 1e-4,
    rng: np.random.Generator | None = None,
    provisioning: bool = True,
    t_cut: float = 0.001,
) -> dict[str, list[list[str]]]:
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
        p_packet (float, optional): Probability of a packet being generated.
        memory (int, optional): Number of independent link-generation trials
            per slot.
        p_swap (float, optional): Probability of swapping an EPR pair in a
            single trial.
        rates (Dict[Tuple[str, str], float], optional): Per-link p_gen.
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
            {app: [] for app in app_requests},
            {app: float("nan") for app in app_requests},
        )

    G = nx.Graph()
    G.add_edges_from(edges)
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
                rates,
                time_slot_duration,
                k=None if provisioning else 1,
                t_cut=t_cut,
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
                rates,
                time_slot_duration,
                rng,
                provisioning,
                t_cut=t_cut,
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


def rerouting(
    precomputed: dict[str, list[tuple]],
    deadline: float,
    cur_t: float,
    app: str,
    resources: dict[tuple[str, str], float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[str], list[tuple[str, str]], float, float, float] | None:
    res = resources
    best = None
    best_score = None
    tied = None
    deadline_eps = deadline + EPS

    for e2e_fid, path, links, pga_duration in precomputed.get(app, []):
        avail = cur_t
        sum_wait = 0.0
        for lnk in links:
            v = res.get(lnk, 0.0)
            avail = max(avail, v)
            wait = v - cur_t
            if wait > 0.0:
                sum_wait += wait
        finish = avail + pga_duration
        if finish > deadline_eps:
            continue
        path_score = finish + sum_wait
        if best_score is None or path_score < best_score - EPS:
            best_score = path_score
            best = (path, links, avail, pga_duration, e2e_fid)
            tied = None
        elif path_score <= best_score + EPS:
            if tied is None:
                tied = [best]
            tied.append((path, links, avail, pga_duration, e2e_fid))

    if tied is not None and rng is not None:
        best = tied[int(rng.integers(len(tied)))]

    return best


def compute_path_durations(
    pga_params: dict[str, float],
    rates: dict[tuple[str, str], float],
    simple_paths: dict[tuple[str, str], list] | None = None,
    provisioned_paths: list[list[str]] | None = None,
    src: str | None = None,
    dst: str | None = None,
) -> list[tuple]:
    duration_cache = {}
    result = []
    if provisioned_paths is not None:
        fid_lookup = {
            tuple(p): fid
            for fid, p in all_simple_paths(simple_paths, src, dst)
        }
        items = (
            (fid_lookup.get(tuple(p), float("nan")), p)
            for p in provisioned_paths
        )
    else:
        items = (
            (item[0], item[1])
            for item in all_simple_paths(simple_paths, src, dst)
        )
    for e2e_fid, path in items:
        links = [
            (min(u, v), max(u, v))
            for u, v in pairwise(path)
        ]
        n_swap = max(0, len(path) - 2)
        effective_p_gen = min(rates[lk] for lk in links)
        key = (n_swap, effective_p_gen)
        if key not in duration_cache:
            duration_cache[key] = duration_pga(
                p_packet=pga_params["p_packet"],
                epr_pairs=int(pga_params["epr_pairs"]),
                n_swap=n_swap,
                memory=pga_params["memory"],
                p_swap=pga_params["p_swap"],
                p_gen=effective_p_gen,
                time_slot_duration=pga_params["slot_duration"],
                t_cut=pga_params.get("t_cut", 0.001),
            )
        result.append((e2e_fid, path, links, duration_cache[key]))
    return result


def bottlenecks(
    resources: dict[tuple[str, str], float],
    links: list[tuple[str, str]] | None,
    avail: float | None,
) -> list[tuple[str, str]]:
    if not links or avail is None:
        return []
    return [lnk for lnk in links if abs(resources[lnk] - avail) < EPS]


def dynamic_routing(
    candidate_paths: list[tuple[float, Any, list[tuple[str, str]], float]],
    min_fidelity: float,
    deadline: float,
    cur_t: float = 0.0,
    resources: dict[tuple[str, str], float] | None = None,
    mode: str = "wc",
    rng: np.random.Generator | None = None,
) -> tuple[tuple | None, float | None, list[tuple[str, str]]]:
    non_work_conserving = mode in ("nwc", "fastest")
    fastest_only = mode == "fastest"
    res = resources
    next_avail = None
    next_avail_path_links = None
    best = None
    best_score = None
    tied = None
    deadline_eps = deadline + EPS
    cur_t_eps = cur_t + EPS

    fastest_dur = None
    if fastest_only:
        for e2e_fid, _, _, pga_duration in candidate_paths:
            if e2e_fid < min_fidelity:
                continue
            if fastest_dur is None or pga_duration < fastest_dur:
                fastest_dur = pga_duration

    for e2e_fid, path, links, pga_duration in candidate_paths:
        if e2e_fid < min_fidelity:
            continue
        if fastest_only and pga_duration > fastest_dur + EPS:
            continue
        avail = cur_t
        sum_wait = 0.0
        for lnk in links:
            v = res[lnk]
            avail = max(avail, v)
            wait = v - cur_t
            if wait > 0.0:
                sum_wait += wait
        finish = avail + pga_duration
        if finish > deadline_eps:
            continue
        if avail > cur_t_eps:
            if next_avail is None or avail < next_avail:
                next_avail = avail
                next_avail_path_links = links
            if not non_work_conserving:
                continue
        path_score = finish + sum_wait
        if best_score is None or path_score < best_score - EPS:
            best_score = path_score
            best = (path, links, avail, pga_duration, e2e_fid)
            tied = None
        elif path_score <= best_score + EPS:
            if tied is None:
                tied = [best]
            tied.append((path, links, avail, pga_duration, e2e_fid))

    if tied is not None and rng is not None:
        best = tied[int(rng.integers(len(tied)))]

    if best is None:
        return (
            None,
            next_avail,
            bottlenecks(res, next_avail_path_links, next_avail),
        )

    if best[2] <= cur_t_eps:
        ret = (best[0], best[1], cur_t, best[3], best[4])
        return ret, (None if non_work_conserving else next_avail), []
    return None, best[2], bottlenecks(res, best[1], best[2])


def static_routing(
    app_requests: dict[str, dict[str, Any]],
    simple_paths: dict[tuple[str, str], list[list[str]]],
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, list[list[str]]], dict[str, float]]:
    ret = {}
    e2e_fids = {}
    path_cache = {}
    for app, req in app_requests.items():
        src, dst = req["src"], req["dst"]
        if (src, dst) in path_cache:
            chosen_path, best_fid = path_cache[(src, dst)]
            min_fid = req.get("min_fidelity", 0.0)
            if not chosen_path or best_fid < min_fid:
                ret[app] = []
                e2e_fids[app] = float("nan")
            else:
                ret[app] = [chosen_path]
                e2e_fids[app] = best_fid
            continue
        best_fid = float("-inf")
        tied = []
        for path in all_simple_paths(simple_paths, src, dst):
            e2e_fid, path_nodes = path[0], path[1]
            path_nodes_list = list(path_nodes)
            if e2e_fid > best_fid + EPS:
                best_fid = e2e_fid
                tied = [path_nodes_list]
            elif e2e_fid >= best_fid - EPS:
                tied.append(path_nodes_list)
        chosen_path = tied[0] if tied else []
        if rng is not None and len(tied) > 1:
            chosen_path = tied[int(rng.integers(len(tied)))]
        min_fid = req.get("min_fidelity", 0.0)
        if not chosen_path:
            path_cache[(src, dst)] = ([], float("nan"))
            ret[app] = []
            e2e_fids[app] = float("nan")
        else:
            chosen_fid = best_fid
            if chosen_fid < min_fid:
                path_cache[(src, dst)] = ([], chosen_fid)
                ret[app] = []
                e2e_fids[app] = float("nan")
            else:
                path_cache[(src, dst)] = (chosen_path, chosen_fid)
                ret[app] = [chosen_path]
                e2e_fids[app] = chosen_fid
    return ret, e2e_fids
