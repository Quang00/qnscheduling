"""
Simulation Of PGAs Scheduling
-----------------------------
This module provides classes and functions to simulate the scheduling of
Packet Generation Attempts (PGAs) in a quantum network. Each PGA tries to
generate entangled EPR pairs over a specified route within a defined time
window, considering resource availability and link busy times.
"""

import heapq
import re
import time
from bisect import bisect_left
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scheduling.routing import (
    compute_path_durations,
    dynamic_routing,
    rerouting,
)
from utils.helper import compute_link_utilization, track_link_waiting

INIT_PGA_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
EPS = 1e-12


class PGA:
    def __init__(
        self,
        name: str,
        arrival: float,
        start: float,
        end: float,
        route: List[str],
        resources: Dict[Tuple[str, str], float],
        link_busy: Dict[Tuple[str, str], float],
        link_p_gens: List[float] | np.ndarray,
        epr_pairs: int,
        slot_duration: float,
        rng: np.random.Generator,
        log: List[Dict[str, Any]],
        p_swap: float,
        memory: int = 1,
        coherence: float = 0.020,
        deadline: float | None = None,
        route_links: List[Tuple[str, str]] | None = None,
    ) -> None:
        """Packet Generation Attempt (PGA) simulation. A PGA tries to
        generate EPR pairs over a specified route within a defined time window,
        considering resource availability and link busy times.

        The possible outcomes for a PGA:
        - If the PGA starts but cannot generate the required E2E EPR pairs
          within its time window, it is marked as "failed".
        - If the PGA successfully generates the required E2E EPR pairs within
          its time window, it is marked as "completed".

        The conditions for EPR generation:
        - In every time slot each link attempts to generate an EPR pair; with
          `memory` multiplexed trials per slot the per-slot success
          probability is ``1 - (1 - p_gen) ** memory``.
        - A generated link pair stays live for the coherence window (`t_mem`
          slots) and is refreshed by later successes on the same link.
        - The end-to-end BSM fires at the earliest slot where every link of
          the route simultaneously holds a live pair; it consumes the pair of
          every link (destructive, regardless of outcome) and delivers an
          end-to-end pair with probability ``p_swap ** n_swap``.
        - Delivered end-to-end pairs decohere after the same coherence
          window; the PGA completes once `epr_pairs` end-to-end pairs are
          alive simultaneously.

        Args:
            name (str): PGA identifier.
            arrival (float): Arrival time of the PGA in the simulation.
            start (float): Start time of the PGA in the simulation.
            end (float): End time of the PGA in the simulation.
            route (List[str]): List of nodes in the PGA's route.
            resources (Dict[Tuple[str, str], float]): Dictionary tracking
            when undirected links become free.
            link_busy (Dict[Tuple[str, str], float]): Dictionary to track busy
            time of links.
            link_p_gens (List[float] | np.ndarray): Per-link probabilities of
            generating an EPR pair in a single trial, in route_links order.
            epr_pairs (int): Number of EPR pairs to generate for this PGA.
            slot_duration (float): Duration of a time slot for EPR generation.
            rng (np.random.Generator): Random number generator for
            probabilistic events.
            log (List[Dict[str, Any]]): Log to record PGA performance metrics.
            p_swap (float): Probability of swapping an EPR pair.
            memory (int): Number of independent link-generation trials per
            slot (multiplexed memory modes).
            coherence (float): Coherence time in seconds of a generated pair.
            Converted internally to an integer number of slots `t_mem`; a
            pair generated at slot ``s`` is live during
            ``[s, s + t_mem - 1]``.
            deadline (float, optional): Deadline time for the PGA. Defaults to
            None, which means no deadline.
        """
        self.name = name
        self.arrival = float(arrival)
        self.start = float(start)
        self.end = float(end)
        self.route = route
        self.resources = resources
        self.link_busy = link_busy
        self.epr_pairs = int(epr_pairs)
        self.slot_duration = float(slot_duration)
        self.rng = rng
        self.log = log
        self.deadline = None if deadline is None else float(deadline)
        self.links = route_links
        self.n_swap = max(0, len(self.route) - 2)
        self.p_swap = float(p_swap)
        self.coherence = max(0, int(round(coherence / self.slot_duration)))
        self.memory = max(1, int(memory))
        self.link_p_gens = np.asarray(link_p_gens, dtype=float)
        self.link_qs = 1.0 - (1.0 - self.link_p_gens) ** self.memory

    def _sample_success_slots(self, q: float, max_slots: int) -> np.ndarray:
        slots = []
        last = -1
        chunk = int(max_slots * q * 1.25) + 16
        while last < max_slots - 1:
            gaps = self.rng.geometric(q, size=chunk)
            offsets = last + np.cumsum(gaps)
            slots.append(offsets)
            last = int(offsets[-1])
        merged = np.concatenate(slots)
        return merged[merged < max_slots]

    def _simulate_e2e_pairs(self, max_slots: int) -> np.ndarray:
        if self.coherence <= 0 or max_slots <= 0:
            return np.array([], dtype=np.int64)
        if np.any(self.link_qs <= 0.0):
            return np.array([], dtype=np.int64)

        t_mem = self.coherence
        link_slots = [
            self._sample_success_slots(q, max_slots).tolist()
            for q in self.link_qs
        ]
        if any(not s for s in link_slots):
            return np.array([], dtype=np.int64)

        sizes = [len(s) for s in link_slots]
        cursors = [0] * len(link_slots)
        p_bsms = self.p_swap**self.n_swap
        rng_random = self.rng.random
        deliveries = []
        alive_lo = 0
        fresh_from = 0

        while True:
            t = -1
            for i, s in enumerate(link_slots):
                j = bisect_left(s, fresh_from, cursors[i])
                if j == sizes[i]:
                    return np.asarray(deliveries, dtype=np.int64)
                cursors[i] = j
                if s[j] > t:
                    t = s[j]

            stable = False
            while not stable:
                stable = True
                floor = t - t_mem + 1
                for i, s in enumerate(link_slots):
                    j = cursors[i]
                    if floor > s[j]:
                        j = bisect_left(s, floor, j)
                        if j == sizes[i]:
                            return np.asarray(deliveries, dtype=np.int64)
                        cursors[i] = j
                    if s[j] > t:
                        t = s[j]
                        stable = False

            fresh_from = t + 1
            if p_bsms >= 1.0 or rng_random() < p_bsms:
                deliveries.append(t)
                while deliveries[alive_lo] <= t - t_mem:
                    alive_lo += 1
                if len(deliveries) - alive_lo >= self.epr_pairs:
                    return np.asarray(deliveries, dtype=np.int64)

    def _update_resources_and_links(
        self,
        completion: float,
        attempts_run: int,
    ) -> None:
        """Mark nodes and links busy through ``completion``."""
        for link in self.links:
            self.resources[link] = max(
                self.resources.get(link, 0.0), completion
            )

        if attempts_run > 0 and self.links:
            busy = attempts_run * self.slot_duration
            for link in self.links:
                self.link_busy[link] = self.link_busy.get(link, 0.0) + busy

    def run(self) -> Dict[str, Any]:
        attempts_run = 0
        pairs_generated = 0
        current_time = self.start
        diff = self.end - self.start
        t_budget = diff if diff > 0.0 else 0.0
        status = "failed"

        if t_budget > EPS:
            max_attempts = int((t_budget + EPS) // self.slot_duration)
            deliveries = self._simulate_e2e_pairs(max_attempts)
            attempts_run = max_attempts

            if deliveries.size:
                t_mem = self.coherence
                oldest_alive = np.searchsorted(
                    deliveries, deliveries - (t_mem - 1)
                )
                live_counts = (
                    np.arange(1, deliveries.size + 1) - oldest_alive
                )
                hits = np.flatnonzero(live_counts >= self.epr_pairs)
                if hits.size:
                    attempts_run = int(deliveries[hits[0]]) + 1
                    pairs_generated = self.epr_pairs
                    status = "completed"
                else:
                    pairs_generated = int(live_counts.max())

            current_time = self.start + attempts_run * self.slot_duration
            completion = (
                current_time if current_time < self.end else self.end
            )
            self._update_resources_and_links(completion, attempts_run)
        else:
            completion = self.start

        burst = completion - self.start
        turnaround = max(0.0, completion - self.arrival)
        waiting = max(0.0, turnaround - burst)

        result = {
            "pga": self.name,
            "arrival_time": self.arrival,
            "start_time": self.start,
            "burst_time": burst,
            "completion_time": completion,
            "turnaround_time": turnaround,
            "waiting_time": waiting,
            "pairs_generated": pairs_generated,
            "status": status,
            "deadline": self.deadline,
        }
        self.log.append(result)
        return result


def simulate_dynamic(
    app_specs: Dict[str, Dict[str, Any]],
    durations: Dict[str, float],
    pga_parameters: Dict[str, Dict[str, float]],
    pga_rel_times: Dict[str, float],
    pga_network_paths: Dict[str, List[List[str]]],
    rng: np.random.Generator,
    full_dynamic: bool = True,
    rerouting_mode: bool = False,
    all_links: List[Tuple[str, str]] | None = None,
    simple_paths: Dict[str, List[List[str]]] | None = None,
    static_routing_mode: bool = False,
    dynamic_mode: str = "wc",
    horizon_time: float | None = None,
    warmup_time: float = 0.0,
    rng_arrivals: Dict[str, np.random.Generator] | None = None,
    instance_arrival_rate: float = 10.0,
    rates: Dict[Tuple[str, str], float] = None,
    app_e2e_fidelities: Dict[str, float] | None = None,
):
    log = []
    defer_counts = {}
    pga_release_times = {}
    pga_names = []
    seen_pgas = set()

    pga_route_links = {
        app: [
            tuple(sorted((u, v)))
            for u, v in zip(paths[0][:-1], paths[0][1:], strict=False)
        ]
        for app, paths in pga_network_paths.items()
    }
    resources = {link: 0.0 for link in all_links}
    link_busy = dict.fromkeys(all_links, 0.0)
    link_busy_record = dict.fromkeys(all_links, 0.0)
    link_waiting = {
        link: {
            "total_waiting_time": 0.0,
            "block_events": 0,
            "acquisitions": 0,
        }
        for link in all_links
    }
    routing_decision_cpt = 0
    routing_decision_runtime = 0.0

    deadline_budgets = {
        app: app_specs[app].get("deadline_budget", 0.0) for app in app_specs
    }
    base_release = {app: pga_rel_times.get(app, 0.0) for app in app_specs}
    max_instances = {
        app: max(0, int(app_specs[app].get("instances", 0)))
        for app in app_specs
    }
    release_indices = {app: 0 for app in app_specs}
    mean_instance_interarrival = 1.0 / instance_arrival_rate
    poisson_next_release = {
        app: float(base_release.get(app, 0.0)) for app in app_specs
    }

    events_queue = []
    ready_queue = []

    routing_metadata = {}
    pga_best = {}
    rerouting_candidates = {}
    if full_dynamic and simple_paths is not None:
        for app in app_specs:
            _t0 = time.perf_counter()
            routing_metadata[app] = compute_path_durations(
                pga_parameters[app],
                simple_paths=simple_paths,
                src=app_specs[app]["src"],
                dst=app_specs[app]["dst"],
                rates=rates,
            )
            routing_decision_runtime += time.perf_counter() - _t0
            min_fid = app_specs[app].get("min_fidelity", 0.0)
            feasible_durs = [
                dur
                for fid, _, _, dur in routing_metadata[app]
                if fid >= min_fid
            ]
            pga_best[app] = (
                min(feasible_durs) if feasible_durs else float("nan")
            )
    elif rerouting_mode:
        for app, app_paths in pga_network_paths.items():
            _t0 = time.perf_counter()
            rerouting_candidates[app] = compute_path_durations(
                pga_parameters[app],
                provisioned_paths=app_paths,
                rates=rates,
                simple_paths=simple_paths,
                src=app_specs[app]["src"],
                dst=app_specs[app]["dst"],
            )
            routing_decision_runtime += time.perf_counter() - _t0

    def enqueue_release(app: str) -> None:
        idx = release_indices[app]
        deadline_budget = deadline_budgets[app]

        if max_instances[app] > 0 and idx >= max_instances[app]:
            return

        release = poisson_next_release[app]
        _rng_arr = (
            rng_arrivals.get(app) if rng_arrivals is not None else None
        ) or rng
        poisson_next_release[app] = release + _rng_arr.exponential(
            mean_instance_interarrival
        )

        if release >= horizon_time:
            return

        deadline = release + deadline_budget
        heapq.heappush(
            events_queue,
            (release, deadline, release, app, idx, release, "release"),
        )
        release_indices[app] += 1

    def track_defer_wait(
        defer_until: float,
        blocking_links: List[Tuple[str, str]],
    ) -> None:
        if cur_t < warmup_time or not blocking_links:
            return
        track_link_waiting(
            max(0.0, defer_until - cur_t),
            link_waiting,
            blocking_links=blocking_links,
        )

    for app in app_specs:
        enqueue_release(app)

    cur_t = 0.0

    while events_queue:
        cur_t = events_queue[0][0]

        if cur_t >= horizon_time:
            break

        while events_queue and events_queue[0][0] <= cur_t + EPS:
            (
                event_time,
                deadline,
                arrival_time,
                app,
                i,
                ready_time,
                event_type,
            ) = heapq.heappop(events_queue)
            if event_type == "release":
                enqueue_release(app)

            if event_time >= horizon_time:
                continue

            heapq.heappush(
                ready_queue,
                (deadline, ready_time, arrival_time, app, i, event_time),
            )

        if not ready_queue:
            continue

        while ready_queue:
            deadline, rdy_t, arrival_time, app, i, _ = heapq.heappop(
                ready_queue
            )

            pga_name = f"{app}{i}"
            if pga_name not in seen_pgas:
                seen_pgas.add(pga_name)
                pga_names.append(pga_name)
                pga_release_times[pga_name] = arrival_time

            routed_fid = np.nan
            if static_routing_mode:
                route_links = pga_route_links.get(app, [])
                selected_path = pga_network_paths[app][0]
                duration = durations.get(app, 0.0)
                routed_fid = (
                    app_e2e_fidelities.get(app, np.nan)
                    if app_e2e_fidelities is not None
                    else np.nan
                )
                last_available = max(
                    (resources.get(lk, 0.0) for lk in route_links),
                    default=0.0,
                )
            elif full_dynamic and simple_paths is not None:
                routing_decision_cpt += 1
                _t0 = time.perf_counter()
                routed, next_avail, next_avail_links = dynamic_routing(
                    routing_metadata[app],
                    app_specs[app].get("min_fidelity", 0.0),
                    deadline,
                    cur_t,
                    resources,
                    mode=dynamic_mode,
                )
                routing_decision_runtime += time.perf_counter() - _t0
                if routed is None:
                    if next_avail is not None:
                        defer_counts[pga_name] = (
                            defer_counts.get(pga_name, 0) + 1
                        )
                        track_defer_wait(next_avail, next_avail_links)
                        heapq.heappush(
                            events_queue,
                            (
                                next_avail,
                                deadline,
                                arrival_time,
                                app,
                                i,
                                rdy_t,
                                "resume"
                            ),
                        )
                    else:
                        log.append({
                            "pga": pga_name,
                            "arrival_time": arrival_time,
                            "ready_time": rdy_t,
                            "start_time": np.nan,
                            "burst_time": 0.0,
                            "completion_time": cur_t,
                            "turnaround_time": max(0.0, cur_t - arrival_time),
                            "waiting_time": max(0.0, cur_t - rdy_t),
                            "pairs_generated": 0,
                            "status": "drop",
                            "deadline": deadline,
                        })
                    continue
                (
                    selected_path,
                    route_links,
                    last_available,
                    duration,
                    routed_fid,
                ) = routed
            else:
                route_links = pga_route_links.get(app, [])
                selected_path = pga_network_paths[app][0]
                duration = durations.get(app, 0.0)
                if rerouting_mode:
                    routed_fid = (
                        app_e2e_fidelities.get(app, np.nan)
                        if app_e2e_fidelities is not None
                        else np.nan
                    )

                last_available = 0.0
                for link in route_links:
                    last_available = max(
                        last_available, resources.get(link, 0.0)
                    )

                if (
                    last_available > cur_t + EPS
                    and rerouting_mode
                    and last_available + duration > deadline + EPS
                ):
                    routing_decision_cpt += 1
                    _t0 = time.perf_counter()
                    alt_path = rerouting(
                        rerouting_candidates,
                        deadline,
                        cur_t,
                        app,
                        resources,
                    )
                    routing_decision_runtime += time.perf_counter() - _t0
                    if alt_path is not None:
                        (
                            selected_path,
                            route_links,
                            last_available,
                            duration,
                            routed_fid,
                        ) = alt_path

            _stamp = (
                {
                    "e2e_fidelity": routed_fid,
                    "pga_duration": duration,
                    "hops": len(selected_path) - 1,
                }
                if (full_dynamic and simple_paths is not None)
                or rerouting_mode
                or static_routing_mode
                else {}
            )
            if full_dynamic and simple_paths is not None:
                best = pga_best.get(app, float("nan"))
                _stamp["routing_efficiency"] = (
                    best / duration
                    if duration > 0 and not np.isnan(best)
                    else float("nan")
                )
            elif static_routing_mode:
                _stamp["routing_efficiency"] = 1.0

            if last_available > cur_t + EPS:
                if last_available + duration <= deadline + EPS:
                    defer_counts[pga_name] = (
                        defer_counts.get(pga_name, 0) + 1
                    )
                    track_defer_wait(
                        last_available,
                        [
                            lk for lk in route_links
                            if abs(resources.get(lk, 0.0) - last_available)
                            < EPS
                        ],
                    )
                    heapq.heappush(
                        events_queue,
                        (
                            last_available,
                            deadline,
                            arrival_time,
                            app,
                            i,
                            rdy_t,
                            "resume",
                        ),
                    )
                else:
                    log.append({
                        "pga": pga_name,
                        "arrival_time": arrival_time,
                        "ready_time": rdy_t,
                        "start_time": np.nan,
                        "burst_time": 0.0,
                        "completion_time": cur_t,
                        "turnaround_time": max(0.0, cur_t - arrival_time),
                        "waiting_time": max(0.0, cur_t - rdy_t),
                        "pairs_generated": 0,
                        "status": "drop",
                        "deadline": deadline,
                        **_stamp,
                    })
                continue

            start_time = cur_t
            completion = start_time + duration

            if deadline is not None and start_time + duration > deadline + EPS:
                log.append({
                    "pga": pga_name,
                    "arrival_time": arrival_time,
                    "ready_time": rdy_t,
                    "start_time": np.nan,
                    "burst_time": 0.0,
                    "completion_time": cur_t,
                    "turnaround_time": max(0.0, cur_t - arrival_time),
                    "waiting_time": max(0.0, cur_t - rdy_t),
                    "pairs_generated": 0,
                    "status": "drop",
                    "deadline": deadline,
                    **_stamp,
                })
                continue

            recording = start_time >= warmup_time
            link_p_gens = [rates[lk] for lk in route_links]
            pga = PGA(
                name=pga_name,
                arrival=arrival_time,
                start=start_time,
                end=completion,
                route=selected_path,
                resources=resources,
                link_busy=link_busy_record if recording else link_busy,
                link_p_gens=link_p_gens,
                epr_pairs=int(pga_parameters[app]["epr_pairs"]),
                slot_duration=pga_parameters[app]["slot_duration"],
                rng=rng,
                log=log,
                p_swap=pga_parameters[app]["p_swap"],
                memory=int(pga_parameters[app].get("memory", 1)),
                coherence=pga_parameters[app].get("coherence", 0.020),
                deadline=deadline,
                route_links=route_links,
            )
            result = pga.run()
            result["ready_time"] = float(rdy_t)
            result.update(_stamp)

            if recording:
                for lk in route_links:
                    if lk in link_waiting:
                        link_waiting[lk]["acquisitions"] += 1
                if result["completion_time"] > horizon_time:
                    excess = result["completion_time"] - horizon_time
                    for lk in route_links:
                        if lk in link_busy_record:
                            link_busy_record[lk] = max(
                                0.0, link_busy_record[lk] - excess
                            )

    df = pd.DataFrame(log)
    del log
    link_utilization = compute_link_utilization(
        link_busy_record, warmup_time, horizon_time,
    )

    return (
        df,
        pga_names,
        pga_release_times,
        link_utilization,
        link_waiting,
        routing_decision_cpt,
        routing_decision_runtime,
        defer_counts,
    )
