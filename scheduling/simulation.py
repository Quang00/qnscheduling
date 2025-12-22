"""
Simulation Of PGAs Scheduling
-----------------------------
This module provides classes and functions to simulate the scheduling of
Packet Generation Attempts (PGAs) in a quantum network. Each PGA tries to
generate entangled EPR pairs over a specified route within a defined time
window, considering resource availability and link busy times. The function,
`simulate_static`, simulates a static schedule of PGAs and returns performance
metrics, link utilizations, and other relevant data. While the function
`simulate_dynamic` implement a dynamic scheduling approach.
"""

import heapq
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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
        p_gen: float,
        epr_pairs: int,
        slot_duration: float,
        rng: np.random.Generator,
        log: List[Dict[str, Any]],
        policy: str,
        p_swap: float,
        memory_lifetime: int,
        deadline: float | None = None,
        route_links: List[Tuple[str, str]] | None = None,
    ) -> None:
        """Packet Generation Attempt (PGA) simulation. The goal is to
        simulate end-to-end EPR pairs generation in a quantum network.
        Each PGA attempts to generate a specified number of EPR pairs over a
        defined route within a given time window, considering resource
        availability and link busy times.

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
            p_gen (float): Probability of generating an EPR pair in a single
            trial.
            epr_pairs (int): Number of EPR pairs to generate for this PGA.
            slot_duration (float): Duration of a time slot for EPR generation.
            rng (np.random.Generator): Random number generator for
            probabilistic events.
            log (List[Dict[str, Any]]): Log to record PGA performance metrics.
            policy (str): Scheduling policy for the PGA, either "best_effort"
            or "deadline". If "deadline", the PGA will attempt to complete
            within the maximum burst time defined in durations.
            p_swap (float): Probability of swapping an EPR pair.
            memory_lifetime (int): Memory lifetime in number of time slot
            units.
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
        self.p_gen = float(p_gen)
        self.epr_pairs = int(epr_pairs)
        self.slot_duration = float(slot_duration)
        self.rng = rng
        self.log = log
        self.policy = policy
        self.deadline = None if deadline is None else float(deadline)
        self.links = route_links
        self.n_swap = max(0, len(self.route) - 2)
        self.p_swap = float(p_swap)
        self.memory_lifetime = max(0, int(memory_lifetime))

    def _simulate_e2e_attempts(self, max_attempts: int) -> np.ndarray:
        """Single end-to-end entanglement for a batch of attempts."""
        if self.memory_lifetime <= 0 or self.p_gen <= 0.0 or max_attempts <= 0:
            return np.zeros(max_attempts, dtype=bool)

        n_links = self.n_swap + 1
        t_mem = self.memory_lifetime
        size = (max_attempts, n_links)

        starts = self.rng.geometric(self.p_gen, size=size) - 1
        ends = starts + (t_mem - 1)

        candidate = starts.max(axis=1)
        last_valid = ends.min(axis=1)

        succ = (candidate < t_mem) & (last_valid >= candidate)

        if self.n_swap > 0:
            if self.p_swap < 1.0:
                p_bsms = self.p_swap**self.n_swap
                swap_ok = self.rng.random(max_attempts) < p_bsms
                succ &= swap_ok

        return succ

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
        wait_until = 0.0

        for link in self.links:
            wait_until = max(wait_until, self.resources.get(link, 0.0))

        if wait_until > self.start + EPS:
            completion = wait_until
            status = "deadline_miss"
        else:
            current_time = self.start
            t_budget = max(0.0, self.end - self.start)
            status = "failed"

            if t_budget > EPS and self.policy == "deadline":
                max_attempts = int((t_budget + EPS) // self.slot_duration)
                succ = self._simulate_e2e_attempts(max_attempts)
                csum = (
                    np.cumsum(succ, dtype=int)
                    if len(succ)
                    else np.array([], dtype=int)
                )
                hit = (
                    np.searchsorted(csum, self.epr_pairs, side="left")
                    if len(csum)
                    else len(csum)
                )

                if len(csum) and hit < len(csum):
                    attempts_run = int(hit + 1)
                    pairs_generated = int(csum[attempts_run - 1])
                    status = "completed"
                else:
                    attempts_run = max_attempts
                    pairs_generated = int(csum[-1]) if len(csum) else 0

                current_time = self.start + attempts_run * self.slot_duration
                completion = min(self.end, current_time)
                self._update_resources_and_links(completion, attempts_run)
            elif self.policy == "best_effort":
                while pairs_generated < self.epr_pairs:
                    succ = self._simulate_e2e_attempts(1)
                    pairs_generated += int(succ.sum())
                    attempts_run += 1

                completion = self.start + attempts_run * self.slot_duration
                status = (
                    "completed" if pairs_generated >= self.epr_pairs
                    else "failed"
                )
                self._update_resources_and_links(completion, attempts_run)
            else:
                completion = self.start

        burst = completion - self.start
        turnaround = completion - self.arrival
        waiting = self.start - self.arrival

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
            "deadline": self.deadline if self.policy == "deadline" else None,
        }
        self.log.append(result)
        return result


def simulate_static(
    schedule: List[Tuple[str, float, float, float]],
    app_specs: Dict[str, Dict[str, Any]],
    pga_parameters: Dict[str, Dict[str, float]],
    pga_rel_times: Dict[str, float],
    pga_periods: Dict[str, float],
    pga_network_paths: Dict[str, List[str]],
    policies: Dict[str, str],
    rng: np.random.Generator,
) -> Tuple[
    pd.DataFrame,
    List[str],
    Dict[str, float],
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[Tuple[str, str], Dict[str, float | int]],
]:
    """Simulate periodic PGA scheduling. The provided static schedule defines
    when each PGA starts and ends. Each scheduled entry is a PGA that attempts
    to generate EPR pairs over a specified route within a scheduled time
    window.

    Args:
        schedule (List[Tuple[str, float, float, float]]): List of tuples where
        each contains the PGA name, start time, end time, and deadline of the
        scheduled PGA.
        pga_parameters (Dict[str, Dict[str, float]]): Parameters for each PGA,
        including the probability of generating an EPR pair, number of
        required successes, and slot duration.
        pga_rel_times (Dict[str, float]): Relative release times for each PGA.
        pga_periods (Dict[str, float]): Periods for each PGA, indicating the
        time interval between successive releases of the PGA.
        pga_network_paths (Dict[str, list[str]]): List of nodes for each PGA's
        path in the network.
        policies (Dict[str, str]): Scheduling policy for each PGA. This can be
        "best_effort" or "deadline".
        rng (np.random.Generator): Random number generator for probabilistic
        events.

    Returns:
        Tuple[
            pd.DataFrame,
            List[str],
            Dict[str, float],
            Dict[Tuple[str, str], Dict[str, float]],
        ]: Contains:
            - DataFrame with PGA performance metrics.
            - List of PGA names.
            - Dictionary mapping PGA names to their release times.
            - Dictionary mapping undirected links to busy time and utilization.
            - Dictionary mapping undirected links to waiting metrics.
    """
    log = []
    pga_release_times = {}
    pga_names = []

    instances_required = {
        app: max(0, int(app_specs.get(app, {}).get("instances", 0)))
        for app in pga_network_paths
    }
    total_required = sum(instances_required.values())
    completed_instances = {app: 0 for app in instances_required}
    completed_total = 0

    pga_route_links = {
        app: [
            tuple(sorted((u, v)))
            for u, v in zip(path[:-1], path[1:], strict=False)
        ]
        for app, path in pga_network_paths.items()
    }
    all_links = {link for links in pga_route_links.values() for link in links}
    resources = {link: 0.0 for link in all_links}
    link_busy = {}
    link_waiting = {
        link: {"total_waiting_time": 0.0, "pga_waited": 0}
        for link in all_links
    }
    min_start = float("inf")
    max_completion = 0.0

    for pga_name, sched_start, sched_end, sched_deadline in schedule:
        m = INIT_PGA_RE.match(pga_name)
        app, idx = (m.group(1), int(m.group(2))) if m else (pga_name, 0)

        required = instances_required.get(app, 0)
        if required > 0 and completed_instances[app] >= required:
            continue

        r0 = float(pga_rel_times.get(app, 0.0))
        T = float(pga_periods.get(app, 0.0))
        arrival = r0 + idx * T

        pga = PGA(
            name=pga_name,
            arrival=arrival,
            start=sched_start,
            end=sched_end,
            route=pga_network_paths[app],
            resources=resources,
            link_busy=link_busy,
            p_gen=pga_parameters[app]["p_gen"],
            epr_pairs=int(pga_parameters[app]["epr_pairs"]),
            slot_duration=pga_parameters[app]["slot_duration"],
            rng=rng,
            log=log,
            policy=policies[app],
            p_swap=pga_parameters[app]["p_swap"],
            memory_lifetime=pga_parameters[app]["memory_lifetime"],
            deadline=sched_deadline,
            route_links=pga_route_links.get(app),
        )
        result = pga.run()
        waiting_time = max(0.0, float(result.get("waiting_time", 0.0)))
        track_link_waiting(
            pga_route_links.get(app),
            waiting_time,
            link_waiting,
        )

        pga_names.append(pga_name)
        pga_release_times[pga_name] = sched_start
        min_start = min(min_start, result["start_time"])
        max_completion = max(max_completion, result["completion_time"])

        if result.get("status") == "completed":
            completed_instances[app] += 1
            completed_total += 1
            if completed_total == total_required:
                break

    df = pd.DataFrame(log)
    link_utilization = compute_link_utilization(
        link_busy,
        min_start,
        max_completion,
    )

    return df, pga_names, pga_release_times, link_utilization, link_waiting


def simulate_dynamic(
    app_specs: Dict[str, Dict[str, Any]],
    durations: Dict[str, float],
    pga_parameters: Dict[str, Dict[str, float]],
    pga_rel_times: Dict[str, float],
    pga_network_paths: Dict[str, List[str]],
    rng: np.random.Generator,
    arrival_rate: float | None = None,
) -> Tuple[
    pd.DataFrame,
    List[str],
    Dict[str, float],
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[Tuple[str, str], Dict[str, float | int]],
]:
    """Simulate online dynamic PGA scheduling. PGAs are released periodically
    according to their specified release times and periods. At each time step,
    the scheduler checks for newly released PGAs and adds them to the ready
    queue. The scheduler then attempts to start PGAs from the ready queue based
    on resource availability and deadlines. The scheduling is based on earliest
    deadline first (EDF) policy.

    Args:
        app_specs (Dict[str, Dict[str, Any]]): Application specifications.
        durations (Dict[str, float]): Duration of each application.
        pga_parameters (Dict[str, Dict[str, float]]): Parameters for each PGA.
        pga_rel_times (Dict[str, float]): Release times for each PGA.
        pga_network_paths (Dict[str, List[str]]): Network paths for each PGA.
        rng (np.random.Generator): Random number generator.
        arrival_rate (float | None): Mean rate lambda for Poisson arrivals.
        When None, arrivals remain periodic.

    Returns:
        Tuple[
            pd.DataFrame,
            List[str],
            Dict[str, float],
            Dict[Tuple[str, str], Dict[str, float]],
        ]: Contains:
            - DataFrame with PGA performance metrics.
            - List of PGA names.
            - Dictionary mapping PGA names to their release times.
            - Dictionary mapping undirected links to busy time and utilization.
            - Dictionary mapping undirected links to waiting metrics.
    """
    log = []
    pga_release_times = {}
    pga_names = []
    seen_pgas = set()
    drop_logged = set()

    pga_route_links = {
        app: [
            tuple(sorted((u, v)))
            for u, v in zip(path[:-1], path[1:], strict=False)
        ]
        for app, path in pga_network_paths.items()
    }
    all_links = {link for links in pga_route_links.values() for link in links}
    resources = {link: 0.0 for link in all_links}
    link_busy = {}
    link_waiting = {
        link: {"total_waiting_time": 0.0, "pga_waited": 0}
        for link in all_links
    }
    min_start = float("inf")
    max_completion = 0.0

    periods = {app: app_specs[app].get("period") for app in app_specs}
    inst_req = {app: app_specs[app].get("instances") for app in app_specs}
    base_release = {app: pga_rel_times.get(app, 0.0) for app in app_specs}
    completed_instances = {app: 0 for app in app_specs}
    release_indices = {app: 0 for app in app_specs}
    poisson_enabled = arrival_rate is not None and arrival_rate > 0.0
    poisson = (1.0 / arrival_rate) if poisson_enabled else None
    poisson_next_release = (
        {
            app: base_release.get(app, 0.0) + rng.exponential(poisson)
            for app in app_specs
        }
        if poisson_enabled
        else {}
    )

    events_queue = []
    ready_queue = []

    def enqueue_release(app: str) -> None:
        if inst_req[app] <= completed_instances[app]:
            return
        idx = release_indices[app]
        period = periods[app]

        if poisson_enabled:
            release = poisson_next_release.get(app, base_release[app])
            poisson_next_release[app] = release + rng.exponential(poisson)
        else:
            release = base_release[app] + period * idx

        deadline = release + period
        heapq.heappush(events_queue, (release, deadline, release, app, idx))
        release_indices[app] += 1

    for app in app_specs:
        enqueue_release(app)

    t = 0.0

    while events_queue or ready_queue:
        if not ready_queue:
            t = events_queue[0][0]

        while events_queue and events_queue[0][0] <= t + EPS:
            rdy_t, deadline, arrival_time, app, i = heapq.heappop(events_queue)
            heapq.heappush(
                ready_queue, (deadline, rdy_t, arrival_time, app, i)
            )

        if not ready_queue:
            continue

        while ready_queue:
            deadline, rdy_t, arrival_time, app, i = heapq.heappop(ready_queue)

            pga_name = f"{app}{i}"
            if pga_name not in seen_pgas:
                seen_pgas.add(pga_name)
                pga_names.append(pga_name)
                pga_release_times[pga_name] = arrival_time

            route_links = pga_route_links.get(app, [])
            duration = durations.get(app, 0.0)

            last_available = 0.0
            for link in route_links:
                last_available = max(last_available, resources.get(link, 0.0))

            if last_available > t + EPS:
                if last_available + duration > deadline + EPS:
                    if (app, i) not in drop_logged:
                        drop_logged.add((app, i))

                        wait = max(0.0, t - rdy_t)
                        result = {
                            "pga": pga_name,
                            "arrival_time": arrival_time,
                            "start_time": np.nan,
                            "burst_time": 0.0,
                            "completion_time": t,
                            "turnaround_time": t - arrival_time,
                            "waiting_time": wait,
                            "pairs_generated": 0,
                            "status": "drop",
                            "deadline": deadline,
                        }
                        log.append(result)

                        track_link_waiting(route_links, wait, link_waiting)

                    if inst_req[app] > completed_instances[app]:
                        enqueue_release(app)
                else:
                    wait = max(0.0, t - rdy_t)
                    result = {
                        "pga": pga_name,
                        "arrival_time": arrival_time,
                        "start_time": np.nan,
                        "burst_time": 0.0,
                        "completion_time": t,
                        "turnaround_time": t - arrival_time,
                        "waiting_time": wait,
                        "pairs_generated": 0,
                        "status": "defer",
                        "deadline": deadline,
                    }
                    log.append(result)
                    track_link_waiting(route_links, wait, link_waiting)
                    heapq.heappush(
                        events_queue,
                        (last_available, deadline, arrival_time, app, i)
                    )
                continue

            start_time = t
            period = periods[app]
            completion = start_time + duration

            if completion > deadline + EPS or duration > period + EPS:
                result = {
                    "pga": pga_name,
                    "arrival_time": arrival_time,
                    "start_time": start_time,
                    "burst_time": 0.0,
                    "completion_time": start_time,
                    "turnaround_time": start_time - arrival_time,
                    "waiting_time": start_time - arrival_time,
                    "pairs_generated": 0,
                    "status": "drop",
                    "deadline": deadline,
                }
                log.append(result)
                track_link_waiting(
                    route_links, result["waiting_time"], link_waiting
                )

                if duration > period + EPS:
                    completed_instances[app] += 1

                if inst_req[app] > completed_instances[app]:
                    enqueue_release(app)
                continue

            pga = PGA(
                name=pga_name,
                arrival=arrival_time,
                start=start_time,
                end=completion,
                route=pga_network_paths[app],
                resources=resources,
                link_busy=link_busy,
                p_gen=pga_parameters[app]["p_gen"],
                epr_pairs=int(pga_parameters[app]["epr_pairs"]),
                slot_duration=pga_parameters[app]["slot_duration"],
                rng=rng,
                log=log,
                policy=app_specs[app].get("policy"),
                p_swap=pga_parameters[app]["p_swap"],
                memory_lifetime=pga_parameters[app]["memory_lifetime"],
                deadline=deadline,
                route_links=route_links,
            )
            result = pga.run()

            track_link_waiting(
                route_links, result.get("waiting_time", 0.0), link_waiting
            )

            min_start = min(min_start, result["start_time"])
            max_completion = max(max_completion, result["completion_time"])

            status = result.get("status", "")
            if status == "completed":
                completed_instances[app] += 1
                if inst_req[app] > completed_instances[app]:
                    enqueue_release(app)
                continue

            next_ready_time = result["completion_time"] + EPS
            if next_ready_time + duration <= deadline + EPS:
                if status == "failed":
                    result["status"] = "retry"
                heapq.heappush(
                    events_queue,
                    (next_ready_time, deadline, arrival_time, app, i)
                )
            else:
                if status == "failed":
                    result["status"] = "drop"
                if inst_req[app] > completed_instances[app]:
                    enqueue_release(app)

    df = pd.DataFrame(log)
    link_utilization = compute_link_utilization(
        link_busy, min_start, max_completion
    )

    for link in all_links:
        link_waiting.setdefault(
            link, {"total_waiting_time": 0.0, "pga_waited": 0}
        )

    return df, pga_names, pga_release_times, link_utilization, link_waiting
