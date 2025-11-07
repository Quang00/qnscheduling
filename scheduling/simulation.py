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
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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
        resources: Dict[str, float],
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
            resources (Dict[str, float]): Dictionary of resources indexed by
            node names.
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
        self.links = [
            tuple(sorted((u, v)))
            for u, v in zip(route[:-1], route[1:], strict=False)
        ]
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
            p_swap = float(np.clip(self.p_swap, 0.0, 1.0))
            if p_swap <= 0.0:
                return np.zeros_like(succ, dtype=bool)
            elif p_swap < 1.0:
                swap_ok = self.rng.random(max_attempts) < (p_swap**self.n_swap)
                succ &= swap_ok

        return succ

    def run(self) -> Dict[str, Any]:
        attempts_run = 0
        pairs_generated = 0
        wait_until = 0.0

        for node in self.route:
            wait_until = max(wait_until, self.resources.get(node, 0.0))

        if wait_until > self.start + EPS:
            completion = wait_until
            status = "conflict"
            return

        current_time = self.start
        t_budget = max(0.0, self.end - self.start)
        status = "failed"

        if t_budget > EPS and self.policy == "deadline":
            if self.slot_duration <= 0.0:
                max_attempts = 0
                succ = np.zeros(0, dtype=bool)
            else:
                max_attempts = int((t_budget + EPS) // self.slot_duration)
                succ = self._simulate_e2e_attempts(max_attempts)

            if self.epr_pairs <= 0:
                attempts_run = 0
                pairs_generated = 0
                status = "completed"
            else:
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

            for node in self.route:
                self.resources[node] = max(
                    self.resources.get(node, 0.0), completion
                )

            if attempts_run > 0 and self.links:
                busy = attempts_run * self.slot_duration
                for link in self.links:
                    self.link_busy[link] = self.link_busy.get(link, 0.0) + busy
        else:
            completion = self.start

        burst = completion - self.start
        turnaround = completion - self.arrival
        waiting = turnaround - burst

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


def simulate_static(
    schedule: List[Tuple[str, float, float, float]],
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
    """
    log = []
    pga_release_times = {}
    pga_names = []

    all_nodes = {n for path in pga_network_paths.values() for n in path}
    resources = {n: 0.0 for n in all_nodes}
    link_busy = {}
    min_start = float("inf")
    max_completion = 0.0

    for pga_name, sched_start, sched_end, sched_deadline in schedule:
        m = INIT_PGA_RE.match(pga_name)
        app, idx = (m.group(1), int(m.group(2))) if m else (pga_name, 0)

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
        )
        result = pga.run()

        pga_names.append(pga_name)
        pga_release_times[pga_name] = sched_start
        min_start = min(min_start, result["start_time"])
        max_completion = max(max_completion, result["completion_time"])

    df = pd.DataFrame(log)
    horizon = max_completion - min_start if log else 0.0
    link_utilization = {}
    if horizon > 0:
        link_utilization = {
            link: {
                "busy_time": busy,
                "utilization": busy / horizon,
            }
            for link, busy in link_busy.items()
        }
    elif link_busy:
        link_utilization = {
            link: {"busy_time": busy, "utilization": 0.0}
            for link, busy in link_busy.items()
        }

    return df, pga_names, pga_release_times, link_utilization


def simulate_dynamic(
    app_specs: Dict[str, Dict[str, Any]],
    durations: Dict[str, float],
    parallel_map: dict[str, set[str]],
    pga_parameters: Dict[str, Dict[str, float]],
    pga_rel_times: Dict[str, float],
    pga_network_paths: Dict[str, List[str]],
    rng: np.random.Generator,
):
    df = pd.DataFrame(
        columns=[
            "pga",
            "arrival_time",
            "start_time",
            "burst_time",
            "completion_time",
            "turnaround_time",
            "waiting_time",
            "pairs_generated",
            "status",
            "deadline",
        ]
    )
    log = []
    pga_release_times = {}
    pga_names = []

    all_nodes = {n for path in pga_network_paths.values() for n in path}
    resources = {n: 0.0 for n in all_nodes}
    link_busy = {}
    min_start = float("inf")
    max_completion = 0.0

    app_idx = defaultdict(int)
    apps_deadlines = {
        app: pga_rel_times.get(app, 0.0) + spec.get("period", 0.0)
        for app, spec in app_specs.items()
    }

    priority_queue = [
        (deadline, app) for app, deadline in apps_deadlines.items()
    ]
    heapq.heapify(priority_queue)

    while priority_queue:
        deadline, app = heapq.heappop(priority_queue)
        route = pga_network_paths[app]
        release = float(pga_rel_times.get(app, 0.0))
        latest_available_resource_time = max(
            (resources.get(n, 0.0) for n in route), default=0.0
        )
        earliest_start = max(release, latest_available_resource_time)
        completion = earliest_start + durations[app]
        idx = app_idx[app]
        pga_name = f"{app}{idx}"

        if completion > deadline:
            log.append(
                {
                    "pga": pga_name,
                    "arrival_time": release,
                    "start_time": earliest_start,
                    "burst_time": 0.0,
                    "completion_time": earliest_start,
                    "turnaround_time": 0.0,
                    "waiting_time": 0.0,
                    "pairs_generated": 0,
                    "status": "failed",
                    "deadline": deadline,
                }
            )
            pga_names.append(pga_name)
            pga_release_times[pga_name] = release
            app_idx[app] += 1
            continue

        pga = PGA(
            name=pga_name,
            arrival=release,
            start=earliest_start,
            end=earliest_start + durations[app],
            route=route,
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
        )
        result = pga.run()

        pga_names.append(pga_name)
        pga_release_times[pga_name] = release
        min_start = min(min_start, result["start_time"])
        max_completion = max(max_completion, result["completion_time"])
        app_idx[app] += 1

        status = result.get("status", "")
        if status == "failed":
            period = app_specs[app].get("period", 0.0)
            next_release = release + period
            pga_rel_times[app] = max(
                next_release, result["completion_time"] + EPS
            )
            next_deadline = pga_rel_times[app] + period
            heapq.heappush(priority_queue, (next_deadline, app))

    df = pd.DataFrame(log)
    horizon = max_completion - min_start if log else 0.0
    link_utilization = {}
    if horizon > 0:
        link_utilization = {
            link: {
                "busy_time": busy,
                "utilization": busy / horizon,
            }
            for link, busy in link_busy.items()
        }
    elif link_busy:
        link_utilization = {
            link: {"busy_time": busy, "utilization": 0.0}
            for link, busy in link_busy.items()
        }

    return df, pga_names, pga_rel_times, link_utilization
