"""
Simulation Of Probabilistic Job Scheduling
-------------------------------------------
This module provides classes and functions to simulate the scheduling of
jobs in a quantum network. Each job represents a Packet Generation
Attempt (PGA) that tries to generate entangled EPR pairs over a specified route
within a defined time window, considering resource availability and link busy
times. The main function, `simulate_periodicity`, orchestrates the scheduling
and execution of these jobs based on a provided static schedule, tracking
resource usage, link busy times, and job performance metrics.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

INIT_JOB_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
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
        Each pga attempts to generate a specified number of EPR pairs over a
        defined route within a given time window, considering resource
        availability and link busy times.

        Args:
            name (str): Job name for identification.
            arrival (float): Arrival time of the job in the simulation.
            start (float): Start time of the job in the simulation.
            end (float): End time of the job in the simulation.
            route (List[str]): List of nodes in the job's route.
            resources (Dict[str, float]): Dictionary of resources indexed by
            node names.
            link_busy (Dict[Tuple[str, str], float]): Dictionary to track busy
            time of links.
            p_gen (float): Probability of generating an EPR pair in a single
            trial.
            epr_pairs (int): Number of EPR pairs to generate for this job.
            slot_duration (float): Duration of a time slot for EPR generation.
            rng (np.random.Generator): Random number generator for
            probabilistic events.
            log (List[Dict[str, Any]]): Log to record job performance metrics.
            policy (str): Scheduling policy for the job, either "best_effort"
            or "deadline". If "deadline", the job will attempt to complete
            within the maximum burst time defined in durations.
            p_swap (float): Probability of swapping an EPR pair.
            memory_lifetime (int): Memory lifetime in number of time slot
            units.
            deadline (float, optional): Deadline time for the job. Defaults to
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
        wait_until = 0.0
        for node in self.route:
            wait_until = max(
                wait_until,
                self.resources.get(node, 0.0),
            )

        if wait_until > self.start + EPS:
            completion = wait_until
            status = "conflict"
        else:
            current_time = self.start
            t_budget = max(0.0, self.end - self.start)
            status = "failed"

        if t_budget > EPS and self.policy == "deadline":
            if self.slot_duration <= 0.0:
                max_attempts = 0
            else:
                max_attempts = int((t_budget + EPS) // self.slot_duration)

            succ = self._simulate_e2e_attempts(max_attempts)

            if self.epr_pairs <= 0:
                attempts_run = 0
                status = "completed"
            else:
                csum = np.cumsum(succ, dtype=int)
                hit = np.searchsorted(csum, self.epr_pairs, side="left")

                if hit < len(csum):
                    attempts_run = int(hit + 1)
                    status = "completed"
                else:
                    attempts_run = max_attempts

            current_time = self.start + attempts_run * self.slot_duration

            completion = min(self.end, current_time)

            for node in self.route:
                self.resources[node] = max(
                    self.resources.get(node, 0.0), completion
                )

            if attempts_run > 0 and self.links:
                for link in self.links:
                    busy = attempts_run * self.slot_duration
                    self.link_busy[link] = self.link_busy.get(link, 0.0) + busy

        burst = completion - self.start
        turnaround = completion - self.arrival
        waiting = turnaround - burst

        result = {
            "job": self.name,
            "arrival_time": self.arrival,
            "start_time": self.start,
            "burst_time": burst,
            "completion_time": completion,
            "turnaround_time": turnaround,
            "waiting_time": waiting,
            "status": status,
            "deadline": self.deadline,
        }
        self.log.append(result)
        return result


def simulate_periodicity(
    schedule: List[Tuple[str, float, float, float]],
    job_parameters: Dict[str, Dict[str, float]],
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    job_network_paths: Dict[str, List[str]],
    policies: Dict[str, str],
    rng: np.random.Generator,
) -> Tuple[
    pd.DataFrame,
    List[str],
    Dict[str, float],
    Dict[Tuple[str, str], Dict[str, float]],
]:
    """Simulate periodic jobs scheduling. The provided static schedule defines
    when each job starts and ends. Each job is a PGA that attempts to generate
    EPR pairs over a specified route within a scheduled time window.

    Args:
        schedule (List[Tuple[str, float, float]]): List of tuples where each
        contains the job name, start time, and end time of the scheduled job.
        job_parameters (Dict[str, Dict[str, float]]): Parameters for each job,
        including the probability of generating an EPR pair, number of
        successes, and slot duration.
        job_rel_times (Dict[str, float]): Relative release times for each
        job.
        job_periods (Dict[str, float]): Periods for each job, indicating the
        time interval between successive releases of the job.
        job_network_paths (Dict[str, list[str]]): List of nodes for each job's
        path in the network.
        policies (Dict[str, str]): Scheduling policy for each job.
        This can be "best_effort" or "deadline".
        distances (Dict[tuple, float]): Dictionary of distances between nodes
        in the network.
        rng (np.random.Generator): Random number generator for probabilistic
        events.

    Returns:
        Tuple[
            pd.DataFrame,
            List[str],
            Dict[str, float],
            Dict[Tuple[str, str], Dict[str, float]],
        ]: Contains:
            - DataFrame with job performance metrics.
            - List of job names.
            - Dictionary mapping job names to their release times.
            - Dictionary mapping undirected links to busy time and utilization.
    """
    log = []
    release_times = {}
    job_names = []

    all_nodes = {n for path in job_network_paths.values() for n in path}
    resources = {n: 0.0 for n in all_nodes}
    link_busy = {}
    min_start = float("inf")
    max_completion = 0.0

    for job_name, sched_start, sched_end, sched_deadline in schedule:
        m = INIT_JOB_RE.match(job_name)
        app, idx = (m.group(1), int(m.group(2))) if m else (job_name, 0)

        r0 = float(job_rel_times.get(app, 0.0))
        T = float(job_periods.get(app, 0.0))
        arrival = r0 + idx * T

        pga = PGA(
            name=job_name,
            arrival=arrival,
            start=sched_start,
            end=sched_end,
            route=job_network_paths[app],
            resources=resources,
            link_busy=link_busy,
            p_gen=job_parameters[app]["p_gen"],
            epr_pairs=int(job_parameters[app]["epr_pairs"]),
            slot_duration=job_parameters[app]["slot_duration"],
            rng=rng,
            log=log,
            policy=policies[app],
            p_swap=job_parameters[app]["p_swap"],
            memory_lifetime=job_parameters[app]["memory_lifetime"],
            deadline=sched_deadline,
        )
        result = pga.run()

        job_names.append(job_name)
        release_times[job_name] = sched_start
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

    return df, job_names, release_times, link_utilization
