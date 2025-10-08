"""
Simulation Of Probabilistic Job Scheduling
-------------------------------------------
This module implements a discrete-event simulation for quantum network
applications. It defines a `Job` class that represents a non-preemptive job
with probabilistic success in generating EPR pairs. The function
`simulate_periodicity` orchestrates the scheduling and execution of these jobs
based on a provided static schedule.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scheduling.pga import probability_e2e
from utils.helper import edges_delay

INIT_JOB_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
EPS = 1e-12


class Job:
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
        delay_map: Dict[Tuple[str, str], float] | None = None,
        deadline: float | None = None,
    ) -> None:
        """Probabilistic non-preemptive job.

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
            delay_map (Dict[tuple, float], optional): Dictionary of delays
            between nodes. Defaults to None, which means no additional delays.
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
        self.delay_map = {}
        self.deadline = None if deadline is None else float(deadline)

        self.delays = []
        self.links = []
        total_delay = 0.0
        max_prefix = 0.0
        prev = None
        for node in self.route:
            if prev is not None:
                link = (prev, node)
                self.links.append(link)
                delay = max(
                    0.0,
                    self.delay_map.get(
                        (prev, node),
                        self.delay_map.get((node, prev), 0.0),
                    ),
                )
                total_delay += delay
                if total_delay > max_prefix:
                    max_prefix = total_delay
                self.delays.append(delay)
            prev = node

        self.total_delay = total_delay
        self.max_delay_prefix = max_prefix
        self.per_attempt_time = self.total_delay + self.slot_duration

        self.n_swap = max(0, len(self.route) - 2)
        self.p_e2e = probability_e2e(
            self.n_swap, int(memory_lifetime), self.p_gen, float(p_swap)
        )

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

            if (
                t_budget > EPS
                and self.policy == "deadline"
                and self.per_attempt_time > EPS
                and self.max_delay_prefix <= t_budget + EPS
            ):
                max_attempts = int((t_budget + EPS) // self.per_attempt_time)
                p_success = max(0.0, min(1.0, self.p_e2e))

                if max_attempts > 0 and p_success > 0.0:
                    failures = self.rng.negative_binomial(
                        self.epr_pairs, p_success
                    )
                    trials_needed = failures + self.epr_pairs
                    if trials_needed <= max_attempts:
                        status = "completed"
                        attempts_run = trials_needed
                    else:
                        attempts_run = max_attempts
                elif max_attempts > 0:
                    attempts_run = max_attempts

                current_time = (
                    self.start + attempts_run * self.per_attempt_time
                )

            completion = min(self.end, current_time)

            for node in self.route:
                self.resources[node] = max(
                    self.resources.get(node, 0.0), completion
                )

            if attempts_run > 0 and self.links:
                per_attempt_slot = self.slot_duration
                for link, delay in zip(self.links, self.delays, strict=True):
                    busy = attempts_run * (per_attempt_slot + delay)
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
    distances: Dict[tuple, float],
    rng: np.random.Generator,
) -> Tuple[
    pd.DataFrame,
    List[str],
    Dict[str, float],
    Dict[Tuple[str, str], Dict[str, float]],
]:
    """Simulate periodic jobs scheduling.

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
    delay_map = edges_delay(distances)
    link_busy = {}
    min_start = float("inf")
    max_completion = 0.0

    for inst_name, sched_start, sched_end, sched_deadline in schedule:
        m = INIT_JOB_RE.match(inst_name)
        app, idx = (m.group(1), int(m.group(2))) if m else (inst_name, 0)

        r0 = float(job_rel_times.get(app, 0.0))
        T = float(job_periods.get(app, 0.0))
        arrival = r0 + idx * T

        job = Job(
            name=inst_name,
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
            delay_map=delay_map,
            deadline=sched_deadline,
        )
        result = job.run()

        job_names.append(inst_name)
        release_times[inst_name] = sched_start
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
