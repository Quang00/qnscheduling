"""
Simulation Of Probabilistic Job Scheduling
-------------------------------------------
This module implements a simulation of quantum network applications scheduling
using the SimPy library. It defines a `Job` class that represents a
non-preemptive job with probabilistic success in generating EPR pairs.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import simpy

from utils.helper import edges_delay

INIT_JOB_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


class Job:
    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        arrival: float,
        start: float,
        route: List[str],
        resources: Dict[str, simpy.PriorityResource],
        p_gen: float,
        epr_pairs: int,
        slot_duration: float,
        rng: np.random.Generator,
        log: List[Dict[str, Any]],
        policy: str,
        delay_map: Dict[tuple, float] = None,
        deadline: float = None,
    ) -> None:
        """Probabilistic non-preemptive job.

        Args:
            env (simpy.Environment): Environment in which the job runs.
            name (str): Job name for identification.
            arrival (float): Arrival time of the job in the simulation.
            start (float): Start time of the job in the simulation.
            route (List[str]): List of nodes in the job's route.
            resources (Dict[str, simpy.PriorityResource]): Dictionary of
            resources indexed by node names.
            p_gen (float): Probability of generating an EPR pair in a single
            trial.
            epr_pairs (int): Number of EPR pairs to generate for this job.
            slot_duration (float): Duration of a time slot for EPR generation.
            rng (np.random.Generator): Random number generator for
            probabilistic events.
            log (List[Dict[str, Any]]): Log to record job performance metrics.
            durations (Dict[str, float]): Dictionary of job durations
            indexed by job name.
            policy (str): Scheduling policy for the job, either "best_effort"
            or "deadline". If "deadline", the job will attempt to complete
            within the maximum burst time defined in durations.
        """
        self.env = env
        self.name = name
        self.arrival = arrival
        self.start = start
        self.route = route
        self.resources = resources
        self.p_gen = p_gen
        self.epr_pairs = epr_pairs
        self.slot_duration = slot_duration
        self.rng = rng
        self.log = log
        self.policy = policy
        self.delay_map = delay_map or {}
        self.deadline = deadline
        env.process(self.run())

    def run(self):
        # Wait for arrival and scheduled start
        yield self.env.timeout(max(0.0, self.arrival - self.env.now))
        yield self.env.timeout(max(0.0, self.start - self.env.now))

        delays = []
        prev = None
        for node in self.route:
            if prev is not None:
                delays.append(self.delay_map.get((prev, node), 0.0))
            prev = node

        reqs = [self.resources[node].request() for node in self.route]
        yield simpy.AllOf(self.env, reqs)
        requests = list(zip(self.route, reqs))

        t0 = self.env.now
        successes = 0

        if self.policy == "deadline":
            if self.deadline is None:
                time_budget = 0.0
            else:
                time_budget = max(0.0, self.deadline - t0)
            status = "failed"
            while successes < self.epr_pairs:
                for delay in delays:
                    if delay <= 0:
                        continue
                    if (self.env.now - t0 + delay) > time_budget:
                        break
                    yield self.env.timeout(delay)

                # Check remaining budget
                elapsed = self.env.now - t0
                if elapsed >= time_budget:
                    break

                wait = min(self.slot_duration, time_budget - elapsed)
                yield self.env.timeout(wait)
                if wait < self.slot_duration:
                    break

                # Bernoulli attempt
                if self.rng.random() < self.p_gen:
                    successes += 1
            else:
                status = "completed"
        elif self.policy == "best_effort":
            while successes < self.epr_pairs:
                for delay in delays:
                    if delay > 0:
                        yield self.env.timeout(delay)
                yield self.env.timeout(self.slot_duration)
                if self.rng.random() < self.p_gen:
                    successes += 1
            status = "completed"

        completion = self.env.now

        # Release resources
        for node, req in requests:
            self.resources[node].release(req)

        # Metrics
        burst = completion - t0
        turnaround = completion - self.arrival
        waiting = turnaround - burst

        result = {
            "job": self.name,
            "arrival_time": self.arrival,
            "start_time": t0,
            "burst_time": burst,
            "completion_time": completion,
            "turnaround_time": turnaround,
            "waiting_time": waiting,
            "status": status,
            "deadline": self.deadline,
        }
        self.log.append(result)


def simulate(
    schedule: List[Tuple[str, float, float]],
    job_parameters: Dict[str, Dict[str, float]],
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    job_network_paths: Dict[str, List[str]],
    policies: Dict[str, str],
    distances: Dict[tuple, float],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Simulate job scheduling and execution.

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
        Tuple[pd.DataFrame, List[str], Dict[str, float]]: A tuple containing:
            - DataFrame with job performance metrics.
            - List of job names.
            - Dictionary mapping job names to their release times.
    """
    env = simpy.Environment()
    log = []
    release_times = {}
    job_names = []
    retry_map = {}
    next_idx = {}

    for name, _, _ in schedule:
        initial_job = INIT_JOB_RE.match(name)
        if initial_job:
            b, idx = initial_job.group(1), int(initial_job.group(2))
            next_idx[b] = max(next_idx.get(b, 0), idx + 1)
        retry_map[name] = name

    all_nodes = {n for path in job_network_paths.values() for n in path}
    resources = {n: simpy.PriorityResource(env, capacity=1) for n in all_nodes}
    delay_map = edges_delay(distances)

    for inst_name, sched_start, deadline in schedule:
        initial_job = INIT_JOB_RE.match(inst_name)
        base, idx = (
            (initial_job.group(1), int(initial_job.group(2)))
            if initial_job
            else (inst_name, 0)
        )
        rel = job_rel_times.get(base, 0.0) + idx * job_periods.get(base, 0.0)
        release_times[inst_name] = rel
        job_names.append(inst_name)
        params = job_parameters[base]

        Job(
            env=env,
            name=inst_name,
            arrival=rel,
            start=sched_start,
            route=job_network_paths[base],
            resources=resources,
            p_gen=params["p_gen"],
            epr_pairs=int(params["epr_pairs"]),
            slot_duration=params["slot_duration"],
            rng=rng,
            log=log,
            policy=policies[base],
            delay_map=delay_map,
            deadline=deadline,
        )

    env.run()
    return pd.DataFrame(log), job_names, release_times
