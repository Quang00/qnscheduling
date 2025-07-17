"""
Siimulation Of Probabilistic Job Scheduling
-------------------------------------------
This module simulates the scheduling and execution of probabilistic jobs
(entanglement generation) in a quantum network. It uses the SimPy library
to model the environment and resources, and it records job performance metrics
such as burst time, turnaround time, and waiting time. It also generates a
summary of the simulation results, including makespan, throughput, and average
waiting time.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import simpy

BASE_JOB_RE = re.compile(r"^([A-Za-z]+)")


class Job:
    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        arrival: float,
        start: float,
        route: List[str],
        resources: Dict[str, simpy.Resource],
        p_gen: float,
        epr_pairs: int,
        slot_duration: float,
        rng: np.random.Generator,
        log: List[Dict[str, Any]],
        durations: Dict[str, float],
    ) -> None:
        """Job class to represent a probabilistic job in the simulation.

        Args:
            env (simpy.Environment): Environment in which the job runs.
            name (str): Job name for identification.
            arrival (float): Arrival time of the job in the simulation.
            start (float): Start time of the job in the simulation.
            route (List[str]): List of nodes in the job's route.
            resources (Dict[str, simpy.Resource]): Resources available at each
            node.
            p_gen (float): Probability of generating an EPR pair in a single
            trial.
            epr_pairs (int): Number of EPR pairs to generate for this job.
            slot_duration (float): Duration of a time slot for EPR generation.
            rng (np.random.Generator): Random number generator for
            probabilistic events.
            log (List[Dict[str, Any]]): Log to record job performance metrics.
            durations (Dict[str, float]): Dictionary of job durations
            indexed by job name.
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
        self.durations = durations
        env.process(self.run())

    def run(self):
        # Wait for arrival and scheduled start
        yield self.env.timeout(max(0.0, self.arrival - self.env.now))
        yield self.env.timeout(max(0.0, self.start - self.env.now))

        requests = []
        for node in self.route:
            req = self.resources[node].request()
            yield req
            requests.append((node, req))

        t0 = self.env.now
        successes = 0
        base = BASE_JOB_RE.match(self.name).group(1)
        max_burst = self.durations.get(base, float("inf"))

        while successes < self.epr_pairs:
            elapsed = self.env.now - t0
            if elapsed >= max_burst:
                status = "failed"
                break

            wait_time = min(self.slot_duration, max_burst - elapsed)
            yield self.env.timeout(wait_time)

            if wait_time < self.slot_duration:
                status = "failed"
                break

            if self.rng.random() < self.p_gen:
                successes += 1
        else:
            status = "completed"

        completion = self.env.now

        for node, req in requests:
            self.resources[node].release(req)

        burst = completion - t0
        turnaround = completion - self.arrival
        waiting = turnaround - burst

        self.log.append(
            {
                "job": self.name,
                "arrival_time": self.arrival,
                "start_time": t0,
                "burst_time": burst,
                "completion_time": completion,
                "turnaround_time": turnaround,
                "waiting_time": waiting,
                "status": status,
            }
        )


def simulate(
    schedule: List[Tuple[str, float, float]],
    job_parameters: Dict[str, Dict[str, float]],
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    job_network_paths: Dict[str, List[str]],
    durations: Dict[str, float],
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Simulate the job scheduling and execution.

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
        durations (Dict[str, float]): Durations for each job.
        These are used to determine the maximum burst time for each job.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, List[str], Dict[str, float]]: A tuple containing:
            - DataFrame with job performance metrics.
            - List of job names.
            - Dictionary mapping job names to their release times.
    """
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    log = []
    release_times = {}
    job_names = []
    all_nodes = {n for path in job_network_paths.values() for n in path}
    resources = {n: simpy.Resource(env, capacity=1) for n in all_nodes}

    for inst_name, sched_start, _ in schedule:
        job_regex = BASE_JOB_RE.match(inst_name)
        base = job_regex.group(1) if job_regex else inst_name
        idx_str = inst_name[len(base):]
        idx = int(idx_str) if idx_str.isdigit() else 0

        rel = job_rel_times.get(base, 0.0) + idx * job_periods.get(
            base, 0.0
        )
        release_times[inst_name] = rel
        job_names.append(inst_name)

        Job(
            env=env,
            name=inst_name,
            arrival=rel,
            start=sched_start,
            route=job_network_paths[base],
            resources=resources,
            p_gen=job_parameters[base]["p_gen"],
            epr_pairs=int(job_parameters[base]["e_pairs"]),
            slot_duration=job_parameters[base]["slot_duration"],
            rng=rng,
            log=log,
            durations=durations,
        )

    env.run()
    return pd.DataFrame(log), job_names, release_times
