"""
Simulation Of Probabilistic Job Scheduling
-------------------------------------------
This module implements a simulation of quantum network applications scheduling
using the SimPy library. It defines a `Job` class to represent jobs in the
simulation, a dispatcher function to handle job retries, and a `simulate`
function to run the simulation. The simulation tracks job performance metrics
and allows for retrying failed jobs based on specified policies. The jobs are
scheduled based on their arrival times and the resources available in the
network. The simulation can handle both best-effort and deadline-based job
scheduling policies.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import simpy

APP_RE = re.compile(r"^([A-Za-z]+)")
INIT_JOB_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
RETRY_JOB_RE = re.compile(r"^([A-Za-z]+\d+)_(\d+)$")


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
        durations: Dict[str, float],
        policy: str,
        event_store: simpy.Store,
    ) -> None:
        """Job class to represent a probabilistic job in the simulation.

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
            event_store (simpy.Store): Store to hold job events for
            post-processing.
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
        self.policy = policy
        self.event_store = event_store
        env.process(self.run())

    def run(self):
        # Wait for arrival and scheduled start
        yield self.env.timeout(max(0.0, self.arrival - self.env.now))
        yield self.env.timeout(max(0.0, self.start - self.env.now))

        m_retry = RETRY_JOB_RE.match(self.name)
        if m_retry:
            _, retry_count = m_retry.groups()
            idx = int(retry_count)
        else:
            m_orig = INIT_JOB_RE.match(self.name)
            idx = int(m_orig.group(2)) if m_orig else 0

        requests = []
        for node in self.route:
            req = self.resources[node].request(priority=idx)
            yield req
            requests.append((node, req))

        t0 = self.env.now
        successes = 0
        app_name = APP_RE.match(self.name).group(1)
        max_burst = self.durations.get(app_name, float("inf"))

        if self.policy == "deadline":
            status = "failed"
            while successes < self.epr_pairs:
                elapsed = self.env.now - t0
                if elapsed >= max_burst:
                    break
                wait = min(self.slot_duration, max_burst - elapsed)
                yield self.env.timeout(wait)
                if wait < self.slot_duration:
                    break
                if self.rng.random() < self.p_gen:
                    successes += 1
            else:
                status = "completed"
        elif self.policy == "best_effort":
            while successes < self.epr_pairs:
                yield self.env.timeout(self.slot_duration)
                if self.rng.random() < self.p_gen:
                    successes += 1
            status = "completed"

        completion = self.env.now

        for node, req in requests:
            self.resources[node].release(req)

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
        }
        self.log.append(result)
        self.event_store.put(result)


def dispatcher(
    env: simpy.Environment,
    store: simpy.Store,
    resources: Dict[str, simpy.PriorityResource],
    job_parameters: Dict[str, Dict[str, float]],
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    job_network_paths: Dict[str, List[str]],
    durations: Dict[str, float],
    policies: Dict[str, str],
    log: List[Dict[str, Any]],
    job_names: List[str],
    release_times: Dict[str, float],
    retry_policy: str,
    max_retries: int,
    rng: np.random.Generator,
    retry_map: Dict[str, str],
    latency: float = 0.0,
):
    """Handle job retries in the dispatcher.

    Args:
        env (simpy.Environment): Environment for the simulation.
        store (simpy.Store): Store to hold job events.
        resources (Dict[str, simpy.PriorityResource]): Resources available for
        job execution.
        job_parameters (Dict[str, Dict[str, float]]): Parameters for each job.
        job_rel_times (Dict[str, float]): Relative release times for each job.
        job_periods (Dict[str, float]): Periods for each job.
        job_network_paths (Dict[str, List[str]]): Network paths for each job.
        durations (Dict[str, float]): Durations for each job.
        policies (Dict[str, str]): Scheduling policies for each job.
        log (List[Dict[str, Any]]): Log of job events.
        job_names (List[str]): List of job names.
        release_times (Dict[str, float]): Release times for each job.
        retry_policy (str): Retry policy for failed jobs.
        max_retries (int): Maximum number of retries for failed jobs.
        rng (np.random.Generator): Random number generator for job retries.
        retry_map (Dict[str, str]): Mapping of retry job names to original job.
        latency (float, optional): Latency apply before retrying failed jobs.
    """
    retry_counts = {}

    while True:
        result = yield store.get()
        if result["status"] != "failed":
            continue

        yield env.timeout(latency)

        job_name = result["job"]
        initial_job = retry_map[job_name]
        count = retry_counts.get(initial_job, 0)

        if retry_policy == "limited" and count >= max_retries:
            continue

        retry_counts[initial_job] = count + 1
        retry_name = f"{initial_job}_{count}"
        retry_map[retry_name] = initial_job

        orig_job = INIT_JOB_RE.match(initial_job)
        app_name = APP_RE.match(initial_job).group(1)

        initial_idx = int(orig_job.group(2)) if orig_job else 0
        new_idx = initial_idx + (count + 1)

        rel = job_rel_times.get(app_name, 0.0) + new_idx * job_periods.get(
            app_name, 0.0
        )
        release_times[retry_name] = rel
        job_names.append(retry_name)
        params = job_parameters[app_name]

        Job(
            env=env,
            name=retry_name,
            arrival=rel,
            start=rel,
            route=job_network_paths[app_name],
            resources=resources,
            p_gen=params["p_gen"],
            epr_pairs=int(params["epr_pairs"]),
            slot_duration=params["slot_duration"],
            rng=rng,
            log=log,
            durations=durations,
            policy=policies[app_name],
            event_store=store,
        )


def simulate(
    schedule: List[Tuple[str, float, float]],
    job_parameters: Dict[str, Dict[str, float]],
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    job_network_paths: Dict[str, List[str]],
    durations: Dict[str, float],
    policies: Dict[str, str],
    latency: float = 0.0,
    retry_policy: str = "limited",
    max_retries: int = 0,
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
        policies (Dict[str, str]): Scheduling policy for each job.
        This can be "best_effort" or "deadline".
        latency (float, optional): Latency to apply before retrying failed
        jobs.
        retry_policy (str, optional): Retry policy for failed jobs. Can be
        "limited" for a maximum number of retries or "unlimited" for infinite
        retries.
        max_retries (int, optional): Maximum number of retries for failed jobs
        if retry_policy is "limited".
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
    retry_map = {}
    next_idx = {}

    # Initialize retry_map and next_idx based on the initial jobs in schedule
    for name, _, _ in schedule:
        initial_job = INIT_JOB_RE.match(name)
        if initial_job:
            b, idx = initial_job.group(1), int(initial_job.group(2))
            next_idx[b] = max(next_idx.get(b, 0), idx + 1)
        retry_map[name] = name

    store = simpy.Store(env)
    all_nodes = {n for path in job_network_paths.values() for n in path}
    resources = {n: simpy.PriorityResource(env, capacity=1) for n in all_nodes}

    env.process(
        dispatcher(
            env=env,
            store=store,
            resources=resources,
            job_parameters=job_parameters,
            job_rel_times=job_rel_times,
            job_periods=job_periods,
            job_network_paths=job_network_paths,
            durations=durations,
            policies=policies,
            log=log,
            job_names=job_names,
            release_times=release_times,
            retry_policy=retry_policy,
            max_retries=max_retries,
            rng=rng,
            retry_map=retry_map,
            latency=latency,
        )
    )

    # Create initial jobs based on the schedule
    for inst_name, sched_start, _ in schedule:
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
            durations=durations,
            policy=policies[base],
            event_store=store,
        )

    env.run()
    return pd.DataFrame(log), job_names, release_times
