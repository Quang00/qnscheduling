"""
Simulation Of Probabilistic Job Scheduling
-------------------------------------------
This module implements a simulation of quantum network applications scheduling
using the SimPy library. It defines a `Job` class that represents a
non-preemptive job with probabilistic success in generating EPR pairs. The
function, `simulate_periodicity`, orchestrates the scheduling and execution of
these jobs based on a provided schedule.
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
        done_event: simpy.events.Event | None = None,
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
            delay_map (Dict[tuple, float], optional): Dictionary of delays
            between nodes. Defaults to None, which means no additional delays.
            deadline (float, optional): Deadline time for the job. Defaults to
            None, which means no deadline.
            done_event (simpy.events.Event, optional): Event to signal job
            completion.
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
        self.done_event = done_event
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

        if self.done_event is not None and not self.done_event.triggered:
            self.done_event.succeed(result)


def simulate_periodicity(
    schedule: List[Tuple[str, float, float]],
    job_parameters: Dict[str, Dict[str, float]],
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    job_network_paths: Dict[str, List[str]],
    policies: Dict[str, str],
    distances: Dict[tuple, float],
    instances: Dict[str, int],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
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
        instances (Dict[str, int]): Number of instances to run for each base
        job.
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
            app, idx = initial_job.group(1), int(initial_job.group(2))
            next_idx[app] = max(next_idx.get(app, 0), idx + 1)
        retry_map[name] = name

    all_nodes = {n for path in job_network_paths.values() for n in path}
    resources = {n: simpy.PriorityResource(env, capacity=1) for n in all_nodes}
    delay_map = edges_delay(distances)
    apps = set(job_parameters.keys())

    for app in apps:
        instances.setdefault(app, int(job_parameters[app].get("instances", 1)))
    done_events = {app: env.event() for app in apps}
    success_counts = {app: 0 for app in apps}

    def watch_job(event: simpy.Event, app: str):
        """Watch for job completion events.

        Args:
            event (simpy.Event): Event to monitor.
            app (str): Application name.
        """
        result = yield event
        if result.get("status") == "completed":
            success_counts[app] += 1
            if (not done_events[app].triggered) and success_counts[
                app
            ] >= instances[app]:
                done_events[app].succeed(True)

    def launch_job(
        inst_name: str,
        app: str,
        arrival: float,
        start: float,
        deadline: float
    ):
        """Launch a job in the simulation.

        Args:
            inst_name (str): Instance name of the job.
            app (str): Application name.
            arrival (float): Arrival time of the job.
            start (float): Start time of the job.
            deadline (float): Deadline of the job.
        """
        params = job_parameters[app]
        done_event = env.event()
        Job(
            env=env,
            name=inst_name,
            arrival=arrival,
            start=start,
            route=job_network_paths[app],
            resources=resources,
            p_gen=params["p_gen"],
            epr_pairs=int(params["epr_pairs"]),
            slot_duration=params["slot_duration"],
            rng=rng,
            log=log,
            policy=policies[app],
            delay_map=delay_map,
            deadline=deadline,
            done_event=done_event,
        )
        env.process(watch_job(done_event, app))

    # Initialize jobs from the initial schedule
    for inst_name, sched_start, deadline in schedule:
        m = INIT_JOB_RE.match(inst_name)
        app, idx = (m.group(1), int(m.group(2))) if m else (inst_name, 0)
        period = float(job_periods.get(app, 0.0))
        rel0 = float(job_rel_times.get(app, 0.0))
        rel = rel0 + idx * period
        release_times[inst_name] = rel
        job_names.append(inst_name)
        launch_job(inst_name, app, rel, sched_start, deadline)

    def app_spawner(app: str):
        """Spawn new job instances for a given application.

        Args:
            app (str): Application name.
        """
        period = float(job_periods.get(app, 0.0))
        rel0 = float(job_rel_times.get(app, 0.0))
        idx = int(next_idx.get(app, 0))

        while True:
            if done_events[app].triggered:
                break

            rel = rel0 + idx * period

            if env.now < rel:
                yield env.timeout(rel - env.now)
                if done_events[app].triggered:
                    break

            if done_events[app].triggered:
                break

            start_time = rel
            deadline = rel + period

            inst_name = f"{app}{idx}"
            job_names.append(inst_name)
            release_times[inst_name] = rel

            launch_job(inst_name, app, rel, start_time, deadline)
            idx += 1

    # Start spawners for each application
    for app in apps:
        env.process(app_spawner(app))

    env.run(until=simpy.AllOf(env, list(done_events.values())))

    return pd.DataFrame(log), job_names, release_times
