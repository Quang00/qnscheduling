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
        end: float,
        route: List[str],
        resources: Dict[str, simpy.PriorityResource],
        p_gen: float,
        epr_pairs: int,
        slot_duration: float,
        rng: np.random.Generator,
        log: List[Dict[str, Any]],
        policy: str,
        delay_map: Dict[tuple, float] | None = None,
        deadline: float | None = None,
        done_event: simpy.events.Event | None = None,
    ) -> None:
        """Probabilistic non-preemptive job.

        Args:
            env (simpy.Environment): Environment in which the job runs.
            name (str): Job name for identification.
            arrival (float): Arrival time of the job in the simulation.
            start (float): Start time of the job in the simulation.
            end (float): End time of the job in the simulation.
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
        self.arrival = float(arrival)
        self.start = float(start)
        self.end = float(end)
        self.route = route
        self.resources = resources
        self.p_gen = float(p_gen)
        self.epr_pairs = int(epr_pairs)
        self.slot_duration = float(slot_duration)
        self.rng = rng
        self.log = log
        self.policy = policy
        self.delay_map = delay_map or {}
        self.deadline = None if deadline is None else float(deadline)
        self.done_event = done_event
        env.process(self.run())

    def run(self):
        # Wait until arrival then exact scheduled start
        if self.env.now < self.arrival:
            yield self.env.timeout(self.arrival - self.env.now)
        if self.env.now < self.start:
            yield self.env.timeout(self.start - self.env.now)

        delays = []
        prev = None
        for node in self.route:
            if prev is not None:
                delays.append(self.delay_map.get((prev, node), 0.0))
            prev = node

        reqs = [self.resources[node].request() for node in self.route]
        t_request = self.env.now
        yield simpy.AllOf(self.env, reqs)
        requests = list(zip(self.route, reqs))

        # If we had to wait, the schedule is violated
        if self.env.now > t_request:
            status = "conflict"
            completion = self.env.now
        else:
            # Work strictly within [start, end)
            t0 = self.start
            t_budget = max(0.0, self.end - self.start)
            successes = 0
            status = "failed"

            if t_budget > 0:
                if self.policy == "deadline":
                    while successes < self.epr_pairs:
                        for delay in delays:
                            if delay <= 0:
                                continue
                            if (self.env.now - t0 + delay) > t_budget:
                                break
                            yield self.env.timeout(delay)
                        elapsed = self.env.now - t0
                        if elapsed >= t_budget:
                            break
                        wait = min(self.slot_duration, t_budget - elapsed)
                        # Non-preemptive: if we can't fit a full slot, stop
                        if wait < self.slot_duration:
                            break
                        yield self.env.timeout(wait)
                        if self.rng.random() < self.p_gen:
                            successes += 1
                    if successes >= self.epr_pairs:
                        status = "completed"

                elif self.policy == "best_effort":
                    while successes < self.epr_pairs:
                        for delay in delays:
                            if delay > 0:
                                if (self.env.now - t0 + delay) > t_budget:
                                    break
                                yield self.env.timeout(delay)
                        if (self.env.now - t0 + self.slot_duration) > t_budget:
                            break
                        yield self.env.timeout(self.slot_duration)
                        if self.rng.random() < self.p_gen:
                            successes += 1
                    if successes >= self.epr_pairs:
                        status = "completed"
            completion = min(self.end, self.env.now)

        # Release resources
        for node, req in requests:
            self.resources[node].release(req)

        # Metrics (relative to scheduled start)
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
            "deadline": self.deadline,  # kept only as metadata
        }
        self.log.append(result)
        if self.done_event is not None and not self.done_event.triggered:
            self.done_event.succeed(result)


def simulate_periodicity(
    schedule: List[Tuple[str, float, float, float]],
    job_parameters: Dict[str, Dict[str, float]],
    job_rel_times: Dict[str, float],
    job_periods: Dict[str, float],
    job_network_paths: Dict[str, List[str]],
    policies: Dict[str, str],
    distances: Dict[tuple, float],
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

    all_nodes = {n for path in job_network_paths.values() for n in path}
    resources = {n: simpy.PriorityResource(env, capacity=1) for n in all_nodes}
    delay_map = edges_delay(distances)

    for inst_name, sched_start, sched_end, sched_deadline in schedule:
        m = INIT_JOB_RE.match(inst_name)
        app, idx = (m.group(1), int(m.group(2))) if m else (inst_name, 0)

        r0 = float(job_rel_times.get(app, 0.0))
        T = float(job_periods.get(app, 0.0))
        arrival = r0 + idx * T

        Job(
            env=env,
            name=inst_name,
            arrival=arrival,
            start=sched_start,
            end=sched_end,
            route=job_network_paths[app],
            resources=resources,
            p_gen=job_parameters[app]["p_gen"],
            epr_pairs=int(job_parameters[app]["epr_pairs"]),
            slot_duration=job_parameters[app]["slot_duration"],
            rng=rng,
            log=log,
            policy=policies[app],
            delay_map=delay_map,
            deadline=sched_deadline,
            done_event=None,
        )

        job_names.append(inst_name)
        release_times[inst_name] = sched_start

    last_end = max((e for _, _, e, _ in schedule), default=0.0)
    env.run(until=last_end + 1e-9)

    return pd.DataFrame(log), job_names, release_times
