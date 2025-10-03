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
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import simpy

from scheduling.pga import probability_e2e
from utils.helper import edges_delay

INIT_JOB_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


@dataclass
class HalfQubit:
    qubit: str
    creation_time: float


class Memory:
    def __init__(self, capacity: int, coherence_time_s: float):
        self.capacity = int(capacity)
        self.coherence_time_s = float(coherence_time_s)
        self.total: Deque[HalfQubit] = deque()
        self.qubit: Dict[str, Deque[HalfQubit]] = defaultdict(deque)

    def purge_expired(self, now: float) -> None:
        cohenrence_time = now - self.coherence_time_s
        while self.total and self.total[0].creation_time < cohenrence_time:
            exprired = self.total.popleft()
            qubit = self.qubit[exprired.qubit]
            while qubit and qubit[0].creation_time < cohenrence_time:
                qubit.pop()
            if not qubit:
                self.qubit.pop(exprired.qubit, None)

    def store(self, now: float, partner: str) -> bool:
        self.purge_expired(now)
        if self.capacity - len(self.total) <= 0:
            return False
        h = HalfQubit(qubit=partner, creation_time=now)
        self.total.append(h)
        self.qubit[partner].append(h)
        return True

    def discard(self, partner: str) -> Optional[HalfQubit]:
        qubit = self.qubit.get(partner)
        if not qubit:
            return None
        half = qubit.pop()
        for i, t in enumerate(self.total):
            if (
                t.qubit == half.qubit
                and t.creation_time == half.creation_time
            ):
                del self.total[i]
                break
        if not qubit:
            self.qubit.pop(partner, None)
        return half


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
        p_swap: float,
        memory_lifetime: int,
        memory_capacity: int,
        delay_map: Dict[Tuple[str, str], float] | None = None,
        deadline: float | None = None,
        done_event: simpy.events.Event | None = None,
        memories: Dict[str, Memory] | None = None,
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

        if memories is None:
            coherence_time_s = (memory_lifetime + 1) * self.slot_duration
            self.memories = {
                n: Memory(memory_capacity, coherence_time_s) for n in self.route
            }
        else:
            self.memories = memories

        self.n_swap = max(0, len(self.route) - 2)
        self.p_e2e = probability_e2e(
            self.n_swap, int(memory_lifetime), self.p_gen, float(p_swap)
        )

        env.process(self.run())

    def _purge_all(self):
        cur = self.env.now
        for n in self.route:
            self.memories[n].purge_expired(cur)

    def _store_e2e(self) -> bool:
        src_node, dst_node = self.route[0], self.route[-1]
        cur = self.env.now
        src_mem, dst_mem = self.memories[src_node], self.memories[dst_node]
        is_src_mem_stored = src_mem.store(cur, partner=dst_node)
        is_dst_mem_stored = dst_mem.store(cur, partner=src_node)
        if is_src_mem_stored and is_dst_mem_stored:
            return True
        if is_src_mem_stored:
            src_mem.discard(dst_node)
        if is_dst_mem_stored:
            dst_mem.discard(src_node)
        return False

    def run(self):
        # wait for arrival and scheduled start
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

        if self.env.now > t_request:
            status = "conflict"
            completion = self.env.now
        else:
            t0 = self.start
            t_budget = max(0.0, self.end - self.start)
            status = "failed"
            successes = 0

            if t_budget > 0 and self.policy == "deadline":
                while successes < self.epr_pairs:
                    self._purge_all()

                    elapsed = self.env.now - t0
                    if elapsed >= t_budget:
                        break

                    remaining = t_budget - elapsed
                    if remaining < self.slot_duration:
                        break

                    for d in delays:
                        if d > 0:
                            if (self.env.now - t0 + d) > t_budget:
                                break
                            yield self.env.timeout(d)

                    yield self.env.timeout(self.slot_duration)

                    if self.rng.random() < self.p_e2e:
                        if self._store_e2e():
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

        # release resources
        for node, req in requests:
            self.resources[node].release(req)

        # metrics
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
            p_swap=job_parameters[app]["p_swap"],
            memory_lifetime=job_parameters[app]["memory_lifetime"],
            memory_capacity=job_parameters[app]["memory_lifetime"],
        )

        job_names.append(inst_name)
        release_times[inst_name] = sched_start

    last_end = max((e for _, _, e, _ in schedule), default=0.0)
    env.run(until=last_end + 1e-9)

    return pd.DataFrame(log), job_names, release_times
