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

import numpy as np
import pandas as pd
import simpy


def probabilistic_job(
    env: simpy.Environment,
    job_name: str,
    arrival_time: float,
    job_start_time: float,
    job_route: list[str],
    node_resources: dict[str, simpy.Resource],
    p_gen: float,
    epr_pairs: int,
    slot_duration: float,
    job_completion_log: list[dict[str, float]],
    rng: np.random.Generator,
):
    """Probabilistic job simulation.

    Args:
        env (simpy.Environment): Environment for the simulation.
        job_name (str): Job name for identification.
        arrival_time (float): Arrival time of the job in the simulation.
        job_start_time (float): Start time of the job in the simulation.
        job_route (list[str]): List of nodes in the path for the job.
        node_resources (dict[str, simpy.Resource]): Resources for each node in
        the path.
        p_gen (float): Probability of generating an EPR pair in a single trial.
        epr_pairs (int): Number of successes (number of EPR pairs
        generated).
        slot_duration (float): Duration of each time slot in the simulation.
        completion_log (list[dict[str, float]]): List to log job completion
        statistic.
        rng (np.random.Generator): Random number generator for probabilistic
        outcomes.
    """
    yield env.timeout(max(0, arrival_time - env.now))
    yield env.timeout(max(0, job_start_time - env.now))

    requests = []
    for node in job_route:
        request = node_resources[node].request()
        requests.append(request)
        yield request

    start_time = env.now
    successes = 0
    while successes < epr_pairs:
        yield env.timeout(slot_duration)
        val = rng.random()
        if val < p_gen:
            successes += 1
    completion_time = env.now

    for request in requests:
        node_resources[job_route[requests.index(request)]].release(request)

    burst_time = completion_time - start_time
    turnaround_time = completion_time - arrival_time
    waiting_time = turnaround_time - burst_time

    job_completion_log.append(
        {
            "job": job_name,
            "arrival_time": arrival_time,
            "start_time": start_time,
            "burst_time": burst_time,
            "completion_time": completion_time,
            "turnaround_time": turnaround_time,
            "waiting_time": waiting_time,
        }
    )


def simulate(
    schedule: list[tuple[str, float, float]],
    job_parameters: dict[str, dict[str, float]],
    job_release_times: dict[str, float],
    job_periods: dict[str, float],
    job_instance_freq: dict[str, int],
    job_network_paths: dict[str, list[str]],
    seed: int = 42,
):
    """Simulate the job scheduling and execution.

    Args:
        schedule (list[tuple[str, float, float]]): List of tuples where each
        contains the job name, start time, and end time of the scheduled job.
        job_parameters (dict[str, dict[str, float]]): Parameters for each job,
        including the probability of generating an EPR pair, number of
        successes, and slot duration.
        job_release_times (dict[str, float]): Relative release times for each
        job.
        job_periods (dict[str, float]): Periods for each job, indicating the
        time interval between successive releases of the job.
        job_instance_freq (dict[str, int]): Repetitions for each job,
        indicating how many times the job should be executed.
        job_network_paths (dict[str, list[str]]): List of nodes for each job's
        path in the network.
        seed (int): Random seed for reproducibility.
    """
    env = simpy.Environment()
    all_nodes = {n for p in job_network_paths.values() for n in p}
    node_resources = {n: simpy.Resource(env, capacity=1) for n in all_nodes}

    records, job_names = [], []
    total_jobs = sum(job_instance_freq.get(name, 1) for name, _, _ in schedule)

    rng = np.random.default_rng(seed)
    for name, start, _ in schedule:
        reps = job_instance_freq.get(name, 1)
        for i in range(reps):
            rel_i = job_release_times.get(name, 0.0) + i * job_periods[name]
            start_i = start + i * job_periods[name]
            inst = f"{name}_{i}"
            job_names.append(inst)
            env.process(
                probabilistic_job(
                    env,
                    inst,
                    rel_i,
                    start_i,
                    job_network_paths[name],
                    node_resources,
                    job_parameters[name]["p_gen"],
                    job_parameters[name]["k"],
                    job_parameters[name]["slot_duration"],
                    records,
                    rng,
                )
            )

    env.run()
    results = pd.DataFrame(records)
    return results, job_names, job_release_times, total_jobs
