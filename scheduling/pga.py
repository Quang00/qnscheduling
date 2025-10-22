"""
Packet Generation Attempt
-------------------------
This module provides functions to calculate the end-to-end probability of
generating EPR pairs in a quantum network, check if the probability of
generating a certain number of EPR pairs exceeds a given threshold, and
compute the duration of a Packet Generation Attempt (PGA) based on these
probabilities.
"""

from scipy.stats import binom


def probability_e2e(
    n_swap: int,
    memory_lifetime: int = 1,
    p_gen: float = 0.001,
    p_swap: float = 0.95
) -> float:
    """Calculate the end-to-end probability of generating EPR pairs in a given
    path.

    Args:
        n_swap (int): Number of swaps performed.
        memory_lifetime (int, optional): Memory lifetime in number of time
        slot units.
        p_gen (float, optional): Probability of generating an EPR pair in a
        single trial.
        p_swap (float, optional): Probability of swapping an EPR pair in a
        single trial.

    Returns:
        float: End-to-end probability of generating EPR pairs.
    """
    if memory_lifetime < 0:
        raise ValueError("memory_lifetime must be non-negative")
    p_succ_one_link = 1 - (1 - p_gen) ** (memory_lifetime)
    p_succ_all_links = p_succ_one_link ** (n_swap + 1)
    p_bsms = p_swap**n_swap

    return p_succ_all_links * p_bsms


def exceeds_p_packet(n: int, k: int, p_e2e: float, p_packet: float) -> bool:
    """Check if the probability of generating at least k EPR pairs in n trials
    is greater than or equal to p_packet.

    Args:
        n (int): Number of trials.
        k (int): Number of successes (number of EPR pairs generated).
        p_e2e (float): Probability of generating an EPR pair end-to-end in a
        single trial.
        p_packet (float): Probability of a packet being generated.

    Returns:
        bool: True if the probability of generating at least k EPR pairs in n
        trials is greater than or equal to p_packet.
    """
    return binom.sf(k - 1, float(n), p_e2e) >= p_packet


def duration_pga(
    p_packet: float,
    epr_pairs: int,
    n_swap: int,
    memory_lifetime: int = 1,
    p_swap: float = 0.95,
    p_gen: float = 0.001,
    time_slot_duration: float = 1e-4,
) -> float:
    """Calculate the duration of a PGA (Packet Generation Attempt).

    Args:
        p_packet (float): Probability of a packet being generated.
        epr_pairs (int): Number of successes (number of EPR pairs generated).
        n_swap (int): Number of swaps performed.
        memory_lifetime (int, optional): Memory lifetime in number of time
        slot units.
        p_swap (float, optional): Probability of swapping an EPR pair in a
        single trial.
        p_gen (float, optional): Probability of generating an EPR pair in a
        single trial.
        time_slot_duration (float, optional): Duration of a time slot in
        seconds.

    Returns:
        float: Duration of a PGA in seconds.
    """
    p_e2e = probability_e2e(n_swap, memory_lifetime, p_gen, p_swap)

    # exponential search
    low = epr_pairs
    high = low
    while not exceeds_p_packet(high, epr_pairs, p_e2e, p_packet):
        high *= 2

    while low < high:
        mid = (low + high) // 2
        if exceeds_p_packet(mid, epr_pairs, p_e2e, p_packet):
            high = mid
        else:
            low = mid + 1
    return low * time_slot_duration


def compute_durations(
    paths: dict[str, list[str]],
    epr_pairs: dict[str, int],
    p_packet: float,
    memory_lifetime: int,
    p_swap: float,
    p_gen: float,
    time_slot_duration: float,
) -> dict[str, float]:
    """Compute the duration of each application based on the paths and
    link parameters.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        epr_pairs (dict[str, int]): Entanglement generation pairs for each
        application, indicating how many EPR pairs are to be generated.
        p_packet (float): Probability of a packet being generated.
        memory_lifetime (int): Memory lifetime in number of time
        slot units.
        p_swap (float): Probability of swapping an EPR pair in a
        single trial.
        p_gen (float): Probability of generating an EPR pair in a
        single trial.
        time_slot_duration (float): Duration of a time slot in
        seconds.

    Returns:
        dict[str, float]: A dictionary mapping each application to its total
        duration, which includes the time taken for probabilistic generation
        of EPR pairs and the latency based on the distance of the path.
    """
    durations = {}
    for app, route in paths.items():
        length_route = len(route)
        n_swaps = length_route - 2
        if length_route <= 2:
            n_swaps = 0
        pga_time = duration_pga(
            p_packet=p_packet,
            epr_pairs=epr_pairs[app],
            n_swap=n_swaps,
            memory_lifetime=memory_lifetime,
            p_swap=p_swap,
            p_gen=p_gen,
            time_slot_duration=time_slot_duration,
        )
        durations[app] = pga_time
    return durations
