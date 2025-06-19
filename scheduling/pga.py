from scipy.stats import binom


def probability_e2e(
    n_swap: int, p_gen: float = 0.001, p_swap: float = 0.95, memory_size: int = 0
) -> float:
    """Calculate the end-to-end probability of generating EPR pairs in a given path.

    Args:
        n_swap (int): Number of swaps performed.
        p_gen (float, optional): Probability of generating an EPR pair in a single trial.
        p_swap (float, optional): Probability of swapping an EPR pair in a single trial.
        memory_size (int, optional): Memory size in number of time units.

    Returns:
        float: End-to-end probability of generating EPR pairs.
    """
    p_succ_one_link = 1 - (1 - p_gen) ** (memory_size + 1)
    p_succ_all_links = p_succ_one_link ** (n_swap + 1)
    p_bsms = p_swap**n_swap

    return p_succ_all_links * p_bsms


def exceeds_p_packet(n: int, k: int, p_e2e: float, p_packet: float) -> bool:
    """Check if the probability of generating at least k EPR pairs in n trials is greater than or equal to p_packet.

    Args:
        n (int): Number of trials.
        k (int): Number of successes (number of EPR pairs generated).
        p_e2e (float): Probability of generating an EPR pair end-to-end in a single trial.
        p_packet (float): Probability of a packet being generated.

    Returns:
        bool: True if the probability of generating at least k EPR pairs in n trials is greater than or equal to p_packet.
    """
    return binom.sf(k - 1, n, p_e2e) >= p_packet


def duration_pga(
    p_packet: float,
    time_slot: float,
    k: int,
    n_swap: int,
    p_swap: float = 0.95,
    p_gen: float = 0.001,
    memory_duration: int = 0,
) -> float:
    """Calculate the duration of a PGA (Packet Generation Attempt).

    Args:
        p_packet (float): Probability of a packet being generated.
        time_slot (float): Duration of a time slot in microseconds.
        k (int): Number of successes (number of EPR pairs generated).
        n_swap (int): Number of swaps performed.
        p_swap (float, optional): Probability of swapping an EPR pair in a single trial.
        p_gen (float, optional): Probability of generating an EPR pair in a single trial.
        memory_duration (int, optional): Memory duration in number of time units.

    Returns:
        float: Duration of a PGA in microseconds.
    """
    p_e2e = probability_e2e(n_swap, p_gen, p_swap, memory_duration)

    # exponential search
    low = k
    high = low
    while not exceeds_p_packet(high, k, p_e2e, p_packet):
        high *= 2

    while low < high:
        mid = (low + high) // 2
        if exceeds_p_packet(mid, k, p_e2e, p_packet):
            high = mid
        else:
            low = mid + 1
    return low * time_slot
