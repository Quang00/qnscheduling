"""
Packet Generation Attempt
-------------------------
This module provides functions to calculate the end-to-end probability of
generating EPR pairs in a quantum network, check if the probability of
generating a certain number of EPR pairs exceeds a given threshold, and
compute the duration of a Packet Generation Attempt (PGA) based on these
probabilities.
"""

from functools import lru_cache
from math import comb

from scipy.stats import binom


def probability_e2e(
    n_swap: int, memory: int = 1, p_gen: float = 0.001, p_swap: float = 0.6
) -> float:
    """Calculate the end-to-end probability of generating EPR pairs in a given
    path.

    Args:
        n_swap (int): Number of swaps performed.
        memory (int, optional): Number of independent link-generation trials
        per slot.
        p_gen (float, optional): Probability of generating an EPR pair in a
        single trial.
        p_swap (float, optional): Probability of swapping an EPR pair in a
        single trial.

    Returns:
        float: End-to-end probability of generating EPR pairs.
    """
    p_succ_one_link = 1 - (1 - p_gen) ** (memory)
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


@lru_cache(maxsize=None)
def expected_bsm_slots(n_links: int, p_link: float, window: int) -> float:
    """Expected number of slots until an end-to-end BSM is performed.

    Args:
        n_links (int): Number of links that must hold a live pair.
        p_link (float): Per-slot success probability of a single link.
        window (int): Coherence window in slots.

    Returns:
        float: Expected number of slots until the BSM fires.
    """
    r = 1.0 - p_link
    e_max = sum(
        (-1) ** (j + 1) * comb(n_links, j) / (1.0 - r**j)
        for j in range(1, n_links + 1)
    )
    p_window = (
        (1.0 - r**window) ** n_links - (r - r**window) ** n_links
    ) / (1.0 - r**n_links)
    if p_window <= 0.0:
        return float("inf")
    return e_max / p_window


def _pmf(j: int, m: int, p: float) -> float:
    if j < 0 or j > m:
        return 0.0
    return float(binom.pmf(j, m, p))


def _cdf(r: int, s: int, p: float) -> float:
    if r < 0:
        return 0.0
    return float(binom.cdf(min(r, s), s, p))


def naus_probability(k: int, window: int, n: int, p: float) -> float:
    """Probability that some ``window`` consecutive slots contain at least
    ``k`` successes, within ``n`` Bernoulli(``p``) slots. Scan-statistic
    approximation of Naus (1982), as used for PGA lengths by Beauchamp et al.

    Args:
        k (int): Cluster size (number of successes required in a window).
        window (int): Scan window length in slots.
        n (int): Total number of slots.
        p (float): Per-slot success probability.

    Returns:
        float: Probability of observing the cluster within ``n`` slots.
    """
    if k == 1:
        return 1.0 - (1.0 - p) ** n
    if n <= window:
        return float(binom.sf(k - 1, n, p))

    m = window
    bk = _pmf(k, m, p)
    q2 = (
        _cdf(k - 1, m, p) ** 2
        - (k - 1) * bk * _cdf(k - 2, m, p)
        + m * p * bk * _cdf(k - 3, m - 1, p)
    )
    a1 = 2 * bk * _cdf(k - 1, m, p) * (
        (k - 1) * _cdf(k - 2, m, p) - m * p * _cdf(k - 3, m - 1, p)
    )
    a2 = 0.5 * bk**2 * (
        (k - 1) * (k - 2) * _cdf(k - 3, m, p)
        - 2 * (k - 2) * m * p * _cdf(k - 4, m - 1, p)
        + m * (m - 1) * p**2 * _cdf(k - 5, m - 2, p)
    )
    a3 = sum(
        _pmf(2 * k - r, m, p) * _cdf(r - 1, m, p) ** 2
        for r in range(1, k)
    )
    a4 = sum(
        _pmf(2 * k - r, m, p) * _pmf(r, m, p) * (
            (r - 1) * _cdf(r - 2, m, p) - m * p * _cdf(r - 3, m - 1, p)
        )
        for r in range(2, k)
    )
    q3 = _cdf(k - 1, m, p) ** 3 - a1 + a2 + a3 - a4
    if q2 <= 0.0 or q3 <= 0.0:
        return 1.0
    return 1.0 - q2 * (q3 / q2) ** (n / m - 2.0)


_MAX_SLOTS = 10**12


@lru_cache(maxsize=None)
def duration_pga(
    p_packet: float,
    epr_pairs: int,
    n_swap: int,
    memory: int = 1,
    p_swap: float = 0.6,
    p_gen: float = 0.001,
    time_slot_duration: float = 1e-4,
    coherence: float = 0.020,
) -> float:
    """Calculate the duration of a PGA (Packet Generation Attempt).

    Args:
        p_packet (float): Probability of a packet being generated.
        epr_pairs (int): Number of successes (number of EPR pairs generated).
        n_swap (int): Number of swaps performed.
        memory (int, optional): Number of independent link-generation trials
        per slot.
        p_swap (float, optional): Probability of swapping an EPR pair in a
        single trial.
        p_gen (float, optional): Probability of generating an EPR pair in a
        single trial.
        time_slot_duration (float, optional): Duration of a time slot in
        seconds.
        coherence (float, optional): Coherence time in seconds of a
        generated pair; converted internally to a window of
        ``round(coherence / time_slot_duration)`` slots.

    Returns:
        float: Duration of a PGA in seconds.
    """
    if p_packet == 1.0:
        raise ValueError(
            "p_packet cannot be 1.0, as it would lead to infinite duration."
        )
    window = int(round(coherence / time_slot_duration))
    n_links = n_swap + 1
    p_link = 1.0 - (1.0 - p_gen) ** memory
    if p_link <= 0.0 or window < 1 or epr_pairs > window:
        return float("inf")

    p_delivery = p_swap**n_swap / expected_bsm_slots(n_links, p_link, window)
    if p_delivery <= 0.0:
        return float("inf")

    # exponential search
    low = epr_pairs
    high = low
    while naus_probability(epr_pairs, window, high, p_delivery) < p_packet:
        high *= 2
        if high > _MAX_SLOTS:
            return float("inf")

    while low < high:
        mid = (low + high) // 2
        if naus_probability(epr_pairs, window, mid, p_delivery) >= p_packet:
            high = mid
        else:
            low = mid + 1
    return low * time_slot_duration


def compute_durations(
    paths: dict[str, list[str] | None],
    epr_pairs: dict[str, int],
    p_packet: float,
    memory: int,
    p_swap: float,
    time_slot_duration: float,
    rates: dict[tuple, float],
    coherence: float = 0.020,
) -> dict[str, float]:
    """Compute the duration of each application based on the paths and
    link parameters.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        epr_pairs (dict[str, int]): Entanglement generation pairs for each
        application, indicating how many EPR pairs are to be generated.
        p_packet (float): Probability of a packet being generated.
        memory (int): Number of independent link-generation trials per slot.
        p_swap (float): Probability of swapping an EPR pair in a
        single trial.
        time_slot_duration (float): Duration of a time slot in
        seconds.
        rates (dict[tuple, float]): Per-link p_gen, keyed by sorted-tuple
        edges. The effective p_gen for a route is the minimum across its
        edges.
        coherence (float, optional): Coherence time in seconds of a
        generated pair.

    Returns:
        dict[str, float]: A dictionary mapping each application to its total
        duration, which includes the time taken for probabilistic generation
        of EPR pairs and the latency based on the distance of the path.
    """
    durations = {}
    for app, route in paths.items():
        if not route:
            continue
        length_route = len(route)
        n_swaps = length_route - 2
        if length_route <= 2:
            n_swaps = 0
        effective_p_gen = min(
            rates[(min(u, v), max(u, v))]
            for u, v in zip(route[:-1], route[1:], strict=False)
        )
        pga_time = duration_pga(
            p_packet=p_packet,
            epr_pairs=epr_pairs[app],
            n_swap=n_swaps,
            memory=memory,
            p_swap=p_swap,
            p_gen=effective_p_gen,
            time_slot_duration=time_slot_duration,
            coherence=coherence,
        )
        durations[app] = pga_time
    return durations
