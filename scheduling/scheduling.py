import math

def length_pga(k: int, p_packet: float, p_gen: float, p_swap: float, n_swap: float, time_slot: float) -> float:
    """
    Calculate the length of a PGA (Packet Generation Attempt).

    Parameters:
    k (int): The number of succcess.
    p_gen (float): The probability of generating an EPR pair in a given time slot.
    p_packet (float): The probability of a packet being successfully generated.
    p_swap (float): The probability of a successful swap operation.
    n_swap (int): The number of swaps attempted.
    time_slot (float): The duration of a time slot in microseconds.

    Returns:
    float: The length of the PGA in microseconds.
    """
    if n_swap < 0:
        raise ValueError("Number of swaps (n_swap) must be non-negative.")
    if n_swap == 0:
        p_e2e = p_gen
    else:
        p_e2e = (p_gen)**(2 * n_swap) * p_swap * n_swap
    
    n = 100
    p_succ = math.comb(n, k) * (p_e2e)**k * (1 - p_e2e)**(n - k)
    print(f"Probability of success: {p_succ}")
    
    return p_succ

length_pga(2, 0.2, 10e-3, 0.5, 0, 1e-6)