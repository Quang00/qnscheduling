import numpy as np

from scheduling.pga import duration_pga, probability_e2e


def test_p_e2e_basic():
    p_gen = 0.2
    memory_lifetime = 1
    n_swap = 0
    p_swap = 0.95
    assert np.isclose(
        probability_e2e(n_swap, memory_lifetime, p_gen, p_swap),
        p_gen,
    )
    assert np.isclose(
        probability_e2e(n_swap + 1, memory_lifetime, p_gen, p_swap),
        (p_gen**2) * p_swap,
    )


def test_p_e2e_monotonicity():
    base_n_swap = 2
    base_memory_lifetime = 3
    base_p_gen = 0.01
    base_p_swap = 0.9
    base = probability_e2e(
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_gen=base_p_gen,
        p_swap=base_p_swap,
    )
    more_memory = probability_e2e(
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime + 2,
        p_gen=base_p_gen,
        p_swap=base_p_swap,
    )
    better_p_gen = probability_e2e(
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_gen=base_p_gen * 2,
        p_swap=base_p_swap,
    )
    better_swap = probability_e2e(
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_gen=base_p_gen,
        p_swap=base_p_swap * 1.05,
    )
    more_swaps = probability_e2e(
        n_swap=base_n_swap + 2,
        memory_lifetime=base_memory_lifetime,
        p_gen=base_p_gen,
        p_swap=base_p_swap,
    )

    assert more_memory > base
    assert better_p_gen > base
    assert better_swap > base
    assert more_swaps < base


def test_duration_pga_monotonicity_wrt_params():
    base_p_packet = 0.6
    base_epr_pairs = 3
    base_n_swap = 2
    base_memory_lifetime = 2
    base_p_swap = 0.9
    base_p_gen = 0.02
    base_time_slot_duration = 1e-5

    base = duration_pga(
        p_packet=base_p_packet,
        epr_pairs=base_epr_pairs,
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_swap=base_p_swap,
        p_gen=base_p_gen,
        time_slot_duration=base_time_slot_duration,
    )
    better_gen = duration_pga(
        p_packet=base_p_packet,
        epr_pairs=base_epr_pairs,
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_swap=base_p_swap,
        p_gen=base_p_gen * 2,
        time_slot_duration=base_time_slot_duration,
    )
    better_swap = duration_pga(
        p_packet=base_p_packet,
        epr_pairs=base_epr_pairs,
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_swap=base_p_swap * 1.05,
        p_gen=base_p_gen,
        time_slot_duration=base_time_slot_duration,
    )
    more_memory = duration_pga(
        p_packet=base_p_packet,
        epr_pairs=base_epr_pairs,
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime + 1,
        p_swap=base_p_swap,
        p_gen=base_p_gen,
        time_slot_duration=base_time_slot_duration,
    )
    more_swaps = duration_pga(
        p_packet=base_p_packet,
        epr_pairs=base_epr_pairs,
        n_swap=base_n_swap + 1,
        memory_lifetime=base_memory_lifetime,
        p_swap=base_p_swap,
        p_gen=base_p_gen,
        time_slot_duration=base_time_slot_duration,
    )
    more_epr_pairs = duration_pga(
        p_packet=base_p_packet,
        epr_pairs=base_epr_pairs + 1,
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_swap=base_p_swap,
        p_gen=base_p_gen,
        time_slot_duration=base_time_slot_duration,
    )

    higher_ppacket = duration_pga(
        p_packet=base_p_packet + 0.1,
        epr_pairs=base_epr_pairs,
        n_swap=base_n_swap,
        memory_lifetime=base_memory_lifetime,
        p_swap=base_p_swap,
        p_gen=base_p_gen,
        time_slot_duration=base_time_slot_duration,
    )

    assert better_gen < base
    assert better_swap < base
    assert more_memory < base

    assert more_swaps > base
    assert more_epr_pairs > base
    assert higher_ppacket > base
