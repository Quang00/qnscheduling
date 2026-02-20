import pytest

from scheduling.fidelity import (
    fidelity_bounds_and_paths,
    is_e2e_fidelity_feasible,
    werner_adj_list,
)


def test_is_e2e_fidelity_feasible():
    fidelities = {("A", "B"): 0.9, ("B", "C"): 0.8}
    path = ["A", "B", "C"]

    assert is_e2e_fidelity_feasible(path, 0.8, fidelities) is False
    assert is_e2e_fidelity_feasible(path, 0.7, fidelities) is True


def test_werner_adj_basic():
    fidelities = {
        ("A", "B"): 0.75,  # w(AB) = (4 * 0.75 - 1) / 3 = 2/3
    }
    adj = werner_adj_list(fidelities)

    assert ("B", pytest.approx(2 / 3, rel=1e-12, abs=0.0)) in adj["A"]
    assert ("A", pytest.approx(2 / 3, rel=1e-12, abs=0.0)) in adj["B"]


def test_fidelity_bounds_and_paths_basic_graph():
    # A --(1.0)--> B --(0.75)--> C --(0.5)--> D
    #               \----------(0.9)--------> D
    # w(AB) = 1, w(BC) = 2/3, w(CD) = 1/3, w(BD) = 13/15
    fidelities = {
        ("A", "B"): 1.0,
        ("B", "C"): 0.75,
        ("C", "D"): 0.5,
        ("B", "D"): 0.9,
    }
    end_nodes = ["D", "A", "C"]
    bounds, paths = fidelity_bounds_and_paths(end_nodes, fidelities, K=8)

    # A -> D:
    #  - A-B-C-D: w(AB) * w(BC) * w(CD) = 1 * 2/3 * 1/3 = 2/9 => F = 5/12
    #  - A-B-D: w(AB) * w(BD) = 1 * 13/15 => F = (3 * (13/15) + 1 ) / 4 = 0.9
    assert bounds[("A", "D")][0] == pytest.approx(5 / 12, rel=1e-12)
    assert bounds[("A", "D")][1] == pytest.approx(0.9, rel=1e-12)

    # only simple paths should be included
    for _, plist in paths.items():
        for _, p in plist:
            assert len(p) == len(set(p))


def test_fidelity_bounds_and_paths_respects_K():
    fidelities = {
        ("A", "B"): 1.0,
        ("B", "C"): 0.75,
        ("C", "D"): 0.5,
        ("B", "D"): 0.9,
    }
    end_nodes = ["A", "C", "D"]
    bounds, paths = fidelity_bounds_and_paths(end_nodes, fidelities, K=2)

    # A -> D: only A-B-D (2 hops) fits; A-B-C-D (3 hops) should be excluded
    assert ("A", "D") in paths
    assert len(paths[("A", "D")]) == 1
    assert paths[("A", "D")][0][0] == pytest.approx(0.9, rel=1e-12)
    assert paths[("A", "D")][0][1] == ("A", "B", "D")


def test_fidelity_bounds_and_paths_no_path():
    fidelities = {
        ("A", "B"): 0.9,
    }
    end_nodes = ["A", "C"]
    bounds, paths = fidelity_bounds_and_paths(end_nodes, fidelities, K=8)

    assert bounds == {}
    assert paths == {}
