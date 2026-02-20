from scheduling.fidelity import is_e2e_fidelity_feasible


def test_is_e2e_fidelity_feasible():
    fidelities = {("A", "B"): 0.9, ("B", "C"): 0.8}
    path = ["A", "B", "C"]

    assert is_e2e_fidelity_feasible(path, 0.8, fidelities) is False
    assert is_e2e_fidelity_feasible(path, 0.7, fidelities) is True
