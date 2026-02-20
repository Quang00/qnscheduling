from typing import Dict, List, Tuple


def is_e2e_fidelity_feasible(
    path: List[str],
    min_fidelity: float,
    fidelities: Dict[Tuple[str, str], float],
) -> bool:
    """Check if a given path meets the minimum fidelity threshold. The E2E
    fidelity of a path is computed as the product of the Werner parameters
    of the edges along the path.
    """
    required_werner = (4 * float(min_fidelity) - 1) / 3
    e2e_werner = 1.0

    for u, v in zip(path[:-1], path[1:], strict=False):
        key = (u, v) if (u, v) in fidelities else (v, u)
        edge_fidelity = float(fidelities[key])
        e2e_werner *= (4 * edge_fidelity - 1) / 3
    return e2e_werner >= required_werner
