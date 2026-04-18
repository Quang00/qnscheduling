import numpy as np
import pytest

from utils.graph_generator import fat_tree, generate_waxman_graph

MAX_AVG_DEGREE = 6.0
MAX_HOPS = 10


@pytest.fixture(scope="module")
def waxman_result():
    rng = np.random.default_rng(42)
    return generate_waxman_graph(
        n=20,
        alpha=0.5,
        beta=0.5,
        max_avg_degree=MAX_AVG_DEGREE,
        max_hops=MAX_HOPS,
        rng=rng,
    )


@pytest.fixture(scope="module")
def fat_tree_k4():
    return fat_tree(k=4)


class TestGenerateWaxmanGraph:
    def test_waxman_basic(self, waxman_result):
        nodes, edges, fidelities, avg_deg, diameter, end_nodes = waxman_result
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        assert isinstance(fidelities, dict)
        assert isinstance(avg_deg, float)
        assert isinstance(diameter, float)
        assert isinstance(end_nodes, list)

    def test_nodes_and_edges_nonempty(self, waxman_result):
        nodes, edges, _, _, _, _ = waxman_result
        assert len(nodes) == 20
        assert len(edges) > 0

    def test_fidelity_keys_match_edges(self, waxman_result):
        _, edges, fidelities, _, _, _ = waxman_result
        assert set(fidelities.keys()) == set(edges)

    def test_fidelity_values_in_range(self, waxman_result):
        _, _, fidelities, _, _, _ = waxman_result
        for f in fidelities.values():
            assert 0.0 <= f <= 1.0

    def test_avg_degree_within_limit(self, waxman_result):
        _, _, _, avg_deg, _, _ = waxman_result
        assert avg_deg <= MAX_AVG_DEGREE

    def test_diameter_within_limit(self, waxman_result):
        _, _, _, _, diameter, _ = waxman_result
        assert diameter <= MAX_HOPS

    def test_impossible_constraints_returns_empty(self):
        rng = np.random.default_rng(42)
        nodes, edges, fidelities, avg_deg, diameter, end_nodes = (
            generate_waxman_graph(
                n=20, rng=rng, max_retries=5, max_avg_degree=0.1, max_hops=1
            )
        )
        assert nodes == []
        assert edges == []
        assert fidelities == {}
        assert avg_deg == 0.0
        assert end_nodes == []


@pytest.mark.parametrize(
    "k, expected_nodes, expected_qpus",
    [
        (4, 36, 16),
        (6, 99, 54),
    ],
)
def test_fat_tree_counts(k, expected_nodes, expected_qpus):
    nodes, _, _, qpus, _ = fat_tree(k=k)
    assert len(nodes) == expected_nodes
    assert len(qpus) == expected_qpus


class TestFatTree:
    def test_returns_valid_tuple(self, fat_tree_k4):
        nodes, edges, fidelities, qpus, diameter = fat_tree_k4
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        assert isinstance(fidelities, dict)
        assert isinstance(qpus, list)
        assert isinstance(diameter, float)

    def test_f_above_f_min(self):
        f_min = 0.8
        _, _, fidelities, _, _ = fat_tree(k=4, F_min=f_min)
        assert all(f >= f_min for f in fidelities.values())

    def test_diameter_k4(self, fat_tree_k4):
        _, _, _, _, diameter = fat_tree_k4
        assert diameter == 6.0

    def test_dist_fidelity(self):
        _, _, fid_close, _, _ = fat_tree(k=4, qpu_edge_dist=0.1)
        _, _, fid_far, _, _ = fat_tree(k=4, qpu_edge_dist=10.0)
        avg_close = sum(fid_close.values()) / len(fid_close)
        avg_far = sum(fid_far.values()) / len(fid_far)
        assert avg_close > avg_far
