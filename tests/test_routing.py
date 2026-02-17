from collections import defaultdict

import networkx as nx
import numpy as np
import pytest

from scheduling.routing import (
    capacity_threshold,
    find_feasible_path,
    hub_aware,
    least_capacity,
    shortest_paths,
    smallest_bottleneck,
    yen_random,
)


@pytest.fixture
def basic_graph():
    return [
        ("Alice", "Bob"),
        ("Bob", "Charlie"),
        ("Charlie", "David"),
        ("Eve", "Frank"),
    ]


def test_shortest_paths_basic_small(basic_graph):
    apps = {
        "A": {"src": "Alice", "dst": "Charlie"},
        "B": {"src": "Bob", "dst": "Charlie"},
    }
    result = shortest_paths(basic_graph, apps)
    assert result == {
        "A": ["Alice", "Bob", "Charlie"],
        "B": ["Bob", "Charlie"],
    }


def test_shortest_paths_basic_long(basic_graph):
    apps = {
        "A": {"src": "Alice", "dst": "David"},
        "B": {"src": "Bob", "dst": "Charlie"},
    }
    result = shortest_paths(basic_graph, apps)
    assert result == {
        "A": ["Alice", "Bob", "Charlie", "David"],
        "B": ["Bob", "Charlie"],
    }


def test_empty_app_requests(basic_graph):
    assert shortest_paths(basic_graph, {}) == {}


def test_empty_edges_and_no_requests_is_ok():
    assert shortest_paths([], {}) == {}


def test_find_feasible_path_basic():
    edges = [
        ("Alice", "Bob"),
        ("Bob", "Charlie"),
        ("Charlie", "David"),
    ]

    fidelities = {
        ("Alice", "Bob"): 0.9,
        ("Bob", "Alice"): 0.9,
        ("Bob", "Charlie"): 0.9,
        ("Charlie", "Bob"): 0.9,
        ("Charlie", "David"): 0.9,
        ("David", "Charlie"): 0.9,
    }

    app_requests = {
        "A": {
            "src": "Alice",
            "dst": "Charlie",
            "min_fidelity": 0.8,
            "epr": 2,
            "period": 10.0,
        },
        "B": {
            "src": "Bob",
            "dst": "David",
            "min_fidelity": 0.8,
            "epr": 1,
            "period": 15.0,
        },
    }

    result = find_feasible_path(
        edges=edges,
        app_requests=app_requests,
        fidelities=fidelities,
        routing_mode="shortest",
    )

    assert result["A"] == ["Alice", "Bob", "Charlie"]
    assert result["B"] == ["Bob", "Charlie", "David"]


def test_yen_random():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("A", "D"),
            ("D", "C"),
        ]
    )

    rng = np.random.default_rng(42)
    path = yen_random(G, "A", "C", L_max=2, rng=rng)

    assert path is not None


def test_yen_random_no_path():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
        ]
    )

    rng = np.random.default_rng(42)
    path = yen_random(G, "A", "D", L_max=1, rng=rng)

    assert path is None


def test_hub_aware():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("B", "D"),
            ("B", "E"),
            ("A", "F"),
            ("F", "C"),
        ]
    )

    path = hub_aware(G, "A", "C", L_max=2)
    assert path == ["A", "F", "C"]


def test_capacity_aware_threshold_exceeded():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
        ]
    )

    req = {
        "epr": 1,
        "period": 1.0,
    }

    cap = defaultdict(float)
    cap[("A", "B")] = 0.95
    cap[("B", "C")] = 0.95

    path, delta = capacity_threshold(
        G=G,
        src="A",
        dst="C",
        L_max=5,
        req=req,
        cap=cap,
        capacity_threshold=0.99,
        p_packet=0.6,
        memory=1,
        p_swap=0.6,
        p_gen=0.001,
        time_slot_duration=1e-4,
    )

    assert path is None
    assert delta == 0.0


def test_smallest_bottleneck():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "E"),
            ("A", "C"),
            ("C", "D"),
            ("D", "E"),
        ]
    )

    cap = defaultdict(float)
    cap[("A", "B")] = 0.4
    cap[("B", "E")] = 0.2
    cap[("A", "C")] = 0.1
    cap[("C", "D")] = 0.1
    cap[("D", "E")] = 0.1

    req = {
        "epr": 1,
        "period": 1.0,
    }

    path, delta = smallest_bottleneck(
        G=G,
        src="A",
        dst="E",
        L_max=3,
        req=req,
        cap=cap,
        p_packet=0.9,
        memory=1000,
        p_swap=0.6,
        p_gen=0.001,
        time_slot_duration=1e-4,
    )

    assert path == ["A", "C", "D", "E"]
    assert delta > 0.0


def test_least_capacity():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "E"),
            ("A", "C"),
            ("C", "D"),
            ("D", "E"),
        ]
    )

    cap = defaultdict(float)
    cap[("A", "B")] = 0.2
    cap[("B", "E")] = 0.1
    cap[("A", "C")] = 0.1
    cap[("C", "D")] = 0.1
    cap[("D", "E")] = 0.1

    req = {
        "epr": 1,
        "period": 1.0,
    }

    path, delta = least_capacity(
        G=G,
        src="A",
        dst="E",
        L_max=3,
        req=req,
        cap=cap,
        p_packet=0.9,
        memory=1000,
        p_swap=0.6,
        p_gen=0.001,
        time_slot_duration=1e-4,
    )

    assert path == ["A", "B", "E"]
    assert delta > 0.0
