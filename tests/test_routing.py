import math
from collections import defaultdict

import numpy as np
import pytest

from scheduling.fidelity import fidelity_bounds_and_paths
from scheduling.routing import (
    capacity_threshold,
    fidelity_shortest,
    find_feasible_path,
    highest_fidelity,
    least_capacity,
    shortest_paths,
    smallest_bottleneck,
)


def _uniform_fidelities(edges, value=0.9):
    return {edge: value for edge in edges}


@pytest.fixture
def basic_graph():
    return [
        ("Alice", "Bob"),
        ("Bob", "Charlie"),
        ("Charlie", "David"),
        ("Eve", "Frank"),
    ]


@pytest.fixture
def default_req():
    return {"epr": 1, "period": 1.0, "min_fidelity": 0.6}


@pytest.fixture
def pga_params():
    return dict(
        p_packet=0.9,
        memory=1000,
        p_swap=0.6,
        p_gen=0.001,
        time_slot_duration=1e-4
    )


@pytest.fixture
def linear_abc():
    edges = [("A", "B"), ("B", "C")]
    fidelities = _uniform_fidelities(edges)
    _, simple_paths = fidelity_bounds_and_paths(["A", "B", "C"], fidelities)
    return edges, fidelities, simple_paths


@pytest.fixture
def diamond_abcde():
    edges = [("A", "B"), ("B", "E"), ("A", "C"), ("C", "D"), ("D", "E")]
    fidelities = _uniform_fidelities(edges)
    _, simple_paths = fidelity_bounds_and_paths(
        ["A", "B", "C", "D", "E"], fidelities
    )
    return edges, fidelities, simple_paths


@pytest.mark.parametrize(
    "apps, expected",
    [
        (
            {
                "A": {"src": "Alice", "dst": "Charlie"},
                "B": {"src": "Bob", "dst": "Charlie"},
            },
            {
                "A": [["Alice", "Bob", "Charlie"]],
                "B": [["Bob", "Charlie"]],
            },
        ),
        (
            {
                "A": {"src": "Alice", "dst": "David"},
                "B": {"src": "Bob", "dst": "Charlie"},
            },
            {
                "A": [["Alice", "Bob", "Charlie", "David"]],
                "B": [["Bob", "Charlie"]]
            },
        ),
    ],
)
def test_shortest_paths_cases(basic_graph, apps, expected):
    assert shortest_paths(basic_graph, apps) == expected


def test_shortest_paths_empty(basic_graph):
    assert shortest_paths(basic_graph, {}) == {}
    assert shortest_paths([], {}) == {}


def test_find_feasible_path_basic():
    edges = [("Alice", "Bob"), ("Bob", "Charlie"), ("Charlie", "David")]
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
    _, simple_paths = fidelity_bounds_and_paths(
        ["Alice", "Charlie", "David", "Bob"], fidelities
    )
    result, _ = find_feasible_path(
        edges=edges,
        app_requests=app_requests,
        fidelities=fidelities,
        routing_mode="shortest",
        simple_paths=simple_paths,
        rng=np.random.default_rng(42),
    )
    assert result["A"][0] == ["Alice", "Bob", "Charlie"]
    assert result["B"][0] == ["Bob", "Charlie", "David"]


def test_find_feasible_path_min_fidelity_too_low(linear_abc):
    edges, fidelities, simple_paths = linear_abc
    app_requests = {
        "app": {
            "src": "A", "dst": "B", "min_fidelity": 0.4, "epr": 1, "period": 1
        }
    }
    result, e2e_fids = find_feasible_path(
        edges=edges,
        simple_paths=simple_paths,
        app_requests=app_requests,
        fidelities=fidelities,
    )
    assert result["app"] == []
    assert math.isnan(e2e_fids["app"])


@pytest.mark.parametrize("fn", [smallest_bottleneck, least_capacity])
def test_low_fidelity_path_skipped(fn, default_req, pga_params):
    simple_paths = {
        ("A", "E"): [
            (0.3, ("A", "B", "E")),
            (0.8, ("A", "C", "D", "E")),
        ]
    }
    path, delta, e2e_fid = fn(
        simple_paths=simple_paths,
        src="A",
        dst="E",
        req=default_req,
        cap=defaultdict(float),
        **pga_params,
    )
    assert path == ("A", "C", "D", "E")
    assert delta > 0.0
    assert e2e_fid == pytest.approx(0.8)


def test_capacity_threshold_low_fidelity_skipped(default_req, pga_params):
    simple_paths = {
        ("A", "C"): [(0.3, ("A", "B", "C")), (0.8, ("A", "D", "C"))]
    }
    path, delta, e2e_fid = capacity_threshold(
        simple_paths=simple_paths,
        src="A",
        dst="C",
        req=default_req,
        cap=defaultdict(float),
        threshold=0.99,
        **pga_params,
    )
    assert path == ("A", "D", "C")
    assert delta > 0.0
    assert e2e_fid == pytest.approx(0.8)


def test_smallest_bottleneck_selects_min_max_cap(default_req, pga_params):
    edges = [("A", "B"), ("B", "E"), ("A", "C"), ("C", "D"), ("D", "E")]
    _, simple_paths = fidelity_bounds_and_paths(
        ["A", "C", "D", "E"], _uniform_fidelities(edges)
    )
    cap = defaultdict(
        float,
        {
            ("A", "B"): 0.4,
            ("B", "E"): 0.2,
            ("A", "C"): 0.1,
            ("C", "D"): 0.1,
            ("D", "E"): 0.1,
        },
    )
    path, delta, _ = smallest_bottleneck(
        simple_paths=simple_paths,
        src="A",
        dst="E",
        req=default_req,
        cap=cap,
        **pga_params,
    )
    assert path == ("A", "C", "D", "E")
    assert delta > 0.0


def test_least_capacity_selects_min_sum_cap(default_req, pga_params):
    edges = [("A", "B"), ("B", "E"), ("A", "C"), ("C", "D"), ("D", "E")]
    _, simple_paths = fidelity_bounds_and_paths(
        ["A", "B", "E"], _uniform_fidelities(edges)
    )
    cap = defaultdict(
        float,
        {
            ("A", "B"): 0.2,
            ("B", "E"): 0.1,
            ("A", "C"): 0.1,
            ("C", "D"): 0.1,
            ("D", "E"): 0.1,
        },
    )
    path, delta, _ = least_capacity(
        simple_paths=simple_paths,
        src="A",
        dst="E",
        req=default_req,
        cap=cap,
        **pga_params,
    )
    assert path == ("A", "B", "E")
    assert delta > 0.0


def test_capacity_threshold_exceeded(default_req):
    edges = [("A", "B"), ("B", "C")]
    _, simple_paths = fidelity_bounds_and_paths(
        ["A", "B", "C"], _uniform_fidelities(edges)
    )
    cap = defaultdict(float, {("A", "B"): 0.95, ("B", "C"): 0.95})
    path, delta, _ = capacity_threshold(
        simple_paths=simple_paths,
        src="A",
        dst="C",
        req=default_req,
        cap=cap,
        threshold=0.99,
        p_packet=0.6,
        memory=1,
        p_swap=0.6,
        p_gen=0.001,
        time_slot_duration=1e-4,
    )
    assert path is None
    assert delta == 0.0


def test_fidelity_shortest_no_valid_path():
    edges = [("A", "B"), ("B", "C")]
    _, simple_paths = fidelity_bounds_and_paths(
        ["A", "B", "C"], _uniform_fidelities(edges, value=0.55)
    )
    paths, fid = fidelity_shortest(
        simple_paths, "A", "C", 0.6, np.random.default_rng(42)
    )
    assert paths == []
    assert math.isnan(fid)


def test_highest_fidelity_selects_best():
    simple_paths = {
        ("A", "E"): [
            (0.5, ("A", "X", "E")),
            (0.7, ("A", "B", "E")),
            (0.85, ("A", "C", "D", "E")),
        ]
    }
    path, fid = highest_fidelity(
        simple_paths, "A", "E", 0.75, np.random.default_rng(42)
    )
    assert path == ("A", "C", "D", "E")
    assert fid == pytest.approx(0.85)


@pytest.mark.parametrize("routing_mode", ["smallest", "least"])
def test_find_feasible_path_cap_modes_success(
    diamond_abcde, pga_params, routing_mode
):
    edges, fidelities, simple_paths = diamond_abcde
    app_requests = {
        "app": {
            "src": "A", "dst": "E", "min_fidelity": 0.6, "epr": 1, "period": 1
        }
    }
    result, e2e_fids = find_feasible_path(
        edges=edges,
        simple_paths=simple_paths,
        app_requests=app_requests,
        fidelities=fidelities,
        routing_mode=routing_mode,
        **pga_params,
        rng=np.random.default_rng(42),
    )
    assert result["app"]
    assert e2e_fids["app"] >= 0.5


@pytest.mark.parametrize("routing_mode", ["smallest", "least"])
def test_find_feasible_path_cap_modes_no_valid_path(routing_mode, pga_params):
    edges = [("A", "B"), ("B", "C")]
    fidelities = _uniform_fidelities(edges, value=0.55)
    _, simple_paths = fidelity_bounds_and_paths(["A", "B", "C"], fidelities)
    app_requests = {
        "app": {
            "src": "A", "dst": "C", "min_fidelity": 0.99, "epr": 1, "period": 1
        }
    }
    result, e2e_fids = find_feasible_path(
        edges=edges,
        simple_paths=simple_paths,
        app_requests=app_requests,
        fidelities=fidelities,
        routing_mode=routing_mode,
        **pga_params,
        rng=np.random.default_rng(0),
    )
    assert result["app"] == []
    assert math.isnan(e2e_fids["app"])


def test_find_feasible_path_capacity_routing_success(linear_abc, pga_params):
    edges, fidelities, simple_paths = linear_abc
    app_requests = {
        "app": {
            "src": "A", "dst": "C", "min_fidelity": 0.6, "epr": 1, "period": 1
        }
    }
    result, e2e_fids = find_feasible_path(
        edges=edges,
        simple_paths=simple_paths,
        app_requests=app_requests,
        fidelities=fidelities,
        routing_mode="capacity",
        threshold=0.99,
        **pga_params,
    )
    assert result["app"]
    assert e2e_fids["app"] >= 0.6


def test_find_feasible_path_capacity_routing_no_path(linear_abc):
    edges, fidelities, simple_paths = linear_abc
    app_requests = {
        "app": {
            "src": "A",
            "dst": "C",
            "min_fidelity": 0.6,
            "epr": 100,
            "period": 0.001,
        }
    }
    result, e2e_fids = find_feasible_path(
        edges=edges,
        simple_paths=simple_paths,
        app_requests=app_requests,
        fidelities=fidelities,
        routing_mode="capacity",
        threshold=0.99,
        p_packet=0.6,
        memory=1,
        p_swap=0.6,
        p_gen=0.001,
        time_slot_duration=1e-4,
    )
    assert result["app"] == []
    assert math.isnan(e2e_fids["app"])


def test_find_feasible_path_highest(linear_abc):
    edges, fidelities, simple_paths = linear_abc
    app_requests = {
        "app": {
            "src": "A", "dst": "C", "min_fidelity": 0.6, "epr": 1, "period": 1
        }
    }
    result, e2e_fids = find_feasible_path(
        edges=edges,
        simple_paths=simple_paths,
        app_requests=app_requests,
        fidelities=fidelities,
        routing_mode="highest",
        rng=np.random.default_rng(42),
    )
    assert result["app"]
    assert e2e_fids["app"] >= 0.6
