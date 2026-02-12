import pytest

from scheduling.routing import find_feasible_path, shortest_paths


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
