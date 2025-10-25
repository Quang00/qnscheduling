import networkx as nx
import numpy as np
import pandas as pd
import pytest

from utils.helper import (
    edges_delay,
    hyperperiod,
    parallelizable_tasks,
    save_results,
    shortest_paths,
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
        "A": ("Alice", "Charlie"),
        "B": ("Bob", "Charlie"),
    }
    result = shortest_paths(basic_graph, apps)
    assert result == {
        "A": ["Alice", "Bob", "Charlie"],
        "B": ["Bob", "Charlie"],
    }


def test_shortest_paths_basic_long(basic_graph):
    apps = {
        "A": ("Alice", "David"),
        "B": ("Bob", "Charlie"),
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


def test_raises_no_path_for_disconnected(basic_graph):
    with pytest.raises(nx.NetworkXNoPath):
        shortest_paths(basic_graph, {"nopath": ("Alice", "Eve")})


def assert_equal_set(res: dict[str, set[str]]):
    for a, par_set in res.items():
        for b in par_set:
            assert a in res[b], f"Not equal set between {a} and {b}"


def test_parallelizable_tasks_basic():
    paths = {
        "A": ["Alice", "Bob", "David"],
        "B": ["Alice", "Bob"],
        "C": ["Charlie", "Alice", "Bob"],
        "D": ["Charlie", "David"],
    }
    res = parallelizable_tasks(paths)
    assert res == {
        "A": set(),
        "B": {"D"},
        "C": set(),
        "D": {"B"},
    }
    assert_equal_set(res)
    assert set(res.keys()) == set(paths.keys())


def test_parallelizable_tasks_empty():
    assert parallelizable_tasks({}) == {}


def test_everyone_parallel():
    paths = {"A": ["Alice"], "B": ["Bob"], "C": ["Charlie"]}
    res = parallelizable_tasks(paths)
    assert res == {
        "A": {"B", "C"},
        "B": {"A", "C"},
        "C": {"A", "B"},
    }
    assert_equal_set(res)


def test_none_parallel():
    paths = {"A": ["Alice"], "B": ["Alice"], "C": ["Alice"]}
    res = parallelizable_tasks(paths)
    assert res == {"A": set(), "B": set(), "C": set()}
    assert_equal_set(res)


def test_two_parallel():
    paths = {"A": ["Alice"], "B": ["Bob", "David"], "C": ["Charlie", "David"]}
    res = parallelizable_tasks(paths)
    assert res["A"] == {"B", "C"}
    assert res["B"] == {"A"}
    assert res["C"] == {"A"}
    assert_equal_set(res)


def test_save_results_basic(tmp_path):
    df = pd.DataFrame(
        [
            {
                "pga": "A0",
                "arrival_time": 0.0,
                "start_time": 0.0,
                "burst_time": 0.2,
                "completion_time": 0.2,
                "turnaround_time": 0.2,
                "waiting_time": 0.0,
                "status": "completed",
                "deadline": 1.0,
            }
        ]
    )

    pga_names = ["A0", "B0"]
    release_times = {"A0": 0.0, "B0": 5.0}
    apps = {"A": ("srcA", "dstA"), "B": ("srcB", "dstB")}
    instances = {"A": 2, "B": 1}
    epr_pairs = {"A": 3, "B": 1}
    policies = {"A": "deadline", "B": "deadline"}

    save_results(
        df,
        pga_names,
        release_times,
        apps,
        instances,
        epr_pairs,
        policies,
        output_dir=str(tmp_path),
    )

    result = pd.read_csv(tmp_path / "pga_results.csv")

    assert set(result["pga"]) == {"A0", "B0"}
    b0_row = result.loc[result["pga"] == "B0"].iloc[0]

    assert b0_row["status"] == "missing"
    assert np.isnan(b0_row["start_time"])
    assert b0_row["arrival_time"] == 5.0
    assert b0_row["src_node"] == "srcB"
    assert b0_row["dst_node"] == "dstB"
    assert b0_row["instances"] == 1
    assert b0_row["pairs_requested"] == 1
    assert b0_row["policy"] == "deadline"


def edges_delay_basic():
    distances = {("Alice", "Bob"): 1000.0, ("Bob", "Charlie"): 500.0}
    result = edges_delay(distances)

    expected_ab = 1000.0 / 200_000.0
    expected_bc = 500.0 / 200_000.0

    assert result[("Alice", "Bob")] == expected_ab
    assert result[("Bob", "Alice")] == expected_ab
    assert result[("Bob", "Charlie")] == expected_bc
    assert result[("Charlie", "Bob")] == expected_bc

    assert set(result.keys()) == {
        ("Alice", "Bob"),
        ("Bob", "Alice"),
        ("Bob", "Charlie"),
        ("Charlie", "Bob"),
    }


def test_edges_delay_empty():
    assert edges_delay({}) == {}


def test_zero_distance_gives_zero_delay():
    distances = {("Alice", "Bob"): 0.0}
    res = edges_delay(distances)
    assert res == {("Alice", "Bob"): 0.0, ("Bob", "Alice"): 0.0}


def test_hyperperiod_empty_and_negative():
    assert hyperperiod({}) == 0.0
    assert hyperperiod({"Alice": 0.0, "Bob": -1.0}) == 0.0


def test_hyperperiod_basic():
    periods = {"Alice": 10.0, "Bob": 10.0}
    assert hyperperiod(periods) == 10.0

    periods = {"Alice": 4.0, "Bob": 6.0, "Charlie": 10.0}
    assert hyperperiod(periods) == 60.0
