import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd

from utils.helper import (
    compute_edge_fidelities,
    generate_n_apps,
    gml_data,
    parallelizable_tasks,
    save_results,
)


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
        "A": {"D"},
        "B": {"D"},
        "C": {"D"},
        "D": {"A", "B", "C"},
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


def test_shared_link_conflict():
    paths = {
        "A": ["Alice", "Bob"],
        "B": ["Bob", "Alice"],
        "C": ["Alice", "Bob"],
    }
    res = parallelizable_tasks(paths)
    assert res == {"A": set(), "B": set(), "C": set()}
    assert_equal_set(res)


def test_two_parallel():
    paths = {"A": ["Alice"], "B": ["Bob", "David"], "C": ["Charlie", "David"]}
    res = parallelizable_tasks(paths)
    assert res["A"] == {"B", "C"}
    assert res["B"] == {"A", "C"}
    assert res["C"] == {"A", "B"}
    assert_equal_set(res)


def test_shared_node_but_distinct_links():
    paths = {
        "A": ["Alice", "Bob"],
        "B": ["Bob", "Charlie"],
    }
    res = parallelizable_tasks(paths)
    assert res == {"A": {"B"}, "B": {"A"}}
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
    pga_release_times = {"A0": 0.0, "B0": 5.0}
    app_specs = {
        "A": {
            "src": "srcA",
            "dst": "dstA",
            "instances": 2,
            "epr": 3,
            "period": 10.0,
            "policy": "deadline",
        },
        "B": {
            "src": "srcB",
            "dst": "dstB",
            "instances": 1,
            "epr": 1,
            "period": 12.0,
            "policy": "deadline",
        },
    }

    n_edges = 3
    makespan = 5.0
    link_utilization = {
        ("Alice", "Bob"): {"busy_time": 3.0, "utilization": 3.0 / makespan},
        ("Bob", "Charlie"): {"busy_time": 1.5, "utilization": 1.5 / makespan},
    }
    link_waiting = {
        ("Alice", "Bob"): {"total_waiting_time": 2.0, "pga_waited": 1},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_results(
            df=df,
            pga_names=pga_names,
            pga_release_times=pga_release_times,
            app_specs=app_specs,
            n_edges=n_edges,
            link_utilization=link_utilization,
            link_waiting=link_waiting,
            output_dir=tmpdir,
            verbose=False,
        )
        csv_files = {f for f in os.listdir(tmpdir) if f.endswith(".csv")}
        assert "pga_results.csv" in csv_files
        assert "link_utilization.csv" in csv_files
        assert "link_waiting.csv" in csv_files

        lk_ut_df = pd.read_csv(os.path.join(tmpdir, "link_utilization.csv"))
        assert len(lk_ut_df) == 2

        lk_wait_df = pd.read_csv(os.path.join(tmpdir, "link_waiting.csv"))
        assert len(lk_wait_df) == 1


def test_gml_data():
    gml_file = "configurations/network/basic/Dumbbell.gml"
    nodes, edges, fidelities, diameter = gml_data(gml_file)

    assert len(nodes) > 0
    assert len(edges) > 0
    assert len(fidelities) > 0
    assert len(fidelities) == len(edges)
    assert diameter > 0


def test_generate_n_apps():
    rng = np.random.default_rng(seed=42)
    nodes = ["Alice", "David"]
    n_apps = 3
    inst_range = (1, 5)
    epr_range = (1, 3)
    period_range = (10.0, 20.0)
    list_policies = ["policy1", "policy2"]
    bounds = {
        ("Alice", "David"): (0.8, 1.0),
        ("David", "Alice"): (0.8, 1.0),
    }

    apps = generate_n_apps(
        nodes=nodes,
        bounds=bounds,
        n_apps=n_apps,
        inst_range=inst_range,
        epr_range=epr_range,
        period_range=period_range,
        list_policies=list_policies,
        rng=rng,
    )
    assert len(apps) == n_apps


def test_save_results():
    df = pd.DataFrame(
        {
            "pga": ["A1"],
            "arrival_time": [0.0],
            "start_time": [0.0],
            "burst_time": [5.0],
            "completion_time": [5.0],
            "turnaround_time": [5.0],
            "waiting_time": [0.0],
            "status": ["completed"],
        }
    )
    pga_names = ["A1"]
    pga_release_times = {"A1": 0.0}
    app_specs = {
        "A": {
            "src": "Alice",
            "dst": "Bob",
            "instances": 1,
            "epr": 1,
            "period": 10.0,
            "policy": "deadline",
        },
    }
    n_edges = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        save_results(
            df=df,
            pga_names=pga_names,
            pga_release_times=pga_release_times,
            app_specs=app_specs,
            n_edges=n_edges,
            output_dir=tmpdir,
        )
        csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
        assert len(csv_files) > 0


def test_compute_edge_fidelities():
    G = nx.Graph()
    G.add_edge("A", "B", dist=50.0)
    G.add_edge("B", "C", dist=100.0)
    distances = {("A", "B"): 50.0, ("B", "C"): 100.0}

    fidelities = compute_edge_fidelities(G, distances, F_min=0.51)

    assert len(fidelities) == 2
    assert all(0.51 <= f <= 1.0 for f in fidelities.values())
