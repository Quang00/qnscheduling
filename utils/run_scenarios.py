import sys

from utils.parallel_simulations import run_ppacket_sweep_to_csv


def main():
    graph = sys.argv[1] if len(sys.argv) > 1 else "gml"
    topology = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "configurations/network/basic/3_equal_paths.gml"
    )
    coherence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.020
    deadline = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    out_dir = sys.argv[5] if len(sys.argv) > 5 else "results"

    ppacket_values = [0.9]
    arrival_rate_values = [1, 3, 5, 7, 9]
    inst_range_values = [300]
    # arrival_rate_values = [3]
    # inst_range_values = [50, 100, 150, 200]
    simulations_per_point = 20

    base_kwargs = {
        "epr_range": (2, 2),
        "deadline_range": (deadline, deadline),
        "memory": 50,
        "p_swap": 0.5,
        "coherence": coherence,
        "time_slot_duration": 1e-4,
        "graph": graph,
    }
    if graph == "gml" and "3_equal_paths" in topology:
        base_kwargs["end_nodes"] = ["A", "B"]

    scenarios = [
        {
            "id": 1,
            "name": "precomputed",
            "routing_strategy": "static",
        },
        {
            "id": 2,
            "name": "proactive",
            "routing_strategy": "hybrid",
            "routing": "smallest",
        },
        {
            "id": 3,
            "name": "hybrid",
            "routing_strategy": "rerouting",
            "routing": "smallest",
        },
        {
            "id": 4,
            "name": "wc",
            "routing_strategy": "dynamic",
        },
        {
            "id": 5,
            "name": "nwc",
            "routing_strategy": "nwc",
        },
    ]

    for scenario in scenarios:
        routing_strategy = scenario["routing_strategy"]
        sim_kwargs = dict(
            base_kwargs,
            provisioning=routing_strategy == "rerouting",
            full_dynamic=routing_strategy in ("dynamic", "nwc", "fastest"),
            static_routing_mode=routing_strategy == "static",
            dynamic_mode=(
                routing_strategy
                if routing_strategy in ("nwc", "fastest")
                else "wc"
            ),
        )
        if "routing" in scenario:
            sim_kwargs["routing"] = scenario["routing"]

        print(f"Running {scenario['name']}:")

        df, raw_csv_path = run_ppacket_sweep_to_csv(
            ppacket_values=ppacket_values,
            arrival_rate_values=arrival_rate_values,
            inst_range_values=inst_range_values,
            simulations_per_point=simulations_per_point,
            simulation_kwargs=sim_kwargs,
            config=topology,
            output_dir=out_dir,
            keep_seed_outputs=False,
        )
        df.insert(0, "scenario", scenario["id"])
        df.to_csv(raw_csv_path, index=False)

        print(f"Done: {scenario['name']} -> {raw_csv_path}")


if __name__ == "__main__":
    main()
