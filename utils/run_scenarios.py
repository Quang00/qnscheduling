from utils.parallel_simulations import run_ppacket_sweep_to_csv


def main():
    ppacket_values = [0.9]
    arrival_rate_values = [1, 3, 5, 7, 9]
    inst_range_values = [100]
    # arrival_rate_values = [3]
    # inst_range_values = [50, 100, 150, 200]
    topology = "configurations/network/basic/Grid_2_2.gml"
    simulations_per_point = 20

    base_kwargs = {
        "epr_range": (2, 2),
        "deadline_range": (1, 1),
        "memory": 1,
        "p_swap": 0.5,
        "coherence": 0.020,
        "time_slot_duration": 1e-4,
        "graph": "gml",
    }

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
            keep_seed_outputs=False,
        )
        df.insert(0, "scenario", scenario["id"])
        df.to_csv(raw_csv_path, index=False)

        print(f"Done: {scenario['name']} -> {raw_csv_path}")


if __name__ == "__main__":
    main()
