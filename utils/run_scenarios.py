from utils.parallel_simulations import run_ppacket_sweep_to_csv


def main():
    ppacket_values = [0.99]
    arrival_rate_values = [1, 2, 3, 4, 5]
    topology = "configurations/network/advanced/Garr201201.gml"
    simulations_per_point = 200

    base_kwargs = {
        "inst_range": 200,
        "epr_range": (2, 2),
        "period_range": (1, 1),
        "memory": 1000,
        "p_swap": 0.6,
        "p_gen": 1e-3,
        "time_slot_duration": 1e-4,
        "scheduler": "dynamic",
        "n_apps": 1,
        "graph": "gml",
        "windows": (40, 80),
    }

    scenarios = [
        {
            "id": 1,
            "name": "static",
            "routing_strategy": "static",
        },
        {
            "id": 2,
            "name": "hybrid",
            "routing_strategy": "hybrid",
            "routing": "least",
        },
        {
            "id": 3,
            "name": "dynamic",
            "routing_strategy": "dynamic",
        },
    ]

    for scenario in scenarios:
        routing_strategy = scenario["routing_strategy"]
        sim_kwargs = dict(
            base_kwargs,
            provisioning=routing_strategy == "rerouting",
            full_dynamic=routing_strategy == "dynamic",
            static_routing_mode=routing_strategy == "static",
        )
        if "routing" in scenario:
            sim_kwargs["routing"] = scenario["routing"]

        print(f"Running {scenario['name']}:")

        df, raw_csv_path = run_ppacket_sweep_to_csv(
            ppacket_values=ppacket_values,
            arrival_rate_values=arrival_rate_values,
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
