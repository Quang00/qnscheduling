from utils.parallel_simulations import run_ppacket_sweep_to_csv


def main():
    sweep_values = [0.9]
    n_apps_values = [50, 75, 100, 125, 150, 175, 200]
    topology = "configurations/network/Garr201201.gml"
    simulations_per_point = 200

    base_kwargs = {
        "inst_range": (100, 100),
        "epr_range": (2, 2),
        "period_range": (1, 1),
        "memory": 1000,
        "p_swap": 0.6,
        "p_gen": 1e-3,
        "time_slot_duration": 1e-4,
        "scheduler": "dynamic",
        "arrival_rate": 1,
    }

    scenarios = [
        {
            "id": 1,
            "name": "shortest_no-fid",
            "routing": "shortest",
            "fidelity_range": (0.0, 0.0),
        },
        {
            "id": 2,
            "name": "shortest_with-fid",
            "routing": "shortest",
            "fidelity_range": (0.6, 0.85),
        },
        {
            "id": 3,
            "name": "capacity_with-fid_cap-0.8",
            "routing": "capacity",
            "fidelity_range": (0.6, 0.85),
            "capacity_threshold": 0.8,
        },
        {
            "id": 4,
            "name": "capacity_with-fid_cap-1.0",
            "routing": "capacity",
            "fidelity_range": (0.6, 0.85),
            "capacity_threshold": 1.0,
        },
    ]

    for scenario in scenarios:
        sim_kwargs = dict(
            base_kwargs,
            routing=scenario["routing"],
            fidelity_range=scenario["fidelity_range"],
            capacity_threshold=scenario.get("capacity_threshold", None),
        )

        print(f"Running {scenario['name']}:")

        df, raw_csv_path = run_ppacket_sweep_to_csv(
            ppacket_values=sweep_values,
            simulations_per_point=simulations_per_point,
            simulation_kwargs=sim_kwargs,
            config=topology,
            n_apps_values=n_apps_values,
            keep_seed_outputs=False,
        )
        df.insert(0, "scenario", scenario["id"])
        df.to_csv(raw_csv_path, index=False)

        print(f"Done: {scenario['name']} -> {raw_csv_path}")


if __name__ == "__main__":
    main()
