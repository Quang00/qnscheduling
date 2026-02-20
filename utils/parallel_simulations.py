import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from scheduling.main import run_simulation
from utils.helper import (
    build_default_sim_args,
    build_tasks,
    ppacket_dirname,
    prepare_run_dir,
)


def simulate_one_ppacket(args: tuple) -> dict:
    (
        p_packet,
        run_seed,
        run_dir,
        default_kwargs,
        n_apps_value,
        keep_seed_outputs,
    ) = args

    n_apps_int = int(n_apps_value) if n_apps_value is not None else None

    if keep_seed_outputs:
        base_dir = run_dir
        if n_apps_int is not None:
            base_dir = os.path.join(run_dir, f"napps_{n_apps_int}")
        os.makedirs(base_dir, exist_ok=True)
        ppacket_dir = os.path.join(base_dir, ppacket_dirname(p_packet))
        os.makedirs(ppacket_dir, exist_ok=True)
        sd_dir = os.path.join(ppacket_dir, f"seed_{run_seed}")
        os.makedirs(sd_dir, exist_ok=True)
    else:
        sd_dir = os.path.join(tempfile.gettempdir(), f"seed_{run_seed}")

    sim_kwargs = default_kwargs.copy()
    sim_kwargs.update({"p_packet": p_packet, "seed": run_seed})
    sim_kwargs["output_dir"] = sd_dir
    sim_kwargs["n_apps"] = n_apps_int
    sim_kwargs["save_csv"] = keep_seed_outputs
    sim_kwargs["verbose"] = False

    feasible, summary = run_simulation(
        **sim_kwargs
    )

    summary_metrics = {
        "admission_rate": float("nan"),
        "makespan": float("nan"),
        "throughput": float("nan"),
        "completed_ratio": float("nan"),
        "failed_ratio": float("nan"),
        "drop_ratio": float("nan"),
        "avg_defer_per_pga": float("nan"),
        "avg_retry_per_pga": float("nan"),
        "avg_waiting_time": float("nan"),
        "max_waiting_time": float("nan"),
        "avg_turnaround_time": float("nan"),
        "max_turnaround_time": float("nan"),
        "avg_hops": float("nan"),
        "avg_min_fidelity": float("nan"),
        "avg_e2e_fidelity": float("nan"),
        "avg_pga_duration": float("nan"),
        "total_busy_time": float("nan"),
        "avg_link_utilization": float("nan"),
        "p90_link_utilization": float("nan"),
        "p95_link_utilization": float("nan"),
        "p90_link_avg_wait": float("nan"),
        "p95_link_avg_wait": float("nan"),
        "avg_queue_length": float("nan"),
        "p90_avg_queue_length": float("nan"),
        "p95_avg_queue_length": float("nan"),
    }

    if summary:
        summary_metrics.update(summary)

    result = {
        "p_packet": p_packet,
        "seed": run_seed,
        "n_apps": n_apps_int,
        **summary_metrics,
    }

    if not keep_seed_outputs and os.path.exists(sd_dir):
        if sd_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(sd_dir, ignore_errors=True)

    return result


def run_parallel_sims(
    tasks: list[tuple[Any, ...]],
    max_workers: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    mp_ctx = mp.get_context("spawn")
    records = []
    n_tasks = len(tasks)
    size = max(1, n_tasks // (max_workers * 4))

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
        if show_progress:
            futures = [ex.submit(simulate_one_ppacket, t) for t in tasks]
            with tqdm(
                total=len(futures),
                desc="Simulations",
                unit="run",
            ) as pbar:
                for fut in as_completed(futures):
                    records.append(fut.result())
                    pbar.update(1)
        else:
            for rec in ex.map(simulate_one_ppacket, tasks, chunksize=size):
                records.append(rec)
    return records


def run_ppacket_parallel_simulations(
    ppacket_values: Sequence[float],
    simulations_per_point: int,
    seed_start: int,
    run_dir: str,
    default_kwargs: dict,
    n_apps_values: Sequence[int],
    keep_seed_outputs: bool,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    raw_csv_path: str | None = None,
) -> pd.DataFrame:
    n_apps_list = [int(v) for v in n_apps_values]
    if not n_apps_list:
        raise ValueError("n_apps_values must contain at least one value")

    tasks = build_tasks(
        ppacket_values=ppacket_values,
        sim_per_point=simulations_per_point,
        seed_start=seed_start,
        run_dir=run_dir,
        default_kwargs=default_kwargs,
        n_apps_values=n_apps_list,
        keep_seed_outputs=keep_seed_outputs,
    )

    workers = max_workers or os.cpu_count() or 1
    records = run_parallel_sims(
        tasks=tasks,
        max_workers=int(workers),
        show_progress=show_progress,
    )
    results_df = pd.DataFrame(records)
    if raw_csv_path:
        results_df.to_csv(raw_csv_path, index=False)
    return results_df


def run_ppacket_sweep_to_csv(
    ppacket_values: Sequence[float],
    simulations_per_point: int,
    seed_start: int = 0,
    config: str = "configurations/network/Dumbbell.gml",
    output_dir: str = "results",
    simulation_kwargs: dict | None = None,
    n_apps_values: Sequence[int] = (100,),
    keep_seed_outputs: bool = False,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, str]:
    run_dir, timestamp = prepare_run_dir(
        output_dir,
        ppacket_values,
        keep_seed_outputs=keep_seed_outputs,
    )
    raw_csv_path = os.path.join(run_dir, f"{timestamp}_raw.csv")

    default_kwargs = build_default_sim_args(config, simulation_kwargs)
    default_kwargs["n_apps"] = int(n_apps_values[0])

    df = run_ppacket_parallel_simulations(
        ppacket_values=ppacket_values,
        simulations_per_point=simulations_per_point,
        seed_start=seed_start,
        run_dir=run_dir,
        default_kwargs=default_kwargs,
        n_apps_values=n_apps_values,
        keep_seed_outputs=keep_seed_outputs,
        max_workers=max_workers,
        show_progress=show_progress,
        raw_csv_path=raw_csv_path,
    )
    return df, raw_csv_path
