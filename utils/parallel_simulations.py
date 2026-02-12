import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from typing import Any, Optional, Sequence

import numpy as np
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
        sd_dir = tempfile.mkdtemp(prefix=f"seed_{run_seed}_")

    sim_kwargs = default_kwargs.copy()
    sim_kwargs.update({"p_packet": p_packet, "seed": run_seed})
    sim_kwargs["output_dir"] = sd_dir
    sim_kwargs["n_apps"] = n_apps_int
    sim_kwargs["save_csv"] = keep_seed_outputs

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        feasible, df, durations, link_util, link_waiting = run_simulation(
            **sim_kwargs
        )

    completed = 0
    total = 0
    if feasible and df is not None and not df.empty:
        status_series = df["status"].astype(str)
        completed = int((status_series == "completed").sum())
        total = int(len(status_series))
    durations = durations or {}

    summary_metrics = {
        "admission_rate": float("nan"),
        "makespan": float("nan"),
        "throughput": float("nan"),
        "completed_ratio": float("nan"),
        "failed_ratio": float("nan"),
        "deadline_miss_rate": float("nan"),
        "drop_ratio": float("nan"),
        "avg_defer_per_pga": float("nan"),
        "avg_retry_per_pga": float("nan"),
        "deadline_miss": float("nan"),
        "avg_waiting_time": float("nan"),
        "max_waiting_time": float("nan"),
        "avg_turnaround_time": float("nan"),
        "max_turnaround_time": float("nan"),
        "avg_hops": float("nan"),
        "avg_min_fidelity": float("nan"),
        "avg_pga_duration": float("nan"),
        "total_busy_time": float("nan"),
        "avg_link_utilization": float("nan"),
        "p90_link_utilization": float("nan"),
        "p95_link_utilization": float("nan"),
        "p90_link_avg_wait": float("nan"),
        "p95_link_avg_wait": float("nan"),
        "useful_utilization": float("nan"),
    }
    summary_path = os.path.join(sd_dir, "summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            row = summary_df.iloc[0]
            for key in summary_metrics:
                if key in row:
                    summary_metrics[key] = float(row[key])

    lk_util_stats = {}
    if link_util:
        util_values = [
            metrics.get("utilization", 0.0)
            for metrics in link_util.values()
        ]
        busy_values = [
            metrics.get("busy_time", 0.0)
            for metrics in link_util.values()
        ]
        if util_values:
            lk_util_stats["max_link_utilization"] = float(max(util_values))
            lk_util_stats["min_link_utilization"] = float(min(util_values))
            lk_util_stats["std_link_utilization"] = float(np.std(util_values))
            lk_util_stats["total_link_busy_time"] = float(sum(busy_values))

    lk_wait_stats = {}
    if link_waiting:
        avg_wait_values = []
        for waiting in link_waiting.values():
            total_wait = waiting.get("total_waiting_time", 0.0)
            pga_waited = waiting.get("pga_waited", 0)
            if pga_waited > 0:
                avg_wait_values.append(total_wait / pga_waited)
        if avg_wait_values:
            lk_wait_stats["max_link_avg_wait"] = float(max(avg_wait_values))
            lk_wait_stats["min_link_avg_wait"] = float(min(avg_wait_values))
            lk_wait_stats["std_link_avg_wait"] = float(np.std(avg_wait_values))

    result = {
        "p_packet": p_packet,
        "seed": run_seed,
        "completed": completed,
        "total_jobs": total,
        "n_apps": n_apps_int,
        **summary_metrics,
        **lk_util_stats,
        **lk_wait_stats,
    }
    if not keep_seed_outputs and sd_dir.startswith(tempfile.gettempdir()):
        shutil.rmtree(sd_dir, ignore_errors=True)
    return result


def run_parallel_sims(
    tasks: list[tuple[Any, ...]],
    max_workers: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    mp_ctx = mp.get_context("spawn")
    records: list[dict[str, Any]] = []
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
            for rec in ex.map(simulate_one_ppacket, tasks, chunksize=1):
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
        simulations_per_point=simulations_per_point,
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
