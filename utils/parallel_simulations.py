import multiprocessing as mp
import os
import shutil
import signal
import tempfile
from typing import Any, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from scheduling.main import run_simulation
from utils.helper import (
    build_default_sim_args,
    ppacket_dirname,
    prepare_run_dir,
)

_WORKER_DEFAULT_KWARGS = None
_WORKER_RUN_DIR = None


def _init_worker(default_kwargs: dict[str, Any], run_dir: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_DEFAULT_KWARGS, _WORKER_RUN_DIR
    _WORKER_DEFAULT_KWARGS = default_kwargs
    _WORKER_RUN_DIR = run_dir


def simulate_one_ppacket(args: tuple) -> dict:
    (p_packet, run_seed, arrival_rate, inst_range, keep_seed_outputs) = args
    default_kwargs = _WORKER_DEFAULT_KWARGS
    run_dir = _WORKER_RUN_DIR

    if keep_seed_outputs:
        base_dir = run_dir
        ppacket_dir = os.path.join(base_dir, ppacket_dirname(p_packet))
        sd_dir = os.path.join(ppacket_dir, f"seed_{run_seed}")
        os.makedirs(sd_dir, exist_ok=True)
        cleanup = False
    else:
        sd_dir = tempfile.mkdtemp(prefix=f"seed_{run_seed}_")
        cleanup = True

    sim_kwargs = {**default_kwargs}
    sim_kwargs.update(
        p_packet=p_packet,
        arrival_rate=arrival_rate,
        seed=run_seed,
        output_dir=sd_dir,
        save_csv=keep_seed_outputs,
        verbose=False,
    )
    if inst_range is not None:
        sim_kwargs["inst_range"] = inst_range

    try:
        _, summary = run_simulation(**sim_kwargs)
    finally:
        if cleanup:
            shutil.rmtree(sd_dir, ignore_errors=True)

    summary_metrics = {
        "admission_rate": float("nan"),
        "makespan": float("nan"),
        "throughput": float("nan"),
        "completed_ratio": float("nan"),
        "failed_ratio": float("nan"),
        "drop_ratio": float("nan"),
        "avg_defer_per_pga": float("nan"),
        "avg_retry_per_pga": float("nan"),
        "avg_burst_time": float("nan"),
        "avg_waiting_time": float("nan"),
        "max_waiting_time": float("nan"),
        "avg_turnaround_time": float("nan"),
        "max_turnaround_time": float("nan"),
        "avg_hops": float("nan"),
        "avg_min_fidelity": float("nan"),
        "avg_e2e_fidelity": float("nan"),
        "single_path_share_pct": float("nan"),
        "two_path_share_pct": float("nan"),
        "avg_pga_duration": float("nan"),
        "total_busy_time": float("nan"),
        "top5_busy_share": float("nan"),
        "top10_busy_share": float("nan"),
        "avg_link_utilization": float("nan"),
        "p90_link_utilization": float("nan"),
        "p95_link_utilization": float("nan"),
        "p90_link_avg_wait": float("nan"),
        "p95_link_avg_wait": float("nan"),
        "avg_queue_length": float("nan"),
        "p90_avg_queue_length": float("nan"),
        "p95_avg_queue_length": float("nan"),
        "avg_deg": float("nan"),
        "fairness": float("nan"),
        "routing_decision_count": 0,
        "routing_decision_runtime": float("nan"),
    }

    if summary is not None:
        summary_metrics.update(summary)

    result = {
        "p_packet": p_packet,
        "arrival_rate": arrival_rate,
        "seed": run_seed,
        **summary_metrics,
    }
    if inst_range is not None:
        result["inst_range"] = inst_range
    return result


def run_parallel_sims(
    tasks: list[tuple[Any, ...]],
    max_workers: int,
    show_progress: bool,
    default_kwargs: dict[str, Any],
    run_dir: str,
) -> list[dict[str, Any]]:
    mp_ctx = mp.get_context("spawn")
    n_tasks = len(tasks)
    if n_tasks == 0:
        return []
    n_procs = min(max_workers, n_tasks)
    chunksize = max(1, n_tasks // (n_procs * 4))
    pool = mp_ctx.Pool(
        processes=n_procs,
        initializer=_init_worker,
        initargs=(default_kwargs, run_dir),
    )
    try:
        it = pool.imap_unordered(simulate_one_ppacket, tasks, chunksize)
        if show_progress:
            records = [
                rec for rec in tqdm(
                    it, total=n_tasks, desc="Simulations", unit="run"
                )
            ]
        else:
            records = list(it)

        pool.close()
        return records
    except (KeyboardInterrupt, Exception):
        pool.terminate()
        raise
    finally:
        pool.join()


def run_ppacket_parallel_simulations(
    ppacket_values: Sequence[float],
    arrival_rate_values: Sequence[float],
    simulations_per_point: int,
    seed_start: int,
    run_dir: str,
    default_kwargs: dict,
    keep_seed_outputs: bool,
    inst_range_values: Sequence[int] | None = None,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    raw_csv_path: str | None = None,
) -> pd.DataFrame:
    seed_pool = [seed_start + i for i in range(simulations_per_point)]
    inst_range_sweep = inst_range_values or [None]
    tasks = [
        (p_packet, run_seed, arrival_rate, inst_range, keep_seed_outputs)
        for p_packet in ppacket_values
        for arrival_rate in arrival_rate_values
        for inst_range in inst_range_sweep
        for run_seed in seed_pool
    ]

    workers = max_workers or os.cpu_count() or 1
    records = run_parallel_sims(
        tasks=tasks,
        max_workers=workers,
        show_progress=show_progress,
        default_kwargs=default_kwargs,
        run_dir=run_dir,
    )
    results_df = pd.DataFrame(records)
    if raw_csv_path:
        results_df.to_csv(raw_csv_path, index=False)
    return results_df


def run_ppacket_sweep_to_csv(
    ppacket_values: Sequence[float],
    arrival_rate_values: Sequence[float],
    simulations_per_point: int,
    seed_start: int = 0,
    config: str = "configurations/network/Dumbbell.gml",
    output_dir: str = "results",
    simulation_kwargs: dict | None = None,
    keep_seed_outputs: bool = False,
    inst_range_values: Sequence[int] | None = None,
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

    df = run_ppacket_parallel_simulations(
        ppacket_values=ppacket_values,
        arrival_rate_values=arrival_rate_values,
        simulations_per_point=simulations_per_point,
        seed_start=seed_start,
        run_dir=run_dir,
        default_kwargs=default_kwargs,
        keep_seed_outputs=keep_seed_outputs,
        inst_range_values=inst_range_values,
        max_workers=max_workers,
        show_progress=show_progress,
        raw_csv_path=raw_csv_path,
    )
    return df, raw_csv_path
