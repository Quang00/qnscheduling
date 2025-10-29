import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import (
    AutoMinorLocator,
    FuncFormatter,
    LogFormatterMathtext,
)
from tqdm.auto import tqdm

from scheduling.main import run_simulation
from scheduling.pga import duration_pga
from utils.helper import (
    build_default_sim_args,
    build_tasks,
    generate_metadata,
    ppacket_dirname,
    prepare_run_dir,
)


def set_plot_theme(dpi: int) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )


def plot_pga_vs_memory(
    p_packet: float = 1,
    epr_pairs: int = 4,
    p_swap: float = 0.95,
    memories: list = None,
    n_swaps: list = None,
    path_folder: str = "docs/pga_duration_vs_memory",
) -> None:
    """Plot the duration of a PGA (Packet Generation Attempt) vs. memory
    lifetime for different numbers of entanglement swappings.

    Args:
        p_packet (float, optional): Probability of a packet being generated.
        epr_pairs (int, optional): Number of successes (number of EPR pairs
        generated).
        p_swap (float, optional): Probability of a successful entanglement
        swapping.
        memories (list, optional): List of memory lifetimes in milliseconds.
        n_swaps (list, optional): List of numbers of entanglement swappings.
        path_folder (str, optional): Path to save the plot image.
    """
    if not memories:
        memories = list(range(200, 1001, 100))
    if not n_swaps:
        n_swaps = [0, 2, 4, 6, 8, 10]

    data = []
    for memory in memories:
        for n in n_swaps:
            dur = duration_pga(p_packet, epr_pairs, n, memory, p_swap=p_swap)
            data.append(
                {
                    "Memory (ms)": memory,
                    "Swaps": n,
                    "Duration (s)": dur * 1e-6,
                }
            )

    df = pd.DataFrame(data)
    palette = sns.color_palette("colorblind", n_colors=len(n_swaps))

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    for idx, n in enumerate(n_swaps):
        subset = df[df["Swaps"] == n]
        ax.plot(
            subset["Memory (ms)"],
            subset["Duration (s)"],
            marker="o",
            linestyle="-",
            color=palette[idx],
            label=f"{n} swaps",
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlabel(r"Memory lifetime $\tau_{\mathrm{mem}}$ (ms)")
    ax.set_ylabel(r"PGA duration (s)")
    ax.set_title("PGA Duration vs Memory Lifetime")
    ax.legend(
        title=f"# swaps ($p_{{swap}}={p_swap}$)",
    )

    fig.tight_layout()

    fig.savefig(f"{path_folder}.png", dpi=300, format="png")
    plt.close(fig)


def plot_graph_from_gml(gml_file: str) -> None:
    """Plot a graph from a GML file.

    Args:
        gml_file (str): Path to the GML file.
    """
    G = nx.read_gml(gml_file)
    pos = {n: (data["lon"], data["lat"]) for n, data in G.nodes(data=True)}

    base = os.path.basename(gml_file)
    name, _ = os.path.splitext(base)

    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")

    nx.draw(G, pos, ax=ax, with_labels=True, node_size=80, font_size=5)

    ax.set_title(name)
    ax.axis("off")

    plt.show()


def simulate_one_ppacket(args: tuple) -> dict:
    (
        p_packet,
        run_seed,
        run_dir,
        default_kwargs,
        n_apps_value,
    ) = args

    ppacket_dir = os.path.join(run_dir, ppacket_dirname(p_packet))
    sd_dir = os.path.join(ppacket_dir, f"seed_{run_seed}")
    os.makedirs(sd_dir, exist_ok=True)

    args = default_kwargs.copy()
    args.update({"p_packet": p_packet, "seed": run_seed, "output_dir": sd_dir})

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        feasible, df, durations = run_simulation(**args)

    admission_rate = 1.0 if feasible else 0.0
    completed = 0
    total = 0
    if feasible and df is not None and not df.empty:
        status_series = df["status"].astype(str)
        completed = int((status_series == "completed").sum())
        total = int(len(status_series))
    durations = durations or {}

    summary_metrics = {
        "makespan": float("nan"),
        "throughput": float("nan"),
        "completed_ratio": float("nan"),
        "avg_waiting_time": float("nan"),
        "max_waiting_time": float("nan"),
        "avg_turnaround_time": float("nan"),
        "max_turnaround_time": float("nan"),
    }
    summary_path = os.path.join(sd_dir, "summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            row = summary_df.iloc[0]
            for key in summary_metrics:
                if key in row:
                    summary_metrics[key] = float(row[key])

    link_metrics = {
        "avg_link_utilization": float("nan"),
        "max_link_utilization": float("nan"),
    }
    link_util_path = os.path.join(sd_dir, "link_utilization.csv")
    if os.path.exists(link_util_path):
        util_df = pd.read_csv(link_util_path)
        if not util_df.empty and "utilization" in util_df.columns:
            util_values = (
                util_df["utilization"]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
            )
            util_array = util_values.to_numpy()
            util_mask = np.isfinite(util_array)
            if np.any(util_mask):
                util_finite = util_array[util_mask]
                link_metrics["avg_link_utilization"] = float(
                    np.nanmean(util_finite)
                )
                link_metrics["max_link_utilization"] = float(
                    np.nanmax(util_finite)
                )

    pga_duration_total = summary_metrics.get(
        "total_pga_duration",
        float("nan"),
    )
    if (not np.isfinite(pga_duration_total)) and durations:
        duration_vals = np.array(list(durations.values()), dtype=float)
        duration_vals = duration_vals[np.isfinite(duration_vals)]
        if duration_vals.size:
            pga_duration_total = float(duration_vals.sum())

    return {
        "p_packet": p_packet,
        "seed": run_seed,
        "feasible": feasible,
        "admission_rate": admission_rate,
        "completed": completed,
        "total_jobs": total,
        "n_apps": n_apps_value,
        "pga_duration_total": pga_duration_total,
        **summary_metrics,
        **link_metrics,
    }


def run_parallel_sims(
    tasks: list[tuple[Any, ...]],
    max_workers: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    mp_ctx = mp.get_context("spawn")
    records = []
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


def build_metric_specs(
    n_tasks: int,
    save_path: str,
    run_dir: str,
    plot_label: str,
) -> list[dict[str, Any]]:
    metrics = [
        {
            "key": "admission_rate",
            "plot_type": "line",
            "ylabel": "Admission rate",
            "title": (
                r"Admission Rate vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "clip": (0.0, 1.0),
            "ymin": 0.0,
            "ymax": 1.0,
            "format_str": "{:.2f}",
            "percentage": True,
            "auto_ylim": False,
        },
        {
            "key": "makespan",
            "plot_type": "violin",
            "ylabel": "Makespan (s)",
            "title": (
                r"Makespan vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.1f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "throughput",
            "plot_type": "violin",
            "ylabel": "Throughput (jobs/s)",
            "title": (
                r"Throughput vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.3f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "completed_ratio",
            "plot_type": "violin",
            "ylabel": "Completed ratio",
            "title": (
                r"Completed Ratio vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "clip": (0.0, 1.0),
            "format_str": "{:.2f}",
            "percentage": True,
            "auto_ylim": False,
        },
        {
            "key": "avg_waiting_time",
            "plot_type": "violin",
            "ylabel": "Average Waiting Time (s)",
            "title": (
                r"Average Waiting Time vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "max_waiting_time",
            "plot_type": "violin",
            "ylabel": "Max Waiting Time (s)",
            "title": (
                r"Max Waiting Time vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "avg_turnaround_time",
            "plot_type": "violin",
            "ylabel": "Average Turnaround Time (s)",
            "title": (
                r"Average Turnaround Time vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "pga_duration_total",
            "plot_type": "line",
            "ylabel": "Total PGA duration (s)",
            "title": (
                r"Total PGA Duration vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
            "clip": (0.0, None),
        },
        {
            "key": "avg_link_utilization",
            "plot_type": "violin",
            "ylabel": "Average Link Utilization",
            "title": (
                r"Average Link Utilization vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "clip": (None, 1.0),
            "format_str": "{:.2f}",
            "percentage": True,
            "auto_ylim": True,
            "pad_fraction": 0.1,
            "percentage_format": "{:.2f}%",
        },
        {
            "key": "max_link_utilization",
            "plot_type": "violin",
            "ylabel": "Max link utilization",
            "title": (
                r"Max Link Utilization vs $p_{\mathrm{packet}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "clip": (None, 1.0),
            "format_str": "{:.2f}",
            "percentage": True,
            "auto_ylim": True,
            "pad_fraction": 0.1,
            "percentage_format": "{:.2f}%",
        },
    ]

    for metric in metrics:
        if metric["key"] == "admission_rate":
            metric["base_label"] = plot_label
            metric["plot_path"] = save_path
        else:
            base_label = (
                f"{metric['key']}_vs_ppacket_n_tasks_{n_tasks}"
            )
            metric["base_label"] = base_label
            metric["plot_path"] = os.path.join(
                run_dir,
                f"{base_label}.png",
            )

    return metrics


def render_plot(
    spec: dict[str, Any],
    raw_data: pd.DataFrame,
    color,
    figsize: tuple[float, float],
    dpi: int,
    simulations_per_point: int,
) -> Optional[pd.DataFrame]:

    metric = spec["key"]
    if metric not in raw_data.columns:
        return None

    data = raw_data[["p_packet", metric]].copy()
    data["p_packet"] = pd.to_numeric(data["p_packet"], errors="coerce")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    clip_bounds = spec.get("clip")
    if clip_bounds is not None:
        lo, hi = clip_bounds
        data[metric] = data[metric].clip(lower=lo, upper=hi)

    data = data.dropna().reset_index(drop=True)

    summary_df: Optional[pd.DataFrame] = None
    if spec["plot_type"] == "line":
        summary_df = (
            data.groupby("p_packet", as_index=False)[metric].mean()
        )
        if summary_df.empty:
            return summary_df
        summary_df = summary_df.sort_values("p_packet").reset_index(drop=True)
    else:
        if data.empty:
            return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if spec["plot_type"] == "line":
        sns.lineplot(
            data=summary_df,
            x="p_packet",
            y=metric,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=color,
            ax=ax,
        )
        ax.margins(x=0.02)
    else:
        labelled = data.assign(
            p_packet_label=data["p_packet"].map(lambda val: f"{val:g}")
        )
        order = [f"{val:g}" for val in sorted(data["p_packet"].unique())]
        sns.violinplot(
            data=labelled,
            x="p_packet_label",
            y=metric,
            order=order,
            cut=0,
            inner="quartile",
            density_norm="count",
            color=color,
            linewidth=0.8,
            ax=ax,
        )

    series = (
        summary_df[metric].to_numpy()
        if summary_df is not None
        else data[metric].to_numpy()
    )
    values = np.asarray(series, dtype=float)

    y_min = spec.get("ymin")
    y_max = spec.get("ymax")

    auto_ylim = spec.get("auto_ylim", True)
    if auto_ylim and values.size:
        finite = values[np.isfinite(values)]
        if finite.size:
            data_min = float(finite.min())
            data_max = float(finite.max())
            if data_min == data_max:
                pad = max(abs(data_min) * 0.05, 1e-6)
                data_min -= pad
                data_max += pad
            else:
                pad = spec.get("pad_fraction", 0.05)
                span = data_max - data_min
                data_min -= span * pad
                data_max += span * pad
            if y_min is None:
                y_min = data_min
            else:
                y_min = max(y_min, data_min)
            if y_max is None:
                y_max = data_max
            else:
                y_max = min(y_max, data_max)

    if clip_bounds is not None:
        lo, hi = clip_bounds
        if lo is not None:
            y_min = lo if y_min is None else max(y_min, lo)
        if hi is not None:
            y_max = hi if y_max is None else min(y_max, hi)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    if spec.get("percentage"):
        pct_fmt = spec.get("percentage_format", "{:.1f}%")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, _: pct_fmt.format(val * 100.0))
        )
    else:
        fmt = spec.get("format_str", "{:.2f}")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, _: fmt.format(val))
        )

    if spec["plot_type"] == "line":
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.6)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.2)
    sns.despine(ax=ax)

    ax.set_xlabel(r"$p_{\mathrm{packet}}$")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"], pad=6)

    ax.text(
        0.99,
        0.02,
        f"{simulations_per_point} sims/point",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        color="#444",
    )

    directory = os.path.dirname(spec["plot_path"])
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    fig.savefig(spec["plot_path"], bbox_inches="tight")
    plt.close(fig)

    return summary_df


def plot_metrics_vs_ppacket(
    ppacket_values: list[float],
    simulations_per_point: int = 100,
    seed_start: int = 0,
    config: str = "configurations/network/Dumbbell.gml",
    save_path: str | None = None,
    output_dir: str = "results",
    simulation_kwargs: dict | None = None,
    figsize: tuple[float, float] = (7, 4.5),
    dpi: int = 300,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run multiple simulations varying the packet generation probability.

    Args:
        ppacket_values (list[float]): List of packet generation probabilities
        to test.
        simulations_per_point (int, optional): Number of simulations to run for
        each probability point.
        seed_start (int, optional): Starting seed value for random number
        generation.
        config (str, optional): Path to the network configuration file.
        save_path (str | None, optional): Path to save the plot.
        output_dir (str, optional): Directory to save simulation results.
        simulation_kwargs (dict | None, optional): Additional arguments for the
        simulation.
        figsize (tuple[float, float], optional): Size of the plot figure.
        dpi (int, optional): Dots per inch for the plot.
        max_workers (Optional[int], optional): Maximum number of workers for
        parallel processing.
        show_progress (bool, optional): Whether to show progress bars during
        simulations.

    Returns:
        pd.DataFrame: DataFrame containing the aggregated simulation results.
    """
    default_kwargs = build_default_sim_args(config, simulation_kwargs)
    n_apps_value = int(default_kwargs.get("n_apps", 0))
    plot_label = f"admission_rate_vs_ppacket_n_tasks_{n_apps_value}"

    run_dir, timestamp = prepare_run_dir(output_dir, ppacket_values)
    save_path = save_path or os.path.join(run_dir, f"{plot_label}.png")
    raw_csv_path = os.path.join(run_dir, f"{timestamp}_raw.csv")

    metrics_to_plot = build_metric_specs(
        n_tasks=n_apps_value,
        save_path=save_path,
        run_dir=run_dir,
        plot_label=plot_label,
    )

    generate_metadata(
        run_dir=run_dir,
        timestamp=timestamp,
        ppacket_values=ppacket_values,
        simulations_per_point=simulations_per_point,
        seed_start=seed_start,
        config=config,
        save_path=save_path,
        raw_csv_path=raw_csv_path,
        default_kwargs=default_kwargs,
        metrics_to_plot=metrics_to_plot,
    )

    tasks = build_tasks(
        ppacket_values=ppacket_values,
        simulations_per_point=simulations_per_point,
        seed_start=seed_start,
        run_dir=run_dir,
        default_kwargs=default_kwargs,
        n_apps=n_apps_value,
    )

    records = run_parallel_sims(
        tasks=tasks,
        max_workers=max_workers or os.cpu_count(),
        show_progress=show_progress,
    )

    results_df = pd.DataFrame(records)
    if results_df.empty:
        raise RuntimeError("No simulation data was generated.")
    results_df.to_csv(raw_csv_path, index=False)

    set_plot_theme(dpi)
    palette = sns.color_palette("colorblind", len(metrics_to_plot))

    admission_summary = pd.DataFrame()
    for idx, spec in enumerate(metrics_to_plot):
        summary_df = render_plot(
            spec=spec,
            raw_data=results_df,
            color=palette[idx % len(palette)],
            figsize=figsize,
            dpi=dpi,
            simulations_per_point=simulations_per_point,
        )
        if (
            spec["key"] == "admission_rate"
            and summary_df is not None
            and not summary_df.empty
        ):
            admission_summary = summary_df

    return admission_summary


"""
# Example usage of the plot_metrics_vs_ppacket function.
if __name__ == "__main__":
    sweep_values = np.round(np.linspace(0.1, 0.9, 9), 2).tolist()
    plot_metrics_vs_ppacket(
        ppacket_values=sweep_values,
        simulations_per_point=5000,
        simulation_kwargs={
            "n_apps": 1,
            "inst_range": (100, 100),
            "epr_range": (2, 2),
            "period_range": (1, 1),
            "hyperperiod_cycles": 100,
            "memory_lifetime": 200,
            "p_swap": 0.95,
        },
        config="configurations/network/Garr201201.gml",
    )
"""
