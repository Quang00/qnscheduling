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
    aggregate_metric,
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
                util_df["utilization"].astype(float).replace(
                    [np.inf, -np.inf], np.nan
                )
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
        "total_pga_duration", float("nan")
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
            "prefix": "admission",
            "ylabel": "Admission rate",
            "title": (
                rf"Admission Rate vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "percentage": True,
            "clip": (0.0, 1.0),
            "ymin": 0.0,
            "ymax": 1.0,
            "format_str": "{:.2f}",
            "include_in_return": True,
            "prefixed_columns": False,
        },
        {
            "key": "makespan",
            "prefix": "makespan",
            "ylabel": "Makespan (s)",
            "title": (
                rf"Makespan vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.1f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "throughput",
            "prefix": "throughput",
            "ylabel": "Throughput (jobs/s)",
            "title": (
                rf"Throughput vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.3f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "completed_ratio",
            "prefix": "completed_ratio",
            "ylabel": "Completed ratio",
            "title": (
                rf"Completed Ratio vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "clip": (0.0, 1.0),
            "ymin": 0.0,
            "ymax": 1.0,
            "format_str": "{:.2f}",
        },
        {
            "key": "avg_waiting_time",
            "prefix": "avg_waiting_time",
            "ylabel": "Average waiting time (s)",
            "title": (
                rf"Average Waiting Time vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "max_waiting_time",
            "prefix": "max_waiting_time",
            "ylabel": "Max waiting time (s)",
            "title": (
                rf"Max Waiting Time vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "avg_turnaround_time",
            "prefix": "avg_turnaround_time",
            "ylabel": "Average turnaround time (s)",
            "title": (
                rf"Average Turnaround Time vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "pga_duration_total",
            "prefix": "pga_duration_total",
            "ylabel": "Total PGA duration (s)",
            "title": (
                rf"Total PGA Duration vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
            "clip": (0.0, None),
        },
        {
            "key": "avg_link_utilization",
            "prefix": "avg_link_utilization",
            "ylabel": "Mean link utilization",
            "title": (
                rf"Mean Link Utilization vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "percentage": True,
            "clip": (None, 1.0),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
            "percentage_format": "{:.2f}%",
        },
        {
            "key": "max_link_utilization",
            "prefix": "max_link_utilization",
            "ylabel": "Max link utilization",
            "title": (
                rf"Max Link Utilization vs $p_{{\mathrm{{packet}}}}$ "
                f"(n_tasks={n_tasks})"
            ),
            "percentage": True,
            "clip": (None, 1.0),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
            "percentage_format": "{:.2f}%",
        },
    ]

    for metric in metrics:
        prefix = metric["prefix"]
        metric["mean_col"] = f"mean_{prefix}"
        metric["std_col"] = f"std_{prefix}"
        if metric.get("prefixed_columns", True):
            metric["sem_col"] = f"sem_{prefix}"
            metric["ci_col"] = f"ci95_{prefix}"
            metric["lower_col"] = f"lower_{prefix}"
            metric["upper_col"] = f"upper_{prefix}"
        else:
            metric["sem_col"] = "sem"
            metric["ci_col"] = "ci95"
            metric["lower_col"] = "lower"
            metric["upper_col"] = "upper"

    for metric in metrics:
        if metric["key"] == "admission_rate":
            metric["base_label"] = plot_label
            metric["plot_path"] = save_path
            metric["csv_path"] = os.path.join(run_dir, f"{plot_label}.csv")
        else:
            base_label = f"{metric['key']}_vs_ppacket_n_tasks_{n_tasks}"
            metric["base_label"] = base_label
            metric["plot_path"] = os.path.join(run_dir, f"{base_label}.png")
            metric["csv_path"] = os.path.join(run_dir, f"{base_label}.csv")

    return metrics


def render_plot(
    summary_df: pd.DataFrame,
    spec: dict,
    color,
    figsize: tuple[float, float],
    dpi: int,
    simulations_per_point: int,
) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.dropna(subset=[spec["mean_col"]]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.lineplot(
        data=plot_df,
        x="p_packet",
        y=spec["mean_col"],
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        ax=ax,
        color=color,
        label="Mean",
    )

    x_vals = plot_df["p_packet"].to_numpy()
    lower_vals = plot_df[spec["lower_col"]].to_numpy()
    upper_vals = plot_df[spec["upper_col"]].to_numpy()

    ax.fill_between(
        x_vals,
        lower_vals,
        upper_vals,
        alpha=0.18,
        color=color,
        label="95% CI",
        linewidth=0,
        zorder=0,
    )

    if plot_df["p_packet"].nunique() > 1:
        ax.set_xlim(plot_df["p_packet"].min(), plot_df["p_packet"].max())

    y_min, y_max = spec.get("ymin"), spec.get("ymax")
    if spec.get("auto_ylim"):
        data_min, data_max = np.nanmin(lower_vals), np.nanmax(upper_vals)
        if np.isfinite(data_min) and np.isfinite(data_max):
            pad_frac = spec.get("pad_fraction", 0.05)
            span = data_max - data_min
            pad = (
                (max(abs(data_max), 1.0) * pad_frac)
                if span <= 1e-9
                else (span * pad_frac)
            )
            y_min, y_max = data_min - pad, data_max + pad
            low_clip, high_clip = spec.get("clip", (None, None))
            if low_clip is not None:
                y_min = max(y_min, low_clip)
            if high_clip is not None:
                y_max = min(y_max, high_clip)

    bottom, top = ax.get_ylim()
    bottom = y_min if y_min is not None else bottom
    top = y_max if y_max is not None else top
    if not spec.get("auto_ylim") and y_min is None and y_max is None:
        bottom = 0.0
    if top <= bottom:
        top = bottom + max(abs(bottom) * 0.05, 1e-6)
    ax.set_ylim(bottom, top)

    if spec.get("percentage"):
        pct_fmt = spec.get("percentage_format", "{:.1f}%")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: pct_fmt.format(y * 100.0))
        )
    else:
        fmt = spec.get("format_str", "{:.2f}")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt.format(y)))

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.6)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.2)
    sns.despine(ax=ax)

    ax.set_xlabel(r"$p_{\mathrm{packet}}$")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"], pad=6)

    legend = ax.legend(frameon=False, loc="best", ncols=1)
    if legend is not None:
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)

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

    fig.tight_layout()
    directory = os.path.dirname(spec["plot_path"])
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(spec["plot_path"], bbox_inches="tight")
    plt.close(fig)


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

    Raises:
        RuntimeError: If no valid simulation data is generated.

    Returns:
        pd.DataFrame: DataFrame containing the aggregated simulation results.
    """
    default_kwargs = build_default_sim_args(config, simulation_kwargs)
    n_apps_value = int(default_kwargs.get("n_apps", 0))
    plot_label = f"admission_rate_vs_ppacket_n_tasks_{n_apps_value}"

    run_dir, timestamp = prepare_run_dir(output_dir, ppacket_values)
    save_path = save_path or os.path.join(run_dir, f"{plot_label}.png")
    raw_csv_path = os.path.join(run_dir, f"{plot_label}_raw.csv")

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
    aggregated_results = {}

    for idx, spec in enumerate(metrics_to_plot):
        summary_df = aggregate_metric(
            data=results_df,
            column=spec["key"],
            prefix=spec["prefix"],
            clip=spec.get("clip"),
            prefixed_columns=spec.get("prefixed_columns", True),
        )
        aggregated_results[spec["key"]] = summary_df
        summary_df.to_csv(spec["csv_path"], index=False)
        render_plot(
            summary_df=summary_df,
            spec=spec,
            color=palette[idx % len(palette)],
            figsize=figsize,
            dpi=dpi,
            simulations_per_point=simulations_per_point,
        )

    return aggregated_results.get("admission_rate", pd.DataFrame())


"""
# Example usage of the plot_metrics_vs_ppacket function.
if __name__ == "__main__":
    sweep_values = np.round(np.linspace(0.1, 0.9, 9), 2).tolist()
    plot_metrics_vs_ppacket(
        ppacket_values=sweep_values,
        simulations_per_point=100,
        simulation_kwargs={
            "n_apps": 100,
            "inst_range": (100, 100),
            "epr_range": (2, 2),
            "period_range": (70, 70),
            "hyperperiod_cycles": 100,
            "memory_lifetime": 50,
            "p_swap": 0.95,
        },
        config="configurations/network/Dumbbell.gml",
    )
"""
