import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from typing import Any, Optional, Sequence

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
    palette = sns.color_palette("tab10", n_colors=len(n_swaps))

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
        keep_seed_outputs,
    ) = args

    n_apps_int = int(n_apps_value) if n_apps_value is not None else None

    tmp_dir = None
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
        tmp_dir = sd_dir

    args = default_kwargs.copy()
    args.update({"p_packet": p_packet, "seed": run_seed, "output_dir": sd_dir})
    if n_apps_int is not None:
        args["n_apps"] = n_apps_int

    try:
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
        summary_row = None
        summary_path = os.path.join(sd_dir, "summary.csv")
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
            if not summary_df.empty:
                row = summary_df.iloc[0]
                summary_row = row
                for key in summary_metrics:
                    if key in row:
                        summary_metrics[key] = float(row[key])

        avg_link_utilization = float("nan")
        if summary_row is not None:
            avg_link_utilization = float(
                summary_row.get("avg_link_utilization", avg_link_utilization)
            )

        pga_duration_total = float("nan")
        if summary_row is not None:
            pga_duration_total = float(
                summary_row.get("total_pga_duration", pga_duration_total)
            )

        link_metrics = {
            "max_link_utilization": float("nan"),
        }
        link_util_path = os.path.join(sd_dir, "link_utilization.csv")
        if os.path.exists(link_util_path):
            util_df = pd.read_csv(link_util_path)
            if not util_df.empty and "utilization" in util_df.columns:
                util_values = util_df["utilization"].astype(float)
                avg_link_utilization = float(util_values.mean())
                link_metrics["max_link_utilization"] = float(
                    util_values.max()
                )

        if durations:
            duration_vals = np.array(list(durations.values()), dtype=float)
            if duration_vals.size:
                pga_duration_total = float(duration_vals.sum())

        result = {
            "p_packet": p_packet,
            "seed": run_seed,
            "feasible": feasible,
            "admission_rate": admission_rate,
            "completed": completed,
            "total_jobs": total,
            "n_apps": n_apps_int,
            "pga_duration_total": pga_duration_total,
            "avg_link_utilization": avg_link_utilization,
            **summary_metrics,
            **link_metrics,
        }
        return result
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
    n_tasks_label: str,
    n_tasks_display: str,
    save_path: str,
    run_dir: str,
    plot_label: str,
    group_column: str | None = None,
    group_labels: dict | None = None,
    group_palette: Sequence[str] | None = None,
    group_palette_map: dict[str, Any] | None = None,
    individual_n_apps: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    metric_templates = [
        {
            "key": "admission_rate",
            "plot_type": "line",
            "ylabel": "Admission rate",
            "title": ("Admission Rate vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"),
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
            "title": ("Makespan vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"),
            "format_str": "{:.1f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "throughput",
            "plot_type": "violin",
            "ylabel": "Throughput (jobs/s)",
            "title": ("Throughput vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"),
            "format_str": "{:.3f}",
        },
        {
            "key": "completed_ratio",
            "plot_type": "line",
            "ylabel": "Completed ratio",
            "title": (
                "Completed Ratio vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"
            ),
            "format_str": "{:.2f}",
            "percentage": True,
            "auto_ylim": False,
        },
        {
            "key": "completed_ratio_delta",
            "plot_type": "line",
            "ylabel": "Completed ratio - $p_{\\mathrm{packet}}$",
            "title": (
                "Completed Ratio Delta vs $p_{\\mathrm{packet}}$ "
                "(n_tasks=%s)"
            ),
            "format_str": "{:.3f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "avg_waiting_time",
            "plot_type": "violin",
            "ylabel": "Average Waiting Time (s)",
            "title": (
                "Average Waiting Time vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"
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
                "Max Waiting Time vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"
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
                "Average Turnaround Time vs $p_{\\mathrm{packet}}$ "
                "(n_tasks=%s)"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "pga_duration_total",
            "plot_type": "violin",
            "ylabel": "Total PGA duration (s)",
            "title": (
                "Total PGA Duration vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"
            ),
            "format_str": "{:.2f}",
            "auto_ylim": True,
            "pad_fraction": 0.1,
        },
        {
            "key": "avg_link_utilization",
            "plot_type": "violin",
            "ylabel": "Average Link Utilization",
            "title": (
                "Average Link Utilization vs $p_{\\mathrm{packet}}$ "
                "(n_tasks=%s)"
            ),
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
                "Max Link Utilization vs $p_{\\mathrm{packet}}$ (n_tasks=%s)"
            ),
            "format_str": "{:.2f}",
            "percentage": True,
            "auto_ylim": True,
            "pad_fraction": 0.1,
            "percentage_format": "{:.2f}%",
        },
    ]

    specs = []
    individual_values = []
    seen_values = set()
    for val in individual_n_apps or []:
        intval = int(val)
        if intval in seen_values:
            continue
        seen_values.add(intval)
        individual_values.append(intval)

    for template in metric_templates:
        spec = template.copy()
        title = spec.pop("title")
        spec["title"] = title % n_tasks_display
        if spec["key"] == "admission_rate":
            spec["base_label"] = plot_label
            spec["plot_path"] = save_path
        else:
            base_label = f"{spec['key']}_vs_ppacket_n_tasks_{n_tasks_label}"
            spec["base_label"] = base_label
            spec["plot_path"] = os.path.join(
                run_dir,
                f"{base_label}.png",
            )
        spec["group_column"] = group_column
        if group_labels is not None:
            spec["group_labels"] = group_labels.copy()
        if group_palette is not None:
            spec["group_palette"] = list(group_palette)
        if group_palette_map is not None:
            spec["group_palette_map"] = {
                str(k): v for k, v in group_palette_map.items()
            }
        specs.append(spec)

        for value in individual_values:
            indiv = template.copy()
            title = indiv.pop("title")
            indiv["title"] = title % str(value)
            indiv["base_label"] = f"{indiv['key']}_vs_ppacket_n_tasks_{value}"
            indiv["plot_path"] = os.path.join(
                run_dir,
                f"{indiv['base_label']}.png",
            )
            indiv["group_column"] = None
            indiv.pop("group_palette", None)
            indiv.pop("group_labels", None)
            indiv.pop("group_palette_map", None)
            indiv["filter_column"] = "n_apps"
            indiv["filter_value"] = int(value)
            if group_palette_map is not None:
                lookup_key = str(int(value))
                color = group_palette_map.get(lookup_key)
                if color is not None:
                    indiv["color_override"] = color
            specs.append(indiv)

    return specs


def render_plot(
    spec: dict[str, Any],
    raw_data: pd.DataFrame,
    color,
    figsize: tuple[float, float],
    dpi: int,
    simulations_per_point: int,
) -> Optional[pd.DataFrame]:
    metric = spec["key"]
    plot_type = spec["plot_type"]
    group_column = spec.get("group_column")
    group_labels = spec.get("group_labels") or {}
    group_palette = spec.get("group_palette")
    palette_map = spec.get("group_palette_map") or {}
    color_override = spec.get("color_override")
    filter_column = spec.get("filter_column")
    filter_value = spec.get("filter_value")

    base_color = color_override if color_override is not None else color
    if base_color is None:
        base_color = sns.color_palette("tab10", 1)[0]

    cols = ["p_packet", metric]
    if group_column:
        cols.append(group_column)
    if filter_column:
        cols.append(filter_column)
    data = raw_data[cols].copy()

    if filter_column and filter_value is not None:
        data = data[data[filter_column].isin(np.atleast_1d(filter_value))]

    data = data.reset_index(drop=True)

    summary_df = None
    if plot_type == "line":
        keys = ["p_packet"] + ([group_column] if group_column else [])
        summary_df = (
            data.groupby(keys, as_index=False)
            .agg(
                mean=(metric, "mean"),
                std=(metric, lambda s: s.std(ddof=1)),
                count=(metric, "count"),
            )
            .sort_values(keys)
            .reset_index(drop=True)
        )
        if summary_df.empty:
            return summary_df

        summary_df["sem"] = (
            summary_df["std"] / np.sqrt(summary_df["count"])
            ).where(summary_df["count"] >= 2)
        summary_df["ci95"] = 1.96 * summary_df["sem"]
        summary_df["lower"] = summary_df["mean"] - summary_df["ci95"]
        summary_df["upper"] = summary_df["mean"] + summary_df["ci95"]
        summary_df[metric] = summary_df["mean"]
        if group_column:
            summary_df["group_display"] = (
                summary_df[group_column]
                .map(group_labels)
                .fillna(summary_df[group_column].astype(str))
            )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if plot_type == "line":
        if group_column:
            group_values = summary_df[group_column].dropna().unique().tolist()
            base_palette = (
                list(group_palette)
                if group_palette is not None
                else sns.color_palette("tab10", max(1, len(group_values)))
            )
            line_styles = spec.get("line_styles", ["-", "--", "-.", ":"])
            markers = spec.get("markers", ["o", "s", "D", "^", "v"])

            for i, gv in enumerate(group_values):
                gdf = summary_df[summary_df[group_column] == gv]
                if gdf.empty:
                    continue
                disp = gdf["group_display"].iloc[0]
                col = palette_map.get(
                    str(gv), base_palette[i % len(base_palette)]
                )
                ax.plot(
                    gdf["p_packet"],
                    gdf[metric],
                    marker=markers[i % len(markers)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2.0,
                    markersize=5.0,
                    color=col,
                    label=disp,
                )
                if not gdf[["lower", "upper"]].isnull().values.all():
                    ax.fill_between(
                        gdf["p_packet"],
                        gdf["lower"],
                        gdf["upper"],
                        color=col,
                        alpha=0.18,
                        linewidth=0,
                    )

            ax.legend(frameon=False, loc="best", fontsize=9)
        else:
            sns.lineplot(
                data=summary_df,
                x="p_packet",
                y=metric,
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                color=base_color,
                ax=ax,
            )
            if not summary_df[["lower", "upper"]].isnull().values.all():
                ax.fill_between(
                    summary_df["p_packet"],
                    summary_df["lower"],
                    summary_df["upper"],
                    color=base_color,
                    alpha=0.18,
                    linewidth=0,
                )

        ax.margins(x=0.02)

    else:
        labelled = data.assign(
            p_packet_label=data["p_packet"].map(lambda v: f"{v:g}")
        )
        order = [f"{v:g}" for v in sorted(data["p_packet"].unique())]

        if group_column:
            display_col = "group_display"
            labelled[display_col] = (
                labelled[group_column]
                .map(group_labels)
                .fillna(labelled[group_column].astype(str))
            )
            hue_order = labelled[display_col].unique().tolist()
            base_palette = (
                list(group_palette)
                if group_palette is not None
                else sns.color_palette("tab10", max(1, len(hue_order)))
            )

            pairs = (
                labelled[[group_column, display_col]]
                .dropna(subset=[group_column])
                .drop_duplicates()
            )
            palette = {}
            for disp in hue_order:
                raw = (
                    pairs.loc[pairs[display_col] == disp, group_column].iloc[0]
                    if not pairs.empty
                    else disp
                )
                palette[disp] = palette_map.get(str(raw))
            for i, disp in enumerate(hue_order):
                if palette.get(disp) is None:
                    palette[disp] = base_palette[i % len(base_palette)]

            sns.violinplot(
                data=labelled,
                x="p_packet_label",
                y=metric,
                hue=display_col,
                order=order,
                hue_order=hue_order,
                cut=0,
                inner="quartile",
                density_norm="count",
                palette=palette,
                linewidth=0.8,
                dodge=True,
                ax=ax,
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.legend(frameon=False, loc="best", fontsize=9)
        else:
            sns.violinplot(
                data=labelled,
                x="p_packet_label",
                y=metric,
                order=order,
                cut=0,
                inner="quartile",
                density_norm="count",
                color=base_color,
                linewidth=0.8,
                ax=ax,
            )

    if plot_type == "line" and summary_df is not None:
        if {"lower", "upper"}.issubset(summary_df.columns):
            vals = np.concatenate(
                [
                    summary_df[metric].to_numpy(),
                    summary_df["lower"].to_numpy(),
                    summary_df["upper"].to_numpy(),
                ]
            )
        else:
            vals = summary_df[metric].to_numpy()
    else:
        vals = data[metric].to_numpy()

    vals = pd.Series(vals).dropna().to_numpy()
    if vals.size:
        pad_frac = float(spec.get("pad_fraction", 0.05))
        lo, hi = float(vals.min()), float(vals.max())
        span = hi - lo
        if span == 0:
            span = max(abs(lo) * 0.05, 1e-6)
        lo -= span * pad_frac
        hi += span * pad_frac
        y_min = spec.get("ymin", lo)
        y_max = spec.get("ymax", hi)
        ax.set_ylim(y_min, y_max)

    if spec.get("percentage"):
        fmt = spec.get("percentage_format", "{:.1f}%")
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: fmt.format(v * 100.0)
        ))
    else:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: "0" if v == 0.0 else f"{v:.4g}")
        )

    if plot_type == "line":
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

    plot_path = spec["plot_path"]
    directory = os.path.dirname(plot_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
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
    keep_seed_outputs: bool = False,
    group_column: str | None = None,
    group_labels: dict | None = None,
    group_palette: Sequence[str] | None = None,
    n_apps_values: Sequence[int] | None = None,
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
        keep_seed_outputs (bool, optional): When ``False``, delete the
        per-seed result folders after each simulation finishes.
        group_column (str | None, optional): Column used to split lines/violins
        into groups. Defaults to ``"n_apps"`` when multiple ``n_apps`` values
        are provided.
        group_labels (dict | None, optional): Mapping from group values to
        display labels.
        group_palette (Sequence[str] | None, optional): Custom color palette
        for grouped plots.
        n_apps_values (Sequence[int] | None, optional): Specific application
        counts to sweep. When multiple values are provided, one line/violin
        will be produced per group.

    Returns:
        pd.DataFrame: DataFrame containing the aggregated simulation results.
    """
    n_apps_list = [int(v) for v in n_apps_values]
    default_kwargs = build_default_sim_args(config, simulation_kwargs)
    default_kwargs["n_apps"] = n_apps_list[0]

    n_tasks_label = str(n_apps_list[0]) if len(n_apps_list) == 1 else "varied"
    n_tasks_display = ", ".join(map(str, n_apps_list))
    plot_label = f"admission_rate_vs_ppacket_n_tasks_{n_tasks_label}"

    run_dir, timestamp = prepare_run_dir(
        output_dir,
        ppacket_values,
        keep_seed_outputs=keep_seed_outputs,
    )
    save_path = save_path or os.path.join(run_dir, f"{plot_label}.png")
    raw_csv_path = os.path.join(run_dir, f"{timestamp}_raw.csv")

    resolved_group_column = group_column or (
        "n_apps" if len(n_apps_list) > 1 else None
    )
    resolved_group_labels = group_labels or (
        {v: str(v) for v in n_apps_list}
        if resolved_group_column == "n_apps" else None
    )
    resolved_palette_map = None
    if resolved_group_column:
        keys = [
            str(v)
            for v in (
                n_apps_list
                if resolved_group_column == "n_apps"
                else resolved_group_labels.keys()
            )
        ]
        base = (
            list(group_palette)
            if group_palette
            else sns.color_palette("tab10", len(keys))
        )
        resolved_palette_map = {
            k: base[i % len(base)] for i, k in enumerate(keys)
        }

    metrics_to_plot = build_metric_specs(
        n_tasks_label=n_tasks_label,
        n_tasks_display=n_tasks_display,
        save_path=save_path,
        run_dir=run_dir,
        plot_label=plot_label,
        group_column=resolved_group_column,
        group_labels=resolved_group_labels,
        group_palette=group_palette,
        group_palette_map=resolved_palette_map,
        individual_n_apps=(n_apps_list if len(n_apps_list) > 1 else None),
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
        n_apps_values=n_apps_list,
        keep_seed_outputs=keep_seed_outputs,
    )

    tasks = build_tasks(
        ppacket_values=ppacket_values,
        simulations_per_point=simulations_per_point,
        seed_start=seed_start,
        run_dir=run_dir,
        default_kwargs=default_kwargs,
        n_apps_values=n_apps_list,
        keep_seed_outputs=keep_seed_outputs,
    )

    records = run_parallel_sims(
        tasks=tasks,
        max_workers=max_workers or os.cpu_count(),
        show_progress=show_progress,
    )

    results_df = pd.DataFrame(records)
    if not results_df.empty and {"completed_ratio", "p_packet"}.issubset(
        results_df.columns
    ):
        results_df["completed_ratio_delta"] = (
            results_df["completed_ratio"].astype(float)
            - results_df["p_packet"].astype(float)
        )
    results_df.to_csv(raw_csv_path, index=False)

    set_plot_theme(dpi)
    palette = sns.color_palette("tab10", len(metrics_to_plot))
    for i, spec in enumerate(metrics_to_plot):
        render_plot(
            spec=spec,
            raw_data=results_df,
            color=palette[i % len(palette)],
            figsize=figsize,
            dpi=dpi,
            simulations_per_point=simulations_per_point,
        )

    return results_df, raw_csv_path


"""
# Example usage of the plot_metrics_vs_ppacket function.
if __name__ == "__main__":
    sweep_values = np.round(np.linspace(0.1, 0.9, 9), 2).tolist()
    n_apps_values = [1, 25, 30, 35, 40, 45, 50]

    plot_metrics_vs_ppacket(
        ppacket_values=sweep_values,
        simulations_per_point=3000,
        simulation_kwargs={
            "inst_range": (100, 100),
            "epr_range": (10, 10),
            "period_range": (0.05, 0.05),
            "hyperperiod_cycles": 10e6,
            "memory_lifetime": 2000,
            "p_swap": 0.95,
            "scheduler": "dynamic"
        },
        config="configurations/network/Garr201201.gml",
        n_apps_values=n_apps_values,
    )
"""
