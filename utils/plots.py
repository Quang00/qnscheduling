import os
from datetime import datetime
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def set_plot_theme(dpi: int) -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi


def build_metric_specs(
    v_label: str,
    save_path: str,
    run_dir: str,
    sch_suffix: str,
    x: str = "p_packet",
    group_column: str | None = None,
    group_labels: dict | None = None,
    group_palette: Sequence[str] | None = None,
    group_palette_map: dict[str, Any] | None = None,
    individual_values: Sequence[int | float] | None = None,
    group_line_styles: dict[str, str] | None = None,
    create_individual: bool = False,
) -> list[dict[str, Any]]:
    metric_templates = [
        {
            "key": "admission_rate",
            "ylabel": "Admission rate (%)",
            "percentage": True,
        },
        {
            "key": "makespan",
            "ylabel": "Completion time (s)",
            "yscale": "log",
        },
        {
            "key": "throughput",
            "ylabel": "Throughput (completed PGAs/s)",
        },
        {
            "key": "completed_ratio",
            "ylabel": "Completion ratio (%)",
            "percentage": True,
        },
        {
            "key": "avg_link_utilization",
            "ylabel": "Average link utilization (%)",
            "percentage": True,
        },
        {
            "key": "p90_link_utilization",
            "ylabel": "90th percentile link utilization (%)",
            "percentage": True,
        },
        {
            "key": "p95_link_utilization",
            "ylabel": "95th percentile link utilization (%)",
            "percentage": True,
        },
        {
            "key": "avg_waiting_time",
            "ylabel": "Average waiting time (s)",
        },
        {
            "key": "p90_link_avg_wait",
            "ylabel": "90th percentile link average wait (s)",
        },
        {
            "key": "p95_link_avg_wait",
            "ylabel": "95th percentile link average wait (s)",
        },
        {
            "key": "avg_queue_length",
            "ylabel": "Average queue length",
        },
        {
            "key": "p90_avg_queue_length",
            "ylabel": "90th percentile average queue length",
        },
        {
            "key": "p95_avg_queue_length",
            "ylabel": "95th percentile average queue length",
        },
        {
            "key": "drop_ratio",
            "ylabel": "Drop ratio (%)",
            "percentage": True,
        },
        {
            "key": "avg_hops",
            "ylabel": "Average number of hops",
        },
        {
            "key": "avg_e2e_fidelity",
            "ylabel": "Average E2E fidelity",
        },
        {
            "key": "avg_pga_duration",
            "ylabel": "Average PGA duration (s)",
        },
        {
            "key": "avg_defer_per_pga",
            "ylabel": "Average deferrals per PGA",
        },
        {
            "key": "avg_retry_per_pga",
            "ylabel": "Average retries per PGA",
        },
    ]

    specs = []
    indiv_values = list(individual_values or [])

    for template in metric_templates:
        spec = template.copy()
        if spec["key"] == "admission_rate":
            spec["plot_path"] = save_path
        else:
            file = f"{spec['key']}_vs_{x}_vary_{v_label}_{sch_suffix}.png"
            spec["plot_path"] = os.path.join(run_dir, file)
        spec["x_var"] = x
        spec["group_column"] = group_column
        spec["group_labels"] = group_labels
        spec["group_palette"] = group_palette
        spec["group_palette_map"] = group_palette_map
        spec["group_line_styles"] = group_line_styles
        specs.append(spec)

        if create_individual:
            for v in indiv_values:
                i = template.copy()
                file = f"{i['key']}_vs_{x}_value_{v}_{sch_suffix}.png"
                i["plot_path"] = os.path.join(run_dir, file)
                i["x_var"] = x
                i["filter_column"] = "p_packet" if x == "n_apps" else "n_apps"
                i["filter_value"] = float(v) if x == "n_apps" else int(v)
                i["color_override"] = group_palette_map.get(str(v))
                specs.append(i)
    return specs


def render_plot(
    spec: dict[str, Any],
    raw_data: pd.DataFrame,
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    metric = spec["key"]
    x_var = spec.get("x_var", "p_packet")
    group_column = spec.get("group_column")
    group_labels = spec.get("group_labels") or {}
    group_palette = spec.get("group_palette")
    palette_map = spec.get("group_palette_map") or {}
    group_line_styles = spec.get("group_line_styles") or {}
    color_override = spec.get("color_override")
    filter_column = spec.get("filter_column")
    filter_value = spec.get("filter_value")
    base_color = color_override or sns.color_palette("tab10", 1)[0]
    cols = [x_var, metric]

    if group_column:
        cols.append(group_column)
    if filter_column:
        cols.append(filter_column)
    data = raw_data[cols]
    if filter_column and filter_value is not None:
        data = data[data[filter_column].isin(np.atleast_1d(filter_value))]

    keys = [x_var] + ([group_column] if group_column else [])
    summary_df = data.groupby(keys, as_index=False).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        count=(metric, "count"),
    )
    sem = summary_df["std"] / np.sqrt(summary_df["count"].clip(lower=1))
    sem = sem.where(summary_df["count"] >= 2)
    ci95 = 1.96 * sem
    summary_df["lower"] = summary_df["mean"] - ci95
    summary_df["upper"] = summary_df["mean"] + ci95
    x_values = sorted(summary_df[x_var].unique())

    use_categorical = x_var == "n_apps"
    if use_categorical:
        x_map = {val: idx for idx, val in enumerate(x_values)}
        summary_df["x_plot"] = summary_df[x_var].map(x_map)
    else:
        summary_df["x_plot"] = summary_df[x_var]

    if group_column:
        summary_df["group_display"] = summary_df[group_column].apply(
            lambda x: group_labels.get(x, group_labels.get(str(x), str(x)))
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if group_column:
        group_values = summary_df[group_column].dropna().unique().tolist()
        base_palette = (
            list(group_palette)
            if group_palette
            else sns.color_palette("tab10", max(1, len(group_values)))
        )
        markers = ["o", "s", "D", "^", "v"]

        for i, gv in enumerate(group_values):
            gdf = summary_df[summary_df[group_column] == gv]
            disp = gdf["group_display"].iloc[0]
            col = palette_map.get(str(gv), base_palette[i % len(base_palette)])
            style = group_line_styles.get(str(gv), "-")
            ax.plot(
                gdf["x_plot"],
                gdf["mean"],
                marker=markers[i % len(markers)],
                linestyle=style,
                color=col,
                label=disp,
            )
            ax.fill_between(
                gdf["x_plot"],
                gdf["lower"],
                gdf["upper"],
                color=col,
                alpha=0.2,
            )

        ax.legend()
    else:
        ax.plot(
            summary_df["x_plot"],
            summary_df["mean"],
            marker="o",
            linestyle="-",
            color=base_color,
        )
        ax.fill_between(
            summary_df["x_plot"],
            summary_df["lower"],
            summary_df["upper"],
            color=base_color,
            alpha=0.2,
        )

    if spec.get("yscale"):
        if summary_df["mean"].gt(0).any():
            ax.set_yscale(spec["yscale"])

    ymin = spec.get("ymin")
    ymax = spec.get("ymax")
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if spec.get("percentage"):
        fmt = spec.get("percentage_format", "{:.1f}%")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: fmt.format(v * 100.0))
        )

    xlabel_map = {
        "p_packet": r"$p_{\mathrm{packet}}$",
        "load": "Load",
        "n_apps": "Number of applications",
    }
    ax.set_xlabel(xlabel_map.get(x_var, x_var))
    ax.set_ylabel(spec["ylabel"])

    if use_categorical:
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([f"{int(v)}" for v in x_values])
    else:
        ax.set_xticks(x_values)
    ax.margins(x=0, y=0.05)

    plot_path = spec["plot_path"]
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_vs_ppacket(
    raw_csv_path: str,
    save_path: str | None = None,
    figsize: tuple[float, float] = (7, 4.5),
    dpi: int = 600,
    gp_column: str | None = None,
    gp_labels: dict | None = None,
    group_palette: Sequence[str] | None = None,
    n_apps_values: Sequence[int] | None = None,
    scheduler: str | None = None,
    create_individual: bool = False,
) -> pd.DataFrame:
    results_df = pd.read_csv(raw_csv_path)
    run_dir = os.path.dirname(raw_csv_path) or "."

    if n_apps_values is None:
        apps_list = results_df["n_apps"].dropna().astype(int).unique().tolist()
        apps_list.sort()
    else:
        apps_list = [int(v) for v in n_apps_values]

    n_apps_label = str(apps_list[0]) if len(apps_list) == 1 else "varied"
    scheduler_value = scheduler or "dynamic"
    scheduler_suffix = scheduler_value.title()

    if not save_path:
        save_path = os.path.join(
            run_dir,
            f"admission_rate_vs_ppacket_{n_apps_label}_{scheduler_suffix}.png",
        )
    if not gp_column and len(apps_list) > 1:
        gp_column = "n_apps"
    if not gp_labels and gp_column == "n_apps":
        gp_labels = {v: f"{v} apps" for v in apps_list}

    plt_map = None
    if gp_column:
        keys = apps_list if gp_column == "n_apps" else list(gp_labels.keys())
        plt = (
            list(group_palette)
            if group_palette
            else sns.color_palette("tab10", len(keys))
        )
        plt_map = {str(k): plt[i % len(plt)] for i, k in enumerate(keys)}

    metrics_to_plot = build_metric_specs(
        v_label=n_apps_label,
        save_path=save_path,
        run_dir=run_dir,
        sch_suffix=scheduler_suffix,
        x="p_packet",
        group_column=gp_column,
        group_labels=gp_labels,
        group_palette=group_palette,
        group_palette_map=plt_map,
        individual_values=apps_list if len(apps_list) > 1 else None,
        create_individual=create_individual,
    )

    set_plot_theme(dpi)
    for spec in metrics_to_plot:
        render_plot(
            spec=spec,
            raw_data=results_df,
            figsize=figsize,
            dpi=dpi,
        )

    return results_df


def plot_metrics_vs_load(
    path: str | Sequence[str],
    save_path: str | None = None,
    figsize: tuple[float, float] = (7, 4.5),
    dpi: int = 600,
    group_column: str | None = None,
    gp_labels: dict | None = None,
    group_palette: Sequence[str] | None = None,
    p_packet_values: Sequence[float] | None = None,
    scheduler: str | None = None,
    create_individual: bool = False,
    multi: bool = False,
) -> pd.DataFrame:
    """Example usage:
    df = plot_metrics_vs_load(
        path=[
            "1.csv",
            "2.csv",
            "3.csv",
            "4.csv",
            "5.csv",
        ],
        multi=True,
        gp_labels={
            "1": "Shortest path",
            "2": "Highest fidelity",
            "3": "Capacity 0.8",
            "4": "Least capacity",
            "5": "Smallest bottleneck",
        },
    )
    """
    if multi:
        paths = path if isinstance(path, (list, tuple)) else [path]
        dfs = [pd.read_csv(p) for p in paths]
        results_df = pd.concat(dfs, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join("results", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        if p_packet_values is None:
            val_list = results_df["scenario"].dropna().unique().tolist()
            val_list.sort()
        else:
            val_list = list(p_packet_values)
        value_label = str(val_list[0]) if len(val_list) == 1 else "varied"
        def_gp = "scenario"
        file_prefix = "scenario"
        dft_labels = {v: f"{v}" for v in val_list}
    else:
        results_df = pd.read_csv(path)
        run_dir = os.path.dirname(path) or "."
        if p_packet_values is None:
            val_list = results_df["p_packet"].dropna().unique().tolist()
            val_list.sort()
        else:
            val_list = [float(v) for v in p_packet_values]
        value_label = str(val_list[0]) if len(val_list) == 1 else "varied"
        def_gp = "p_packet"
        file_prefix = "p_packet"
        dft_labels = {v: f"$p_{{\\mathrm{{packet}}}}={v}$" for v in val_list}

    scheduler_value = scheduler or "dynamic"
    sch_suffix = scheduler_value.title()

    if not save_path:
        save_path = os.path.join(
            run_dir,
            f"admission_rate_{file_prefix}_{value_label}_{sch_suffix}.png",
        )
    if not group_column and len(val_list) > 1:
        group_column = def_gp
    if not gp_labels and group_column == def_gp:
        gp_labels = dft_labels

    plt_map = None
    if group_column:
        keys = val_list if group_column == def_gp else list(gp_labels.keys())
        plt = (
            list(group_palette)
            if group_palette
            else sns.color_palette("tab10", len(keys))
        )
        plt_map = {str(k): plt[i % len(plt)] for i, k in enumerate(keys)}

    metrics_to_plot = build_metric_specs(
        v_label=value_label,
        save_path=save_path,
        run_dir=run_dir,
        sch_suffix=sch_suffix,
        x="n_apps",
        group_column=group_column,
        group_labels=gp_labels,
        group_palette=group_palette,
        group_palette_map=plt_map,
        individual_values=val_list if len(val_list) > 1 else None,
        create_individual=create_individual,
    )

    set_plot_theme(dpi)
    for spec in metrics_to_plot:
        render_plot(
            spec=spec,
            raw_data=results_df,
            figsize=figsize,
            dpi=dpi,
        )

    return results_df
