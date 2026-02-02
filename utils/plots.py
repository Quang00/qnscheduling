import os
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
            "key": "avg_waiting_time",
            "ylabel": "Average waiting time (s)",
        },
        {
            "key": "avg_link_utilization",
            "ylabel": "Average link utilization (%)",
            "percentage": True,
        },
        {
            "key": "p95_link_utilization",
            "ylabel": "95th percentile link utilization (%)",
            "percentage": True,
        },
        {
            "key": "p95_link_avg_wait",
            "ylabel": "95th percentile link average wait (s)",
        },
        {
            "key": "deadline_miss_rate",
            "ylabel": "Deadline miss rate (%)",
            "percentage": True,
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
        summary_df["group_display"] = (
            summary_df[group_column]
            .map(group_labels)
            .fillna(summary_df[group_column].astype(str))
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
    raw_csv_path: str,
    save_path: str | None = None,
    figsize: tuple[float, float] = (7, 4.5),
    dpi: int = 600,
    group_column: str | None = None,
    gp_labels: dict | None = None,
    group_palette: Sequence[str] | None = None,
    p_packet_values: Sequence[float] | None = None,
    scheduler: str | None = None,
    create_individual: bool = False,
) -> pd.DataFrame:
    results_df = pd.read_csv(raw_csv_path)
    run_dir = os.path.dirname(raw_csv_path) or "."

    if p_packet_values is None:
        pp_list = results_df["p_packet"].dropna().unique().tolist()
        pp_list.sort()
    else:
        pp_list = [float(v) for v in p_packet_values]

    pp_label = str(pp_list[0]) if len(pp_list) == 1 else "varied"
    scheduler_value = scheduler or "dynamic"
    sch_suffix = scheduler_value.title()

    if not save_path:
        save_path = os.path.join(
            run_dir,
            f"admission_rate_vs_load_p_packet_{pp_label}_{sch_suffix}.png",
        )
    if not group_column and len(pp_list) > 1:
        group_column = "p_packet"
    if not gp_labels and group_column == "p_packet":
        gp_labels = {v: f"$p_{{\\mathrm{{packet}}}}={v}$" for v in pp_list}

    plt_map = None
    if group_column:
        keys = (
            pp_list if group_column == "p_packet" else list(gp_labels.keys())
        )
        plt = (
            list(group_palette)
            if group_palette
            else sns.color_palette("tab10", len(keys))
        )
        plt_map = {str(k): plt[i % len(plt)] for i, k in enumerate(keys)}

    metrics_to_plot = build_metric_specs(
        v_label=pp_label,
        save_path=save_path,
        run_dir=run_dir,
        sch_suffix=sch_suffix,
        x="n_apps",
        group_column=group_column,
        group_labels=gp_labels,
        group_palette=group_palette,
        group_palette_map=plt_map,
        individual_values=pp_list if len(pp_list) > 1 else None,
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


"""
if __name__ == "__main__":
    from utils.parallel_simulations import run_ppacket_sweep_to_csv

    sweep_values = np.round(np.linspace(0.1, 0.9, 9), 2).tolist()
    n_apps_values = [10, 20, 40, 60, 80, 100]

    _, raw_csv_path = run_ppacket_sweep_to_csv(
        ppacket_values=sweep_values,
        simulations_per_point=2,
        simulation_kwargs={
            "inst_range": (10, 10),
            "epr_range": (2, 2),
            "period_range": (1, 1),
            "memory": 100,
            "p_swap": 0.6,
            "p_gen": 1e-3,
            "fidelity_range": (0.6, 0.8),
            "time_slot_duration": 1e-4,
            "routing": "capacity",
            "capacity_threshold": 0.8,
            "scheduler": "dynamic",
        },
        config="configurations/network/Dumbbell.gml",
        n_apps_values=n_apps_values,
        keep_seed_outputs=False,
    )

    plot_metrics_vs_ppacket(
        raw_csv_path=raw_csv_path,
        n_apps_values=n_apps_values,
        scheduler="dynamic",
    )

    plot_metrics_vs_load(
        raw_csv_path=raw_csv_path,
        p_packet_values=sweep_values,
        scheduler="dynamic",
    )
"""
