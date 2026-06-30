import os
from datetime import datetime
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


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
    overlay_multipath: bool = False,
) -> list[dict[str, Any]]:
    metric_templates = [
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
            "key": "drop_ratio",
            "ylabel": "Drop ratio (%)",
            "percentage": True,
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
            "key": "avg_waiting_time",
            "ylabel": "Average waiting time (s)",
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
            "key": "p95_avg_queue_length",
            "ylabel": "95th percentile average queue length",
        },
        {
            "key": "blocking_prob",
            "ylabel": "Blocking probability",
        },
        {
            "key": "avg_turnaround_time",
            "ylabel": "Average turnaround time (s)",
        },
        {
            "key": "p95_waiting_time",
            "ylabel": "95th percentile waiting time (s)",
        },
        {
            "key": "p95_turnaround_time",
            "ylabel": "95th percentile turnaround time (s)",
        },
        {
            "key": "p95_burst_time",
            "ylabel": "95th percentile burst time (s)",
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
            "key": "avg_active_pgas",
            "ylabel": "Average active PGAs",
        },
        {
            "key": "fastest_path_rate",
            "ylabel": "Fastest-path rate",
            "percentage": True,
        },
        {
            "key": "top5_busy_share",
            "ylabel": "Top 5 busy time-share (%)",
            "percentage": True,
        },
        {
            "key": "fairness",
            "ylabel": "Fairness",
        },
        {
            "key": "routing_decision_count",
            "ylabel": "Routing decisions",
            "yscale": "log",
        },
        {
            "key": "routing_decision_runtime",
            "ylabel": "Routing computation time (s)",
            "yscale": "log",
        }
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
        spec["overlay_multipath"] = overlay_multipath
        specs.append(spec)

        if create_individual:
            for v in indiv_values:
                i = template.copy()
                file = f"{i['key']}_vs_{x}_value_{v}_{sch_suffix}.png"
                i["plot_path"] = os.path.join(run_dir, file)
                i["x_var"] = x
                i["filter_column"] = (
                    "p_packet" if x == "arrival_rate" else "arrival_rate"
                )
                i["filter_value"] = float(v)
                i["color_override"] = group_palette_map.get(str(v))
                i["overlay_multipath"] = overlay_multipath
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

    mp_metric = f"multipath_{metric}"
    overlay = (
        bool(spec.get("overlay_multipath"))
        and not metric.startswith("multipath_")
        and mp_metric in raw_data.columns
    )

    cols = [x_var, metric]
    if overlay:
        cols.append(mp_metric)
    if group_column:
        cols.append(group_column)
    if filter_column:
        cols.append(filter_column)
    data = raw_data[cols]
    if filter_column and filter_value is not None:
        data = data[data[filter_column].isin(np.atleast_1d(filter_value))]

    x_values = sorted(data[x_var].unique())
    use_categorical = x_var not in ("p_packet", "load", "arrival_rate")
    x_map = {val: idx for idx, val in enumerate(x_values)}

    def _summarize(metric_col: str) -> pd.DataFrame:
        keys = [x_var] + ([group_column] if group_column else [])
        s = data.groupby(keys, as_index=False).agg(
            mean=(metric_col, "mean"),
            std=(metric_col, "std"),
            count=(metric_col, "count"),
        )
        sem = s["std"] / np.sqrt(s["count"].clip(lower=1))
        sem = sem.where(s["count"] >= 2)
        ci95 = 1.96 * sem
        s["lower"] = s["mean"] - ci95
        s["upper"] = s["mean"] + ci95
        if use_categorical:
            s["x_plot"] = s[x_var].map(x_map)
        else:
            s["x_plot"] = s[x_var]
        if group_column:
            s["group_display"] = s[group_column].apply(
                lambda x: group_labels.get(
                    x, group_labels.get(str(x), str(x))
                )
            )
        return s

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _draw(sdf: pd.DataFrame, linestyle: str, scope_label: str) -> None:
        if group_column:
            group_values = sdf[group_column].dropna().unique().tolist()
            base_palette = (
                list(group_palette)
                if group_palette
                else sns.color_palette("tab10", max(1, len(group_values)))
            )
            markers = ["o", "s", "D", "^", "v"]
            for i, gv in enumerate(group_values):
                gdf = sdf[sdf[group_column] == gv]
                disp = gdf["group_display"].iloc[0]
                col = palette_map.get(
                    str(gv), base_palette[i % len(base_palette)]
                )
                style = (
                    linestyle
                    if scope_label
                    else group_line_styles.get(str(gv), "-")
                )
                label = f"{disp} ({scope_label})" if scope_label else disp
                ax.plot(
                    gdf["x_plot"],
                    gdf["mean"],
                    marker=markers[i % len(markers)],
                    linestyle=style,
                    color=col,
                    label=label,
                )
                ax.fill_between(
                    gdf["x_plot"],
                    gdf["lower"],
                    gdf["upper"],
                    color=col,
                    alpha=0.2,
                )
        else:
            ax.plot(
                sdf["x_plot"],
                sdf["mean"],
                marker="o",
                linestyle=linestyle,
                color=base_color,
                label=scope_label or None,
            )
            ax.fill_between(
                sdf["x_plot"],
                sdf["lower"],
                sdf["upper"],
                color=base_color,
                alpha=0.2,
            )

    summary_df = _summarize(metric)
    _draw(summary_df, "-", "All-paths" if overlay else "")
    if overlay:
        _draw(_summarize(mp_metric), "--", "Multi-path")

    if group_column or overlay:
        ax.legend()

    if spec.get("yscale"):
        if summary_df["mean"].gt(0).any():
            ax.set_yscale(spec["yscale"])

    ymin = spec.get("ymin")
    ymax = spec.get("ymax")
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if spec.get("percentage"):
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=2)
        )

    xlabel_map = {
        "p_packet": r"$p_{\mathrm{packet}}$",
        "load": "Load",
        "n_apps": "Number of applications",
        "arrival_rate": r"Arrival rate $\lambda$",
        "inst_range": (
            "Average number of instances per application"
        ),
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
    overlay_multipath: bool = False,
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
        overlay_multipath=overlay_multipath,
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
    x_var: str = "arrival_rate",
    overlay_multipath: bool = False,
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
            "1": "Precomputed",
            "2": "Proactive",
            "3": "Hybrid",
            "4": "Reactive (wc)",
            "5": "Reactive (nwc)",
        },
        overlay_multipath=True,
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
        if x_var == "inst_range":
            val_list = (
                results_df["inst_range"].dropna().astype(int).unique().tolist()
            )
            val_list.sort()
            def_gp = "arrival_rate"
            file_prefix = "inst_range"
            dft_labels = {v: fr"$\lambda={v}$" for v in val_list}
        else:
            if p_packet_values is None:
                val_list = results_df["p_packet"].dropna().unique().tolist()
                val_list.sort()
            else:
                val_list = [float(v) for v in p_packet_values]
            def_gp = "p_packet"
            file_prefix = "p_packet"
            dft_labels = {
                v: f"$p_{{\\mathrm{{packet}}}}={v}$" for v in val_list
            }
        value_label = str(val_list[0]) if len(val_list) == 1 else "varied"

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
        x=x_var,
        group_column=group_column,
        group_labels=gp_labels,
        group_palette=group_palette,
        group_palette_map=plt_map,
        individual_values=val_list if len(val_list) > 1 else None,
        create_individual=create_individual,
        overlay_multipath=overlay_multipath,
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
