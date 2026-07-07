import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter


# =============================================================================
# Parallelization
# =============================================================================
def parallelizable_tasks(
    paths_for_each_apps: dict[str, List[str] | None],
) -> dict[str, set[str]]:
    """Find parallelizable applications based on shared links of a
    quantum network.

    Args:
        paths_for_each_apps (dict[str, List[str]]): A dictionary where keys are
            application names and values are list of nodes describing the route
            used to run the applications, e.g. on a linear chain Alice-Rob-Bob:

                {
                    'A': ['Alice', 'Rob'],
                    'B': ['Rob', 'Bob'],
                }

    Returns:
        dict[str, set[str]]: A dictionary where keys are application names and
            values are sets of applications that can run in parallel with key
            application, based on shared links. From the example above, the
            output would be A and B can run in parallel since they do not share
            any links:

                {
                    'A': {'B'},
                    'B': {'A'},
                }
    """
    G = nx.Graph()
    conflicts = defaultdict(set)

    # Build conflict graph using undirected links along each path
    for app, v in paths_for_each_apps.items():
        G.add_node(app)
        if not v or len(v) < 2:
            continue
        edges_on_path = {
            tuple(sorted((u, v))) for u, v in zip(v[:-1], v[1:], strict=False)
        }
        for edge in edges_on_path:
            for other_app in conflicts[edge]:
                G.add_edge(app, other_app)
            conflicts[edge].add(app)

    g_complement = nx.complement(G)
    parallelizable_applications = {
        app: set(g_complement.neighbors(app)) for app in g_complement.nodes()
    }

    return parallelizable_applications


# =============================================================================
# Helper simulation functions
# =============================================================================
def app_params_sim(
    paths: dict[str, list[str]],
    app_specs: dict[str, dict[str, Any]],
    p_packet: float,
    memory: int,
    p_swap: float,
    time_slot_duration: float,
    coherence: float = 0.020,
) -> dict[str, dict[str, float | int]]:
    """Prepare application parameters for simulation.

    Args:
        paths (dict[str, list[str]]): Paths for each application in the
        network.
        app_specs (dict[str, dict[str, Any]]): Application metadata produced by
        ``generate_n_apps`` containing network endpoints and requirements.
        p_packet (float): Probability of a packet being generated.
        memory (int): Number of independent link-generation trials per slot.
        p_swap (float): Probability of swapping an EPR pair in a single trial.
        time_slot_duration (float): Duration of a time slot in seconds.
        coherence (float): Coherence time in seconds of a generated pair.

    Returns:
        dict[str, dict[str, float | int]]: Mapping of application name to the
        parameters required by the simulator when instantiating PGAs.
    """
    sim_params = {}
    for key in paths.keys():
        spec = app_specs[key]
        sim_params[key] = {
            "p_packet": p_packet,
            "memory": memory,
            "coherence": coherence,
            "p_swap": p_swap,
            "epr_pairs": int(spec["epr"]),
            "slot_duration": time_slot_duration,
        }
    return sim_params


def build_default_sim_args(config: str, args: dict | None) -> dict:
    default_args = {
        "config": config,
        "inst_range": 100,
        "epr_range": (2, 2),
        "deadline_range": (1, 1),
        "memory": 1,
        "p_swap": 0.5,
        "routing": "smallest",
        "time_slot_duration": 1e-4,
        "graph": "gml",
        "provisioning": False,
        "full_dynamic": True,
        "static_routing_mode": False,
    }
    if args:
        default_args.update(args)
    return default_args


# =============================================================================
# Tracking
# =============================================================================
def compute_link_utilization(
    link_busy: Dict[Tuple[str, str], float],
    min_start: float,
    max_completion: float,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    if not link_busy:
        return {}

    horizon = 0.0
    if (
        np.isfinite(min_start)
        and np.isfinite(max_completion)
        and max_completion > min_start
    ):
        horizon = max_completion - min_start

    if horizon > 0.0:
        return {
            link: {
                "busy_time": busy,
                "utilization": busy / horizon,
            }
            for link, busy in link_busy.items()
        }

    return {
        link: {"busy_time": busy, "utilization": 0.0}
        for link, busy in link_busy.items()
    }


def track_link_waiting(
    waiting_time: float,
    wait_acc: Dict[Tuple[str, str], Dict[str, float]],
    blocking_links: List[Tuple[str, str]] | None = None,
) -> None:
    """Track waiting time statistics per link.

    Args:
        waiting_time (float): Wait incurred by a single deferral.
        wait_acc (Dict[Tuple[str, str], Dict[str, float]]): Accumulator
        for waiting time statistics per link.
        blocking_links (List[Tuple[str, str]] | None): The specific link(s)
        that caused the waiting (with maximum busy time). If provided, waiting
        time is distributed equally among these links.

    Note:
        ``block_events`` counts blocking episodes (one per deferral per link),
        not distinct PGAs, so a single PGA deferred repeatedly contributes
        multiple counts.
    """
    wait = max(0.0, float(waiting_time))
    if wait <= 0.0:
        return
    if blocking_links is None or len(blocking_links) == 0:
        return

    links_to_update = blocking_links
    w = wait / len(blocking_links)

    for link in links_to_update:
        link_acc = wait_acc.setdefault(
            link,
            {
                "total_waiting_time": 0.0,
                "block_events": 0,
            },
        )
        link_acc["total_waiting_time"] = link_acc["total_waiting_time"] + w
        link_acc["block_events"] = link_acc["block_events"] + 1


# =============================================================================
# Results saving and summary
# =============================================================================
def save_results(
    df: pd.DataFrame,
    pga_names: List[str],
    pga_release_times: Dict[str, float],
    app_specs: Dict[str, Dict[str, Any]],
    n_edges: int,
    durations: Dict[str, float] | None = None,
    pga_network_paths: Dict[str, List[str]] | None = None,
    link_utilization: Dict[Tuple[str, str], Dict[str, float]] | None = None,
    link_waiting: Dict[Tuple[str, str], Dict[str, float | int]] | None = None,
    admitted_apps: int | None = None,
    total_apps: int | None = None,
    app_e2e_fidelities: Dict[str, float] | None = None,
    single_path_share: float = float("nan"),
    two_path_share: float = float("nan"),
    app_request_rows: List[Dict[str, Any]] | None = None,
    avg_deg: float = float("nan"),
    output_dir: str = "results",
    save_csv: bool = True,
    verbose: bool = True,
    routing_decision_cpt: int | None = None,
    routing_decision_runtime: float | None = None,
    warmup: float | None = None,
    end_time: float | None = None,
    defer_counts: Dict[str, int] | None = None,
    multi_path_apps: Iterable[str] | None = None,
) -> Dict[str, float]:
    """Save the results of PGA scheduling and execution to a CSV file and print
    a summary of the results.

    Args:
        df (DataFrame): DataFrame containing PGA results with columns:
            - pga: PGA identifier
            - arrival_time: Time when the PGA arrived
            - start_time: Time when the PGA started execution
            - burst_time: Total time required for the PGA to complete
            - completion_time: Time when the PGA completed execution
            - turnaround_time: Total time from arrival to completion
            - waiting_time: Total time the PGA waited before execution
            - status: Status of the PGA (e.g., "completed", "failed")
            - deadline: Deadline for the PGA (if applicable)
            - src_node: Source node of the PGA
            - dst_node: Destination node of the PGA
            - instances: Number of instances for the PGA
            - epr_pairs: Number of EPR pairs for the PGA
        pga_names (List): List of all PGA names that should be present in the
            results.
        pga_release_times (Dict): Dictionary mapping PGA names to their
            relative release times, used to fill in missing PGAs.
        app_specs (Dict): Metadata for each application including endpoints,
            instances, requested EPR pairs, and deadline budget.
        n_edges (int): Number of edges in the network graph.
        durations (Dict | None): Optional mapping of deterministic PGA
            durations per application.
        pga_network_paths (Dict | None): Length of network paths per
            application.
        link_utilization (Dict): Dictionary mapping links to busy time and
            utilization metrics.
        link_waiting (Dict | None): Dictionary mapping links to waiting
            metrics (total waiting time and number of PGAs that waited).
        output_dir (str): Directory where the results CSV file will be saved.
        save_csv (bool): Whether to save results to CSV files.
        verbose (bool): Whether to print summary statistics to stdout.

    Returns:
        Dict[str, float]: Dictionary containing summary metrics including
            admission_rate, makespan, throughput, completion ratios, and
            utilization statistics.
    """
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)

    missing = set(pga_names) - set(df["pga"])
    if missing:
        filler_rows = []
        for pga in missing:
            task = re.sub(r"\d+$", "", pga)
            filler_rows.append(
                {
                    "pga": pga,
                    "arrival_time": pga_release_times.get(pga, np.nan),
                    "start_time": np.nan,
                    "burst_time": np.nan,
                    "completion_time": np.nan,
                    "turnaround_time": np.nan,
                    "waiting_time": np.nan,
                    "status": "missing",
                }
            )
        df = pd.concat([df, pd.DataFrame(filler_rows)], ignore_index=True)

    df["task"] = df["pga"].astype(str).str.replace(r"\d+$", "", regex=True)
    if multi_path_apps is not None:
        df["multi_path"] = df["task"].isin(set(multi_path_apps))
    app_names = list(app_specs.keys())
    has_per_row_routing = all(
        c in df.columns for c in ("hops", "pga_duration", "e2e_fidelity")
    )

    if not save_csv:
        _keep = [
            "pga", "status", "waiting_time", "turnaround_time",
            "burst_time", "arrival_time", "completion_time", "task",
            "multi_path",
        ]
        if has_per_row_routing:
            _keep += [
                c for c in (
                    "hops", "e2e_fidelity", "pga_duration",
                    "routing_efficiency",
                )
                if c in df.columns
            ]
        df = df[[c for c in _keep if c in df.columns]]

    if pga_network_paths:
        path_length = {
            app: max(0, len(path) - 1)
            for app, path in pga_network_paths.items()
            if path is not None
        }
    else:
        path_length = {}
    params = pd.DataFrame(
        {
            "task": app_names,
            "src_node": [app_specs[a]["src"] for a in app_names],
            "dst_node": [app_specs[a]["dst"] for a in app_names],
            "instances": [int(app_specs[a]["instances"]) for a in app_names],
            "pairs_requested": [int(app_specs[a]["epr"]) for a in app_names],
            "hops": [path_length.get(a, np.nan) for a in app_names],
            "pga_duration": [
                float(durations[a]) if durations and a in durations else np.nan
                for a in app_names
            ],
            "e2e_fidelity": [
                (
                    float(app_e2e_fidelities[a])
                    if app_e2e_fidelities and a in app_e2e_fidelities
                    else float("nan")
                )
                for a in app_names
            ],
        }
    )

    if has_per_row_routing:
        merge_cols = [
            c for c in params.columns
            if c not in ("hops", "pga_duration", "e2e_fidelity")
        ]
        df = df.merge(params[merge_cols], on="task", how="left")
    else:
        df = df.merge(params, on="task", how="left")
    df = df.sort_values(by="completion_time").reset_index(drop=True)
    if warmup is not None:
        df = df[df["arrival_time"] >= warmup].reset_index(drop=True)
    if end_time is not None:
        df = df[df["arrival_time"] <= end_time].reset_index(drop=True)

    if end_time is not None and warmup is not None:
        makespan = end_time - warmup
    else:
        makespan = float("nan")

    if save_csv:
        csv_path = os.path.join(output_dir, "pga_results.parquet")
        df.to_parquet(csv_path, index=False)

        if verbose:
            print("\n=== Preview PGA Results ===")
            print(df.head(20).to_string(index=False))

    if save_csv and verbose and app_request_rows:
        app_request_df = pd.DataFrame(app_request_rows)
        app_request_path = os.path.join(output_dir, "app_request.csv")
        app_request_df.to_csv(app_request_path, index=False)

    avg_link_utilization = float("nan")
    p90_link_utilization = float("nan")
    p95_link_utilization = float("nan")
    total_busy_time = float("nan")
    top5_busy_share = float("nan")
    top10_busy_share = float("nan")
    p90_link_avg_wait = float("nan")
    p95_link_avg_wait = float("nan")
    avg_queue_length = float("nan")
    p90_avg_queue_length = float("nan")
    p95_avg_queue_length = float("nan")
    blocking_prob = float("nan")
    link_waiting_path = None
    if link_utilization:
        link_util_rows = [
            {
                "link": f"{min(a, b)}-{max(a, b)}",
                "busy_time": metrics.get("busy_time", float("nan")),
                "utilization": metrics.get("utilization", float("nan")),
            }
            for (a, b), metrics in link_utilization.items()
        ]
        lk_ut_df = (
            pd.DataFrame(link_util_rows)
            .sort_values("utilization", ascending=False)
            .reset_index(drop=True)
        )

        if makespan and makespan > 0:
            lk_ut_df["utilization"] = lk_ut_df["busy_time"] / makespan
        lk_ut_df = (
            lk_ut_df
            .sort_values("utilization", ascending=False)
            .reset_index(drop=True)
        )

        busy_time_sum = lk_ut_df["busy_time"].sum()
        avg_link_utilization = float((busy_time_sum / makespan) / n_edges)
        p90_link_utilization = float(lk_ut_df["utilization"].quantile(0.9))
        p95_link_utilization = float(lk_ut_df["utilization"].quantile(0.95))
        total_busy_time = busy_time_sum

        n_links = len(lk_ut_df)
        top5_n = int(np.ceil(0.05 * n_links))
        top10_n = int(np.ceil(0.10 * n_links))
        if busy_time_sum > 0 and n_links > 0:
            top5_busy_share = float(
                lk_ut_df["busy_time"].nlargest(top5_n).sum() / busy_time_sum
            )
            top10_busy_share = float(
                lk_ut_df["busy_time"].nlargest(top10_n).sum() / busy_time_sum
            )

        if save_csv:
            link_util_path = os.path.join(output_dir, "link_utilization.csv")
            lk_ut_df.to_csv(link_util_path, index=False)

            if verbose:
                print("\n=== Link Utilization ===")
                print(lk_ut_df.to_string(index=False))

    if link_waiting:
        waiting_rows = [
            {
                "link": f"{min(a, b)}-{max(a, b)}",
                "total_waiting_time": waiting.get(
                    "total_waiting_time", float("nan")
                ),
                "block_events": waiting.get("block_events", 0),
                "acquisitions": waiting.get("acquisitions", 0),
            }
            for (a, b), waiting in link_waiting.items()
        ]
        for row in waiting_rows:
            w = row["block_events"]
            total_wait = row["total_waiting_time"]
            accesses = w + row["acquisitions"]
            row["avg_wait"] = total_wait / w if w and w > 0 else 0.0
            row["avg_queue_length"] = (
                total_wait / makespan if makespan and makespan > 0 else 0.0
            )
            row["blocking_prob"] = w / accesses if accesses > 0 else 0.0
        total_blocks = sum(r["block_events"] for r in waiting_rows)
        total_accesses = total_blocks + sum(
            r["acquisitions"] for r in waiting_rows
        )
        blocking_prob = (
            total_blocks / total_accesses if total_accesses > 0 else 0.0
        )
        waiting_df = (
            pd.DataFrame(waiting_rows)
            .sort_values("total_waiting_time", ascending=False)
            .reset_index(drop=True)
        )

        avg_wait_series = pd.to_numeric(
            waiting_df.get("avg_wait"), errors="coerce"
        ).dropna()
        if not avg_wait_series.empty:
            p90_link_avg_wait = float(avg_wait_series.quantile(0.9))
            p95_link_avg_wait = float(avg_wait_series.quantile(0.95))

        avg_queue_series = pd.to_numeric(
            waiting_df.get("avg_queue_length"), errors="coerce"
        ).dropna()
        if not avg_queue_series.empty:
            avg_queue_length = float(avg_queue_series.mean())
            p90_avg_queue_length = float(avg_queue_series.quantile(0.9))
            p95_avg_queue_length = float(avg_queue_series.quantile(0.95))
        if save_csv:
            link_waiting_path = os.path.join(output_dir, "link_waiting.csv")
            waiting_df.to_csv(link_waiting_path, index=False)

            if verbose:
                print("\n=== Link Waiting ===")
                print(waiting_df.to_string(index=False))

    admission_rate = float("nan")
    if admitted_apps is not None and total_apps is not None and total_apps > 0:
        admission_rate = float(admitted_apps) / float(total_apps)

    scopes = [
        ("all", df, app_specs, durations, app_e2e_fidelities, "summary.csv"),
    ]
    if multi_path_apps is not None:
        multi_set = set(multi_path_apps)
        sub_df = (
            df[df["multi_path"]].reset_index(drop=True)
            if "multi_path" in df.columns
            else df.iloc[0:0]
        )
        scopes.append(
            (
                "multi_path",
                sub_df,
                {a: s for a, s in app_specs.items() if a in multi_set},
                (
                    {a: v for a, v in durations.items() if a in multi_set}
                    if durations
                    else durations
                ),
                {
                    a: v
                    for a, v in (app_e2e_fidelities or {}).items()
                    if a in multi_set
                },
                "summary_multipath.csv",
            )
        )

    summaries = {}
    for scope_name, sdf, sspecs, sdurations, se2e, fname in scopes:
        tot_reqs = sdf["pga"].nunique()
        final_status = (
            sdf.assign(_order_time=sdf["completion_time"].fillna(-np.inf))
            .sort_values(["pga", "_order_time"])
            .groupby("pga", as_index=False)
            .tail(1)
        )
        completed_total = int((final_status["status"] == "completed").sum())
        drop_total = int((final_status["status"] == "drop").sum())
        del final_status
        failed_total = tot_reqs - completed_total - drop_total

        executed_burst = pd.to_numeric(sdf["burst_time"], errors="coerce")
        executed_total = int(
            sdf.loc[
                executed_burst.notna() & (executed_burst > 0), "pga"
            ].nunique()
        )

        total = len(sdf)
        throughput = completed_total / makespan
        pga_duration = list(sdurations.values()) if sdurations else []
        served_df = sdf[sdf["status"].isin(("completed", "failed"))]
        avg_wait = (
            served_df["waiting_time"].mean()
            if not served_df.empty
            else float("nan")
        )
        max_wait = (
            served_df["waiting_time"].max()
            if not served_df.empty
            else float("nan")
        )
        avg_turnaround = (
            served_df["turnaround_time"].mean()
            if not served_df.empty
            else float("nan")
        )
        max_turnaround = (
            served_df["turnaround_time"].max()
            if not served_df.empty
            else float("nan")
        )
        p95_wait = (
            float(served_df["waiting_time"].quantile(0.95))
            if not served_df.empty
            else float("nan")
        )
        p95_turnaround = (
            float(served_df["turnaround_time"].quantile(0.95))
            if not served_df.empty
            else float("nan")
        )
        p95_burst = (
            float(served_df["burst_time"].quantile(0.95))
            if not served_df.empty
            else float("nan")
        )
        pga_d = (
            np.mean(pga_duration) if len(pga_duration) > 0 else float("nan")
        )
        completed_ratio = (
            completed_total / tot_reqs if tot_reqs else float("nan")
        )
        drop_ratio = drop_total / tot_reqs if tot_reqs else float("nan")
        failed_ratio = failed_total / tot_reqs if tot_reqs else float("nan")
        if defer_counts is not None:
            defer_count = sum(
                v
                for k, v in defer_counts.items()
                if re.sub(r"\d+$", "", k) in sspecs
            )
        else:
            defer_count = int((sdf["status"] == "defer").sum())
        retry_count = int((sdf["status"] == "retry").sum())
        avg_defer_per_pga = (
            defer_count / tot_reqs if tot_reqs else float("nan")
        )
        avg_retry_per_pga = (
            retry_count / tot_reqs if tot_reqs else float("nan")
        )
        avg_burst_time = (
            sdf["burst_time"].mean()
            if not sdf.empty
            else float("nan")
        )
        total_burst_time = float(executed_burst.fillna(0).clip(lower=0).sum())
        avg_active_pgas = (
            total_burst_time / makespan if makespan and makespan > 0
            else float("nan")
        )
        fastest_path_rate = float("nan")
        if has_per_row_routing:
            cols = [
                c for c in (
                    "hops", "e2e_fidelity", "pga_duration",
                )
                if c in sdf.columns
            ]
            per_pga = sdf.loc[sdf["status"] == "completed", cols].mean()
            avg_hops = float(per_pga.get("hops", float("nan")))
            avg_e2e_fidelity = float(per_pga.get("e2e_fidelity", float("nan")))
            pga_d = float(per_pga.get("pga_duration", float("nan")))
            if "routing_efficiency" in sdf.columns:
                eff = pd.to_numeric(
                    sdf["routing_efficiency"], errors="coerce"
                )
                chose_fastest = (eff >= 1.0 - 1e-9).astype(float)
                chose_fastest[eff.isna()] = np.nan
                fastest_path_rate = float(chose_fastest.mean())
        else:
            sparams = params[params["task"].isin(sspecs)]
            avg_hops = (
                sparams["hops"].mean()
                if not sparams.empty
                else float("nan")
            )
            e2e_fidelity_values = [
                v
                for v in (se2e or {}).values()
                if v is not None and not np.isnan(v)
            ]
            avg_e2e_fidelity = (
                float(np.mean(e2e_fidelity_values))
                if e2e_fidelity_values
                else float("nan")
            )
        admitted_min_fidelities = [
            sspecs[app].get("min_fidelity", float("nan"))
            for app in sspecs.keys()
        ]
        avg_min_fidelity = (
            float(np.mean(admitted_min_fidelities))
            if admitted_min_fidelities
            else float("nan")
        )

        expected_tasks = sorted(sspecs.keys())
        per_task = (
            sdf.groupby(["task", "status"])
            .size()
            .unstack(fill_value=0)
            .reindex(expected_tasks, fill_value=0)
        )
        released_per_task = (
            sdf.groupby("task")["pga"]
            .nunique()
            .reindex(expected_tasks, fill_value=0)
        )

        for col in ["completed", "failed"]:
            if col not in per_task.columns:
                per_task[col] = 0

        tasks_sorted = sorted(per_task.index, key=lambda x: (len(x), x))
        served_tasks = {
            task for task in expected_tasks
            if int(released_per_task.get(task, 0))
            == int(sspecs[task]["instances"] if task in sspecs else 0)
        }
        served_count = len(served_tasks)
        app_throughput = (
            served_count / makespan if makespan else float("nan")
        )
        served_agg = (
            sdf[sdf["task"].isin(served_tasks)]
            .groupby("task", observed=True)
            .agg(
                first=("arrival_time", "min"),
                last=("completion_time", "max"),
            )
        )
        service_times = [
            row["last"] - row["first"]
            for _, row in served_agg.iterrows()
            if pd.notna(row["first"]) and pd.notna(row["last"])
        ]
        avg_service_time = (
            float(np.mean(service_times)) if service_times else float("nan")
        )
        n_apps_in_window = int(sdf["task"].nunique())
        service_ratio = (
            served_count / n_apps_in_window
            if n_apps_in_window > 0 else float("nan")
        )
        fairness = float("nan")
        completion_ratios = []
        for task in expected_tasks:
            task_total = per_task.loc[task].sum()
            task_completed = per_task.loc[task].get("completed", 0)
            if task_total > 0:
                completion_ratios.append(
                    float(task_completed) / float(task_total)
                )
        if len(completion_ratios) > 0:
            ratios = np.array(completion_ratios)
            n = len(ratios)
            sum_ratios = float(np.sum(ratios))
            sum_ratios_sq = float(np.sum(ratios**2))

            if sum_ratios_sq > 0:
                fairness = (sum_ratios**2) / (n * sum_ratios_sq)
            else:
                fairness = 1.0

        if verbose and scope_name == "all":
            print("\n=== Summary ===")
            print(f"Total PGAs       : {total}")
            for task in tasks_sorted:
                row = per_task.loc[task]
                completed = int(row.get("completed", 0))
                failed = int(row.get("failed", 0))
                instances = (
                    int(sspecs[task]["instances"])
                    if task in sspecs else 0
                )
                released = int(released_per_task.get(task, 0))
                served = released == instances
                print(
                    f"    {task:<4} released: {released}/{instances},"
                    f" completed: {completed}, failed: {failed},"
                    f" served: {served}"
                )

            print(f"Admission rate   : {admission_rate:.4f}")
            print(f"Completion time  : {makespan:.4f}")
            print(f"Executed PGAs    : {executed_total}")
            print(f"Throughput       : {throughput:.4f} completed PGAs/s")
            print(f"App throughput   : {app_throughput:.4f} served apps/s")
            print(f"Service ratio    : {service_ratio:.4f}")
            print(f"Completion ratio : {completed_ratio:.4f}")
            print(f"Failed ratio     : {failed_ratio:.4f}")
            print(f"Drop ratio       : {drop_ratio:.4f}")
            print(f"Avg defer per PGA: {avg_defer_per_pga:.4f}")
            print(f"Avg retry per PGA: {avg_retry_per_pga:.4f}")
            print(f"Avg burst time   : {avg_burst_time:.4f}")
            print(f"Avg active PGAs  : {avg_active_pgas:.4f}")
            print(f"Avg service time : {avg_service_time:.4f}")
            print(f"Avg waiting time : {avg_wait:.4f}")
            print(f"Max waiting time : {max_wait:.4f}")
            print(f"P95 waiting time : {p95_wait:.4f}")
            print(f"P95 burst time   : {p95_burst:.4f}")
            print(f"P90 link avg_wait : {p90_link_avg_wait:.4f}")
            print(f"P95 link avg_wait : {p95_link_avg_wait:.4f}")
            print(f"Avg queue length : {avg_queue_length:.4f}")
            print(f"P90 avg_queue_length : {p90_avg_queue_length:.4f}")
            print(f"P95 avg_queue_length : {p95_avg_queue_length:.4f}")
            print(f"Blocking prob    : {blocking_prob:.4f}")
            print(f"Avg turnaround   : {avg_turnaround:.4f}")
            print(f"Max turnaround   : {max_turnaround:.4f}")
            print(f"P95 turnaround   : {p95_turnaround:.4f}")
            print(f"Avg hops         : {avg_hops:.4f}")
            print(f"Avg min fidelity : {avg_min_fidelity:.4f}")
            print(f"Avg E2E fidelity : {avg_e2e_fidelity:.4f}")
            print(f"Single-path share: {single_path_share:.2f}")
            print(f"Two-path share   : {two_path_share:.2f}")
            print(f"Avg PGA duration : {pga_d:.4f}")
            print(f"Fastest-path rate: {fastest_path_rate:.4f}")
            print(f"Total busy time  : {total_busy_time:.4f}")
            print(f"Avg link utilization : {avg_link_utilization:.4f}")
            print(f"P90 link utilization : {p90_link_utilization:.4f}")
            print(f"P95 link utilization : {p95_link_utilization:.4f}")
            print(f"Top-5% busy-time share  : {top5_busy_share:.4f}")
            print(f"Top-10% busy-time share : {top10_busy_share:.4f}")
            print(f"Avg degree       : {avg_deg:.4f}")
            print(f"Fairness         : {fairness:.4f}")
            print(f"Routing decisions count: {routing_decision_cpt}")
            print(f"Routing decisions runtime: {routing_decision_runtime}")

        summary_metrics = {
            "admission_rate": float(admission_rate),
            "makespan": float(makespan),
            "executed_pgas": int(executed_total),
            "throughput": float(throughput),
            "app_throughput": float(app_throughput),
            "service_ratio": float(service_ratio),
            "completed_ratio": float(completed_ratio),
            "failed_ratio": float(failed_ratio),
            "drop_ratio": float(drop_ratio),
            "avg_defer_per_pga": float(avg_defer_per_pga),
            "avg_retry_per_pga": float(avg_retry_per_pga),
            "avg_burst_time": float(avg_burst_time),
            "avg_active_pgas": float(avg_active_pgas),
            "avg_waiting_time": float(avg_wait),
            "max_waiting_time": float(max_wait),
            "p95_waiting_time": float(p95_wait),
            "avg_turnaround_time": float(avg_turnaround),
            "avg_service_time": float(avg_service_time),
            "max_turnaround_time": float(max_turnaround),
            "p95_turnaround_time": float(p95_turnaround),
            "p95_burst_time": float(p95_burst),
            "avg_hops": float(avg_hops),
            "avg_min_fidelity": float(avg_min_fidelity),
            "avg_e2e_fidelity": float(avg_e2e_fidelity),
            "single_path_share_pct": float(single_path_share),
            "two_path_share_pct": float(two_path_share),
            "avg_pga_duration": float(pga_d),
            "fastest_path_rate": float(fastest_path_rate),
            "total_busy_time": float(total_busy_time),
            "avg_link_utilization": float(avg_link_utilization),
            "p90_link_utilization": float(p90_link_utilization),
            "p95_link_utilization": float(p95_link_utilization),
            "top5_busy_share": float(top5_busy_share),
            "top10_busy_share": float(top10_busy_share),
            "p90_link_avg_wait": float(p90_link_avg_wait),
            "p95_link_avg_wait": float(p95_link_avg_wait),
            "avg_queue_length": float(avg_queue_length),
            "p90_avg_queue_length": float(p90_avg_queue_length),
            "p95_avg_queue_length": float(p95_avg_queue_length),
            "blocking_prob": float(blocking_prob),
            "avg_deg": float(avg_deg),
            "fairness": float(fairness),
            "routing_decision_count": (
                int(routing_decision_cpt)
                if routing_decision_cpt is not None
                else None
            ),
            "routing_decision_runtime": (
                float(routing_decision_runtime)
                if routing_decision_runtime is not None
                else None
            ),
        }
        summaries[scope_name] = summary_metrics

        if save_csv:
            pd.DataFrame([summary_metrics]).to_csv(
                os.path.join(output_dir, fname), index=False
            )

    result = dict(summaries["all"])
    if "multi_path" in summaries:
        result.update(
            {f"multipath_{k}": v for k, v in summaries["multi_path"].items()}
        )
    return result


# =============================================================================
# Application and network generation
# =============================================================================
def compute_edge_fidelities(
    G: nx.Graph,
    distances: Dict[Tuple, float],
    T_coh: float = 0.02,
    c_fiber: float = 2e5,
    F0: float = 0.95,
) -> Dict[Tuple, float]:
    fidelities = {}

    for u, v, data in G.edges(data=True):
        L = float(distances.get((u, v), data.get("dist", 0.0)))
        t_herald = L / c_fiber
        f = np.exp(-t_herald / T_coh) * (F0 - 0.25) + 0.25
        data["fidelity"] = f
        fidelities[(u, v)] = f

    return fidelities


def compute_edge_probs(
    G: nx.Graph,
    distances: Dict[Tuple, float],
    attenuation: float = 0.2,
    coupling_efficiency: float = 0.4,
) -> Dict[Tuple, float]:
    probs = {}
    L_attenuation = 10.0 / (attenuation * np.log(10.0))

    for u, v, data in G.edges(data=True):
        L = float(distances.get((u, v), data.get("dist", 0.0)))
        p = 0.5 * coupling_efficiency**2 * np.exp(-L / L_attenuation)
        data["p_gen"] = p
        probs[(min(u, v), max(u, v))] = p

    return probs


def gml_data(
    gml_file: str,
) -> Tuple[list, list, dict[tuple, float], dict, float]:
    """Extracts nodes, edges, distances, fidelities, and diameter from a GML
    file.

    Args:
        gml_file (str): Path to the GML file.

    Returns:
        nodes (list): List of nodes.
        edges (list): List of edges (source, target).
        distances (dict[tuple, float]): Dict mapping directed edges to
        distances.
        fidelities (dict[tuple, float]): Dict mapping directed edges to
        fidelities.
        rates (dict[tuple, float]): Dict mapping directed edges to
        rates.
        diameter (float): Diameter of the graph.
    """
    G = nx.read_gml(gml_file)

    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda edge: (str(edge[0]), str(edge[1])))
    distances = {
        (u, v): float(data.get("dist", 0.0))
        for u, v, data in G.edges(data=True)
    }
    fidelities = compute_edge_fidelities(G, distances)
    rates = compute_edge_probs(G, distances)
    diameter = float(nx.diameter(G))

    return nodes, edges, distances, fidelities, rates, diameter


def generate_n_apps(
    end_nodes: list[str],
    bounds: dict[tuple, tuple],
    n_apps: int,
    inst_range: int,
    epr_range: tuple[int, int],
    deadline_range: tuple[float, float],
    rng: np.random.Generator,
    manual_pairs: list[tuple[str, str]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Generates a specified number of applications with random parameters.

    Args:
        end_nodes (list): List of end nodes in the network.
        n_apps (int): Number of applications to generate.
        inst_range (int): Poisson lambda for the number of instances
        per application.
        epr_range (tuple[int, int]): Range (min, max) for the number of EPR
        pairs for each application.
        deadline_range (tuple[float, float]): Range (min, max) for the deadline
        budget multiplier k (k > 1) of each application. The absolute budget is
        derived later as deadline_budget = k * fastest-path PGA duration, so
        every app is feasible on its best path.
        rng (np.random.Generator): Random number generator for reproducibility.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of application name to its metadata,
        including endpoints, number of instances, requested EPR pairs, and
        deadline budget multiplier.
    """
    apps = {}
    feasible = []
    floor = 0.51
    for i in range(len(end_nodes)):
        for j in range(i + 1, len(end_nodes)):
            src, dst = end_nodes[i], end_nodes[j]
            _, max_f = fidelity_bounds(bounds, src, dst)
            if max_f > floor:
                feasible.append((src, dst, float(max_f)))

    pair_idx = rng.integers(0, len(feasible), size=n_apps)

    for k in range(n_apps):
        if manual_pairs:
            src, dst = manual_pairs[k % len(manual_pairs)]
            _, max_f = fidelity_bounds(bounds, src, dst)
        else:
            src, dst, max_f = feasible[int(pair_idx[k])]
        name_app = get_column_letter(k + 1)
        apps[name_app] = {
            "src": src,
            "dst": dst,
            "instances": max(
                1, min(3 * inst_range, int(rng.poisson(lam=inst_range)))
            ),
            "epr": int(rng.integers(epr_range[0], epr_range[1] + 1)),
            "deadline_budget": float(
                rng.uniform(deadline_range[0], deadline_range[1])
            ),
            "min_fidelity": float(rng.uniform(floor, max_f)),
        }

    return apps


# =============================================================================
# Directory management
# =============================================================================
def ppacket_dirname(value: float) -> str:
    label = f"{value:.6f}".rstrip("0").rstrip(".")
    if not label:
        label = "0"
    return f"ppacket_{label.replace('.', '_')}"


def prepare_run_dir(
    output_dir: str,
    ppacket_values: Iterable[float],
    keep_seed_outputs: bool = True,
) -> tuple[str, str]:
    base_output = output_dir or "results"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_output, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    if keep_seed_outputs:
        for p_val in ppacket_values:
            subdir = os.path.join(run_dir, ppacket_dirname(p_val))
            os.makedirs(subdir, exist_ok=True)
    return run_dir, timestamp


# =============================================================================
# Retrievial
# =============================================================================
def fidelity_bounds(
    bounds: Dict[Tuple[str, str], Tuple[float, float]], src: str, dst: str
) -> Tuple[float, float]:
    return bounds[(src, dst) if src < dst else (dst, src)]


def all_simple_paths(
    paths: Dict[Tuple[str, str], List[Tuple[float, Tuple[str, ...]]]],
    src: str,
    dst: str,
) -> List[Tuple[float, Tuple[str, ...]]]:
    return paths.get((src, dst) if src < dst else (dst, src), [])


def count_edge_disjoint_paths(
    feasible_paths: List[Tuple[float, Tuple[str, ...]]],
) -> int:
    if not feasible_paths:
        return 0
    src, dst = feasible_paths[0][1][0], feasible_paths[0][1][-1]
    edges = (
        (u, v)
        for _, path in feasible_paths
        for u, v in zip(path[:-1], path[1:], strict=False)
    )
    graph = nx.Graph(edges)
    n_disjoint = sum(1 for _ in nx.edge_disjoint_paths(graph, src, dst))
    return n_disjoint
