import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext

from scheduling.pga import duration_pga


def plot_pga_durations_vs_n_swapping(
    p_packet: float = 0.2,
    k: int = 2,
    configs: list = None,
) -> None:
    """Plot the duration of a PGA (Packet Generation Attempt) vs. the number of
    entanglement swappings, in one combined figure and in separate figures for
    each memory lifetime.

    Args:
        p_packet (float, optional):  Probability of a packet being generated.
        k (int, optional): Number of successes (number of EPR pairs generated).
        configs (list, optional): List of tuples (label, p_swap, memory).
    """
    if configs is None:
        configs = [
            (r"$p_{swap}=0.5, \tau_{mem}=0$", 0.5, 0),
            (r"$p_{swap}=0.95, \tau_{mem}=0$", 0.95, 0),
            (r"$p_{swap}=0.5, \tau_{mem}=500\ ms$", 0.5, 500),
            (r"$p_{swap}=0.95, \tau_{mem}=500\ ms$", 0.95, 500),
        ]

    n_swaps = list(range(0, 11))
    durations = {}

    for cfg in configs:
        label, p_swap, memory = cfg
        dur_us = [
            duration_pga(p_packet, k, n, memory, p_swap=p_swap)
            for n in n_swaps
        ]
        dur_s = [d * 1e-6 for d in dur_us]
        durations[cfg] = dur_s

    palette = sns.color_palette(n_colors=len(configs))
    color_map = {cfg: palette[i] for i, cfg in enumerate(configs)}
    plt.figure(figsize=(6, 4), dpi=300)

    for cfg in configs:
        label, p_swap, memory = cfg
        sns.lineplot(
            x=n_swaps,
            y=durations[cfg],
            marker="o",
            label=label,
            color=color_map[cfg]
        )

    ax = plt.gca()
    ax.set_xlabel("Number of Entanglement Swappings")
    ax.set_ylabel("Duration (s)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    plt.title("PGA Duration vs Number of Entanglement Swappings")
    plt.tight_layout()
    plt.savefig("docs/pga_durations_vs_n_swapping_all.png", format="png")
    plt.close()

    unique_memories = sorted({memory for (_, _, memory) in configs})
    for memory in unique_memories:
        plt.figure(figsize=(6, 4), dpi=300)
        cfgs_for_mem = [cfg for cfg in configs if cfg[2] == memory]

        for cfg in cfgs_for_mem:
            label, p_swap, _ = cfg
            sns.lineplot(
                x=n_swaps,
                y=durations[cfg],
                marker="o",
                label=label,
                color=color_map[cfg],
            )

        ax = plt.gca()
        ax.set_xlabel("Number of Entanglement Swappings")
        ax.set_ylabel("Duration (s)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

        plt.title(f"PGA Duration vs Number of Swappings (τ_mem = {memory})")
        plt.tight_layout()
        plt.savefig(f"docs/pga_durations_vs_n_swapping_mem_{memory}.png",
                    format="png")
        plt.close()
