import matplotlib.pyplot as plt
from scheduling.pga import duration_pga
from matplotlib.ticker import LogFormatterMathtext
import seaborn as sns


def plot_pga_durations_vs_n_swapping(
    p_packet: float = 0.2,
    k: int = 2,
    configs: list = None,
) -> None:
    """Plot the duration of a PGA (Packet Generation Attempt) against the number of entanglement swappings.

    Args:
        p_packet (float, optional):  Probability of a packet being generated.
        k (int, optional): Number of successes (number of EPR pairs generated).
        configs (list, optional): List of tuples containing configuration labels, p_swap, and memory lifetime.
    """
    if configs is None:
        configs = [
            (r"$p_{swap}=0.5, \tau_{mem}=0$", 0.5, 0),
            (r"$p_{swap}=0.95, \tau_{mem}=0$", 0.95, 0),
            (r"$p_{swap}=0.5, \tau_{mem}=500\ ms$", 0.5, 500),
            (r"$p_{swap}=0.95, \tau_{mem}=500\ ms$", 0.95, 500),
        ]

    plt.figure(figsize=(6, 4), dpi=300)
    n_swaps = list(range(0, 11))

    for label, p_swap, memory in configs:
        durations_us = [
            duration_pga(p_packet, k, n_swap, memory, p_swap=p_swap)
            for n_swap in n_swaps
        ]
        print(f"Durations for {label}: {durations_us}")
        durations_s = [d * 1e-6 for d in durations_us]
        print(f"Durations in seconds for {label}: {durations_s}")
        sns.lineplot(x=n_swaps, y=durations_s, marker="o", label=label)

    ax = plt.gca()
    ax.set_xlabel("Number of Entanglement Swappings")
    ax.set_ylabel("Duration (s)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    plt.title("PGA Duration vs Number of Entanglement Swappings")
    plt.tight_layout()
    plt.savefig("docs/pga_durations_vs_n_swapping.png", format="png")
