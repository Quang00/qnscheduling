import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext

from scheduling.pga import duration_pga


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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")

    nx.draw(G, pos, ax=ax, with_labels=True, node_size=80, font_size=5)

    ax.set_title(name)
    ax.axis("off")

    plt.show()
