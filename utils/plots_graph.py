import os

import matplotlib.pyplot as plt
import networkx as nx


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


if __name__ == "__main__":
    gml_path = input("Enter the path to the GML file: ")

    if not os.path.exists(gml_path):
        print(f"Error: File '{gml_path}' does not exist.")
    elif not os.path.isfile(gml_path):
        print(f"Error: '{gml_path}' is not a file.")
    else:
        plot_graph_from_gml(gml_path)
