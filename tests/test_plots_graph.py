import matplotlib
import matplotlib.pyplot as plt

from utils.plots_graph import plot_graph_from_gml

matplotlib.use("Agg")


def test_plot_graph_from_gml() -> None:
    plt.show = lambda: None
    plot_graph_from_gml("configurations/network/basic/Chain.gml")
    plt.close("all")


if __name__ == "__main__":
    test_plot_graph_from_gml()
