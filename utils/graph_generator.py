import numpy as np
import networkx as nx

from utils.helper import compute_edge_fidelities


def generate_waxman_graph(
    n: int = 48,
    alpha: float = 0.4,
    beta: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[list, list, dict, dict, list, float, float]:
    G = nx.waxman_graph(n, alpha=alpha, beta=beta, seed=rng)

    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda edge: (str(edge[0]), str(edge[1])))
    pos = nx.get_node_attributes(G, "pos")
    distances = {}
    for u, v in G.edges():
        sq_d = (pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2
        d = sq_d ** 0.5
        distances[(u, v)] = d
        G[u][v]["dist"] = d
    fidelites = compute_edge_fidelities(G, distances)
    end_nodes = sorted(nx.k_core(G).nodes(), key=str)
    avg_fid = sum(fidelites.values()) / len(fidelites) if fidelites else 0
    avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()

    return nodes, edges, distances, fidelites, end_nodes, avg_fid, avg_deg
