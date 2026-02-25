import networkx as nx
import numpy as np

from utils.helper import compute_edge_fidelities


def generate_waxman_graph(
    n: int = 48,
    alpha: float = 0.2,
    beta: float = 0.6,
    rng: np.random.Generator | None = None,
    max_retries: int = 5000,
    max_avg_degree: float = 5.0,
    max_hops: int = 8,
) -> tuple[list, list, dict, float]:
    """Generates a Waxman graph with constraints on connectivity,
    average degree, and diameter.

    Args:
        n (int, optional): Number of nodes in the graph.
        alpha (float, optional:): Controls the influence of distance on edge
        probability. Higher alpha values lead to fewer edges between distant
        nodes.
        beta (float, optional): Controls the overall density of the graph.
        Higher beta values lead to more edges.
        rng (np.random.Generator | None, optional): Random number generator.
        max_retries (int, optional): Maximum number of retries to generate a
        valid graph (connected, with average degree <= max_avg_degree,
        and diameter <= max_hops).
        max_avg_degree (float, optional): Maximum average degree of the graph.
        max_hops (int, optional): Maximum diameter of the graph. Defaults to 8.

    Returns:
        tuple[list, list, dict, float]: A tuple containing:
            - List of nodes in the graph.
            - List of edges in the graph.
            - Dictionary mapping edges to their fidelities.
            - Average degree of the graph.
    """
    G = None
    for _ in range(max_retries):
        G = nx.waxman_graph(n, alpha=alpha, beta=beta, seed=rng)
        if not nx.is_connected(G):
            continue
        avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
        if avg_deg > max_avg_degree:
            continue
        if nx.diameter(G) > max_hops:
            continue
        break
    else:
        return [], [], {}, 0.0

    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda edge: (str(edge[0]), str(edge[1])))
    pos = nx.get_node_attributes(G, "pos")
    distances = {}
    for u, v in G.edges():
        sq_d = (pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2
        d = sq_d**0.5
        distances[(u, v)] = d
        G[u][v]["dist"] = d
    fidelites = compute_edge_fidelities(G, distances)

    return nodes, edges, fidelites, avg_deg


def fat_tree(
    k: int = 4,
    qpu_edge_dist=1.0,
    edge_aggregate_dist=3.0,
    aggregate_core_dist=6.0,
    F_min: float = 0.8,
) -> tuple[list, list, dict, list]:
    """Generates a fat-tree topology with k pods. Each pod contains k/2 edge
    switches and k/2 aggregate switches. The core layer has (k/2)^2 core
    switches. Each edge switch connects to k/2 hosts (QPUs). See:
    https://arxiv.org/pdf/2601.01353

    Args:
        k (int, optional): Number of pods.
        qpu_edge_dist (float, optional): Distance between QPUs and edge
        switches.
        edge_aggregate_dist (float, optional): Distance between edge and
        aggregate switches.
        aggregate_core_dist (float, optional): Distance between aggregate and
        core switches.
        F_min (float, optional): Minimum initial fidelity for edges.

    Returns:
        tuple[list, list, dict, list]: A tuple containing:
            - List of nodes in the graph.
            - List of edges in the graph.
            - Dictionary mapping edges to their fidelities.
            - List of QPUs (hosts) in the graph.
    """
    G = nx.Graph()
    qpus = []
    pods = k
    core_switches = (k // 2) ** 2
    agg_per_pod = k // 2
    edge_per_pod = k // 2
    qpus_per_edge = k // 2

    # Core
    cores = [f"core_{i}" for i in range(core_switches)]
    G.add_nodes_from(cores, layer=0)
    for p in range(pods):
        aggregate = [f"pod{p}_agg_{i}" for i in range(agg_per_pod)]
        edge = [f"pod{p}_edge_{i}" for i in range(edge_per_pod)]
        G.add_nodes_from(aggregate, layer=1, pod=p)
        G.add_nodes_from(edge, layer=2, pod=p)

        # core <-> aggregate
        for i, a in enumerate(aggregate):
            for j in range(k // 2):
                core_i = i * (k // 2) + j
                G.add_edge(a, cores[core_i], dist=float(aggregate_core_dist))

        # aggregate <-> edge
        for e in edge:
            for a in aggregate:
                G.add_edge(e, a, dist=float(edge_aggregate_dist))

        # edge <-> qpu
        for e_i, e in enumerate(edge):
            for qpu in range(qpus_per_edge):
                qpu = f"pod{p}_qpu_{e_i}_{qpu}"
                qpus.append(qpu)
                G.add_node(qpu, layer=3, pod=p)
                G.add_edge(e, qpu, dist=float(qpu_edge_dist))

    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda edge: (str(edge[0]), str(edge[1])))
    distances = {(u, v): float(G.edges[u, v]["dist"]) for (u, v) in edges}
    fidelities = compute_edge_fidelities(G, distances, F_min=F_min)

    return nodes, edges, fidelities, qpus
