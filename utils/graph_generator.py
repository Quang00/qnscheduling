import networkx as nx
import numpy as np

from utils.helper import compute_edge_fidelities, compute_edge_probs


def generate_waxman_graph(
    n: int = 48,
    alpha: float = 0.2,
    beta: float = 0.3,
    rng: np.random.Generator | None = None,
    max_retries: int = 5000,
    max_avg_degree: float = 3.0,
    max_hops: int = 8,
) -> tuple[list, list, dict, dict, float, float]:
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
        tuple[list, list, dict, dict, float, float]: A tuple containing:
            - List of nodes in the graph.
            - List of edges in the graph.
            - Dictionary mapping edges to their fidelities.
            - Dictionary mapping edges to their p_gen rates.
            - Average degree of the graph.
            - Diameter of the graph.
    """
    G = None
    diameter = float("nan")
    for _ in range(max_retries):
        G = nx.waxman_graph(n, alpha=alpha, beta=beta, seed=rng)
        if not nx.is_connected(G):
            continue
        avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
        if avg_deg > max_avg_degree:
            continue
        diameter = float(nx.diameter(G))
        if diameter > max_hops:
            continue
        break
    else:
        return [], [], {}, {}, 0.0, float("nan")

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
    rates = compute_edge_probs(G, distances)

    return nodes, edges, fidelites, rates, avg_deg, diameter


def fat_tree(
    k: int = 4,
    qpu_edge_dist=0.1,
    edge_aggregate_dist=0.3,
    aggregate_core_dist=0.6,
) -> tuple[list, list, dict, dict, list, float]:
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

    Returns:
        tuple[list, list, dict, dict, list, float]: A tuple containing:
            - List of nodes in the graph.
            - List of edges in the graph.
            - Dictionary mapping edges to their fidelities.
            - Dictionary mapping edges to their p_gen rates.
            - List of QPUs (hosts) in the graph.
            - Diameter of the graph.
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
    fidelities = compute_edge_fidelities(G, distances)
    rates = compute_edge_probs(G, distances)
    diameter = float(nx.diameter(G))

    return nodes, edges, fidelities, rates, qpus, diameter


def dragonfly(
    a: int = 4,
    h: int = 2,
    p: int = 2,
    qpu_router_dist: float = 0.1,
    intra_group_dist: float = 0.3,
    global_dist: float = 0.6,
) -> tuple[list, list, dict, dict, list, float]:
    """Generates a dragonfly topology with g = a * h + 1 groups. Each group
    contains a routers connected in a complete graph, each router hosts p
    QPUs and has h global links, so every pair of groups is connected by
    exactly one global link. See: https://doi.org/10.1145/1394608.1382129

    Args:
        a (int, optional): Number of routers per group.
        h (int, optional): Number of global links per router.
        p (int, optional): Number of QPUs (hosts) per router.
        qpu_router_dist (float, optional): Distance between QPUs and routers.
        intra_group_dist (float, optional): Distance between routers within
        a group.
        global_dist (float, optional): Distance between routers of different
        groups.

    Returns:
        tuple[list, list, dict, dict, list, float]: A tuple containing:
            - List of nodes in the graph.
            - List of edges in the graph.
            - Dictionary mapping edges to their fidelities.
            - Dictionary mapping edges to their p_gen rates.
            - List of QPUs (hosts) in the graph.
            - Diameter of the graph.
    """
    G = nx.Graph()
    qpus = []
    groups = a * h + 1
    ports = a * h

    for g in range(groups):
        routers = [f"grp{g}_rtr_{i}" for i in range(a)]
        G.add_nodes_from(routers, layer=0, group=g)

        # router <-> router (complete graph within the group)
        for i, r in enumerate(routers):
            for r2 in routers[i + 1:]:
                G.add_edge(r, r2, dist=float(intra_group_dist))

        # router <-> qpu
        for i, r in enumerate(routers):
            for q in range(p):
                qpu = f"grp{g}_qpu_{i}_{q}"
                qpus.append(qpu)
                G.add_node(qpu, layer=1, group=g)
                G.add_edge(r, qpu, dist=float(qpu_router_dist))

    # group <-> group: port t of group g connects to group (g + t + 1) mod
    # groups and belongs to router t // h; the peer uses port ports - 1 - t
    for g in range(groups):
        for t in range(ports):
            g2 = (g + t + 1) % groups
            if g < g2:
                r = f"grp{g}_rtr_{t // h}"
                r2 = f"grp{g2}_rtr_{(ports - 1 - t) // h}"
                G.add_edge(r, r2, dist=float(global_dist))

    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda edge: (str(edge[0]), str(edge[1])))
    distances = {(u, v): float(G.edges[u, v]["dist"]) for (u, v) in edges}
    fidelities = compute_edge_fidelities(G, distances)
    rates = compute_edge_probs(G, distances)
    diameter = float(nx.diameter(G))

    return nodes, edges, fidelities, rates, qpus, diameter


def clos(
    n_spine: int = 4,
    n_leaf: int = 4,
    hosts_per_leaf: int = 4,
    qpu_leaf_dist: float = 0.1,
    leaf_spine_dist: float = 0.3,
) -> tuple[list, list, dict, dict, list, float]:
    """Generates a leaf-spine (two-tier Clos).

    Args:
        n_spine (int, optional): Number of spine switches.
        n_leaf (int, optional): Number of leaf switches.
        hosts_per_leaf (int, optional): Number of QPUs (hosts) attached to
        each leaf switch.
        qpu_leaf_dist (float, optional): Distance between QPUs and leaf
        switches.
        leaf_spine_dist (float, optional): Distance between leaf and spine
        switches.

    Returns:
        tuple[list, list, dict, dict, list, float]: A tuple containing:
            - List of nodes in the graph.
            - List of edges in the graph.
            - Dictionary mapping edges to their fidelities.
            - Dictionary mapping edges to their p_gen rates.
            - List of QPUs (hosts) in the graph.
            - Diameter of the graph.
    """
    G = nx.Graph()
    qpus = []

    spines = [f"spine_{i}" for i in range(n_spine)]
    G.add_nodes_from(spines, layer=0)

    for j in range(n_leaf):
        leaf = f"leaf_{j}"
        G.add_node(leaf, layer=1)

        # leaf <-> spine
        for s in spines:
            G.add_edge(leaf, s, dist=float(leaf_spine_dist))

        # leaf <-> qpu
        for h in range(hosts_per_leaf):
            qpu = f"leaf_{j}_qpu_{h}"
            qpus.append(qpu)
            G.add_node(qpu, layer=2)
            G.add_edge(leaf, qpu, dist=float(qpu_leaf_dist))

    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda edge: (str(edge[0]), str(edge[1])))
    distances = {(u, v): float(G.edges[u, v]["dist"]) for (u, v) in edges}
    fidelities = compute_edge_fidelities(G, distances)
    rates = compute_edge_probs(G, distances)
    diameter = float(nx.diameter(G))

    return nodes, edges, fidelities, rates, qpus, diameter
