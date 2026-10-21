"""Small graph-quality summaries used to compare graph construction modes."""

from itertools import combinations


def graph_diagnostics(adjacency, support_node_indices=()):
    """Summarize topology and support-pair reachability for one graph."""
    node_count = len(adjacency)
    if any(len(row) != node_count for row in adjacency):
        raise ValueError("Adjacency must be a square matrix.")

    neighbors = []
    degrees = []
    for node_index in range(node_count):
        node_neighbors = {
            other_index
            for other_index in range(node_count)
            if other_index != node_index
            and (adjacency[node_index][other_index] or adjacency[other_index][node_index])
        }
        neighbors.append(node_neighbors)
        degrees.append(len(node_neighbors))

    undirected_edges = sum(degrees) // 2
    possible_edges = node_count * (node_count - 1) // 2

    component_count = 0
    visited = set()
    for start in range(node_count):
        if start in visited:
            continue
        component_count += 1
        pending = [start]
        visited.add(start)
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)

    support_nodes = sorted(set(support_node_indices))
    if any(index < 0 or index >= node_count for index in support_nodes):
        raise ValueError("Support node index is outside the adjacency matrix.")

    support_pairs = list(combinations(support_nodes, 2))
    within_one_hop = 0
    within_two_hops = 0
    for first, second in support_pairs:
        directly_connected = second in neighbors[first]
        if directly_connected:
            within_one_hop += 1
            within_two_hops += 1
            continue
        if any(second in neighbors[neighbor] for neighbor in neighbors[first]):
            within_two_hops += 1

    pair_count = len(support_pairs)
    return {
        "node_count": node_count,
        "undirected_edge_count": undirected_edges,
        "edge_density": (
            undirected_edges / possible_edges if possible_edges else 0.0
        ),
        "mean_degree": sum(degrees) / node_count if node_count else 0.0,
        "max_degree": max(degrees, default=0),
        "connected_components": component_count,
        "support_pair_count": pair_count,
        "support_pairs_within_1_hop": within_one_hop,
        "support_pairs_within_2_hops": within_two_hops,
        "support_pair_coverage_1_hop": (
            within_one_hop / pair_count if pair_count else 0.0
        ),
        "support_pair_coverage_2_hops": (
            within_two_hops / pair_count if pair_count else 0.0
        ),
    }
