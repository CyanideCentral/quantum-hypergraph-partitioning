
def incidence_to_hyperedges(incidence_matrix):
    num_vertices = len(incidence_matrix)
    num_hyperedges = len(incidence_matrix[0]) if num_vertices > 0 else 0

    hyperedges = []
    edge_weights = []
    for e_idx in range(num_hyperedges):
        edge_nodes = []
        for v_idx in range(num_vertices):
            if incidence_matrix[v_idx][e_idx] > 0:
                edge_nodes.append(v_idx)
        hyperedges.append(edge_nodes)
        edge_weights.append(incidence_matrix[v_idx][e_idx])

    return hyperedges, edge_weights