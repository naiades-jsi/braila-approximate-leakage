import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def to_adjacency_mat(network_graph):
    node_id_vec = network_graph.nodes()
    edge_tup_vec = network_graph.edges()

    node_idx_map = {node_id: nodeN for nodeN, node_id in enumerate(node_id_vec)}

    adj_mat = np.zeros((len(node_id_vec), len(node_id_vec)), dtype=float)

    for src_id, dst_id in edge_tup_vec:
        srcN = node_idx_map[src_id]
        dstN = node_idx_map[dst_id]
        adj_mat[srcN, dstN] = 1.0

    return adj_mat, node_id_vec

def _dist_from_set_vec(adj_mat, reached_indicator_vec):
    n_nodes = adj_mat.shape[0]

    path_len_vec = np.empty(n_nodes)
    ones_n = np.ones(n_nodes)
    tmp_n = np.empty(n_nodes)

    for nodeN in range(n_nodes):
        path_len_vec[nodeN] = 0.0 if reached_indicator_vec[nodeN] > 0.0 else np.inf

    for path_len in range(1, n_nodes):
        # x = min(Ax, 1)
        np.dot(reached_indicator_vec, adj_mat, out=tmp_n)
        np.minimum(tmp_n, ones_n, out=reached_indicator_vec)

        # tmp = len / x
        np.reciprocal(reached_indicator_vec, out=tmp_n)
        np.multiply(tmp_n, float(path_len), out=tmp_n)

        # p = min(p, k / x)
        np.minimum(path_len_vec, tmp_n, out=path_len_vec)

    return path_len_vec



def dist_from_set_vec(network_graph, node_id_set):
    adj_mat, node_id_vec = to_adjacency_mat(network_graph)
    node_idx_map = {node_id: nodeN for nodeN, node_id in enumerate(node_id_vec)}

    n_nodes = adj_mat.shape[0]

    # create the initial location vector
    reached_indicator_vec = np.zeros(n_nodes)

    for node_id in node_id_set:
        nodeN = node_idx_map[node_id]
        reached_indicator_vec[nodeN] = 1

    path_len_vec = _dist_from_set_vec(adj_mat, reached_indicator_vec)

    return path_len_vec, node_id_vec


def dist_from_set_map(network_graph, node_id_set):
    path_len_vec, node_id_vec = dist_from_set_vec(network_graph, node_id_set)
    return {node_id: path_len_vec[nodeN] for nodeN, node_id in enumerate(node_id_vec)}


def undir_dist_from_set_vec(network_graph, node_id_set):
    # create an undirected adjacency matrix
    adj_mat, node_id_vec = to_adjacency_mat(network_graph)
    adj_mat = adj_mat + np.transpose(adj_mat)

    node_idx_map = {node_id: nodeN for nodeN, node_id in enumerate(node_id_vec)}

    n_nodes = adj_mat.shape[0]

    # create the initial location vector
    reached_indicator_vec = np.zeros(n_nodes)

    for node_id in node_id_set:
        nodeN = node_idx_map[node_id]
        reached_indicator_vec[nodeN] = 1

    path_len_vec = _dist_from_set_vec(adj_mat, reached_indicator_vec)

    return path_len_vec, node_id_vec


def undir_dist_from_set_map(network_graph, node_id_set):
    path_len_vec, node_id_vec = undir_dist_from_set_vec(network_graph, node_id_set)
    return {node_id: path_len_vec[nodeN] for nodeN, node_id in enumerate(node_id_vec)}


