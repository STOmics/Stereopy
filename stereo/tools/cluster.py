#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:cluster.py
@time:2021/03/19
"""
import leidenalg as la
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import numpy as np
from umap.umap_ import fuzzy_simplicial_set


class Neighbors(object):
    def __init__(self, x, n_neighbors):
        self.x = x
        self.n_neighbors = n_neighbors

    def find_n_neighbors(self):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='ball_tree').fit(self.x)
        dists, indices = nbrs.kneighbors(self.x)
        nn_idx = indices[:, 1:]
        nn_dist = dists[:, 1:]
        return nn_idx, nn_dist

    def get_igraph_from_knn(self, nn_idx, nn_dist):
        j = nn_idx.ravel().astype(int)
        dist = nn_dist.ravel()
        i = np.repeat(np.arange(nn_idx.shape[0]), self.n_neighbors)

        vertex = list(range(nn_dist.shape[0]))
        edges = list(tuple(zip(i, j)))
        G = ig.Graph()
        G.add_vertices(vertex)
        G.add_edges(edges)
        G.es['weight'] = dist
        return G

    def get_parse_distances(self, nn_idx, nn_dist):
        n_obs = self.x.shape[0]
        rows = np.zeros((n_obs * self.n_neighbors), dtype=np.int64)
        cols = np.zeros((n_obs * self.n_neighbors), dtype=np.int64)
        vals = np.zeros((n_obs * self.n_neighbors), dtype=np.float64)

        for i in range(nn_idx.shape[0]):
            for j in range(self.n_neighbors):
                if nn_idx[i, j] == -1:
                    continue  # We didn't get the full knn for i
                if nn_idx[i, j] == i:
                    val = 0.0
                else:
                    val = nn_dist[i, j]

                rows[i * self.n_neighbors + j] = i
                cols[i * self.n_neighbors + j] = nn_idx[i, j]
                vals[i * self.n_neighbors + j] = val

        distances = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
        distances.eliminate_zeros()
        return distances.tocsr()

    def get_connectivities(self, nn_idx, nn_dist):
        n_obs = self.x.shape[0]
        x = coo_matrix(([], ([], [])), shape=(n_obs, 1))
        connectivities = fuzzy_simplicial_set(x, self.n_neighbors, None, None, knn_indices=nn_idx, knn_dists=nn_dist,
                                              set_op_mix_ratio=1.0, local_connectivity=1.0)
        if isinstance(connectivities, tuple):
            connectivities = connectivities[0]
        return connectivities.tocsr()


def run_neighbors(x, neighbors=30):
    neighbor = Neighbors(x, neighbors)
    nn_idx, nn_dist = neighbor.find_n_neighbors()
    return neighbor, nn_idx, nn_dist


def run_louvain(x, neighbor, nn_idx, nn_dist):
    g = neighbor.get_igraph_from_knn(nn_idx, nn_dist)
    louvain_partition = g.community_multilevel(weights=g.es['weight'], return_levels=False)
    clusters = np.arange(x.shape[0])
    for i in range(len(louvain_partition)):
        clusters[louvain_partition[i]] = str(i)
    return clusters


def run_knn_leiden(x, neighbor, nn_idx, nn_dist, diff=1):
    g = neighbor.get_igraph_from_knn(nn_idx, nn_dist)
    optimiser = la.Optimiser()
    leiden_partition = la.ModularityVertexPartition(g, weights=g.es['weight'])
    while diff > 0:
        diff = optimiser.optimise_partition(leiden_partition, n_iterations=10)
    clusters = np.arange(x.shape[0])
    for i in range(len(leiden_partition)):
        clusters[leiden_partition[i]] = str(i)
    return clusters


def run_cluster(x, method='leiden', do_pca=True, n_pcs=30):
    """
    :param x: np.array, shape: (m, n)，m个bin， n为embedding
    :param method:
    :param do_pca:
    :param n_pcs:
    :return:
    """
    if do_pca:
        pca_obj = PCA(n_components=n_pcs)
        x = pca_obj.fit_transform(x)

    neighbor, nn_idx, nn_dist = run_neighbors(x)
    if method == 'leiden':
        cluster = run_knn_leiden(x, neighbor, nn_idx, nn_dist)
    else:
        cluster = run_louvain(x, neighbor, nn_idx, nn_dist)
    return cluster
