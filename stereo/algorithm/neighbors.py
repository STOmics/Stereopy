#!/usr/bin/env python3
# coding: utf-8
"""
@author: Yiran Wu  wuyiran@genomics.cn
@last modified by: Yiran Wu
@file:neighbors.py
@time:2021/09/01
"""
from ..log_manager import logger
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from numpy import random
import igraph as ig
import numpy as np
from typing import Union, Any, Mapping, Optional, Callable
from typing_extensions import Literal
from types import MappingProxyType
from sklearn.metrics import pairwise_distances
AnyRandom = Union[None, int, random.RandomState]

_Method = Literal['umap', 'gauss']
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
# from sklearn.metrics.pairwise_distances.__doc__:
_MetricSparseCapable = Literal[
    'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
]
_MetricScipySpatial = Literal[
    'braycurtis',
    'canberra',
    'chebyshev',
    'correlation',
    'dice',
    'hamming',
    'jaccard',
    'kulsinski',
    'mahalanobis',
    'minkowski',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'yule',
]
_Metric = Union[_MetricSparseCapable, _MetricScipySpatial]


def find_neighbors(
    x: np.ndarray,
    n_neighbors: Optional[int] = 15,
    n_pcs: Optional[int] = None,
    method: Optional[_Method] = 'umap',
    metric: Union[_Metric, _MetricFn] = 'euclidean',
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    knn: bool = True,
    random_state: AnyRandom = 0,
):
    """

    :param x:
        class:`~numpy.ndarray`. PCA representation of data.
    :param n_neighbors:
        Use this number of nearest neighbors.
    :param n_pcs:
    :param method:
        Use 'umap' or 'gauss'. for computing connectivities.
    :param metric:
        A known metric’s name or a callable that returns a distance.
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
    :param metric_kwds:
        Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
    :param knn:
        If `True`, use a hard threshold to restrict the number of neighbors to
        `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        `n_neighbors` nearest neighbor.
    :param random_state:
        A state capable being used as a numpy random state.
    :return:
        neighbor: Neighbors object
        dists: sparse
        connectivities: sparse
    """
    neighbor = Neighbors(
        x=x,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        method=method,
        knn=knn,
        random_state=random_state,
    )
    neighbor.check_setting()
    neighbor.x = neighbor.choose_x()
    use_dense_distances = (neighbor.metric == 'euclidean' and neighbor.x.shape[0] < 8192) or not neighbor.knn
    dists = neighbor.x
    if use_dense_distances:
        dists = pairwise_distances(neighbor.x, metric=neighbor.metric, **metric_kwds)
        knn_indices, knn_distances = neighbor.get_indices_distances_from_dense_matrix(dists)
        if knn:
            dists = neighbor.get_parse_distances_numpy(
                knn_indices, knn_distances, neighbor.x.shape[0],
            )
    else:
        if neighbor.x.shape[0] < 4096:
            dists = pairwise_distances(neighbor.x, metric=neighbor.metric, **metric_kwds)
            neighbor.metric = 'precomputed'
        knn_indices, knn_distances, forest = neighbor.compute_neighbors_umap(
            dists, random_state=neighbor.random_state, metric_kwds=metric_kwds)
    if not use_dense_distances or neighbor.method in {'umap'}:
        connectivities = neighbor.compute_connectivities_umap(knn_indices, knn_distances)
        dists = neighbor.get_parse_distances_umap(knn_indices, knn_distances, )
    if method == 'gauss':
        connectivities = neighbor.compute_connectivities_diffmap(dists)
    return neighbor, dists, connectivities


class Neighbors(object):
    def __init__(self, x, n_neighbors, n_pcs, method, metric, knn, random_state):
        self.x = x
        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.metric = metric
        self.method = method
        self.knn = knn
        self.random_state = random_state

    def check_setting(self):
        if self.method == 'umap' and not self.knn:
            logger.error(f'`method=\'umap\' only with knn=True`.')
        if self.method not in {'umap', 'gauss'}:
            logger.error(f'`method=\'umap\' or method=\'gauss\'`.')

    def choose_x(self):
        self.x = self.x[:, :self.n_pcs]
        return self.x

    def get_indices_distances_from_dense_matrix(self, dists,):
        sample_range = np.arange(dists.shape[0])[:, None]
        indices = np.argpartition(dists, self.n_neighbors - 1, axis=1)[:, :self.n_neighbors]
        indices = indices[sample_range, np.argsort(dists[sample_range, indices])]
        distances = dists[sample_range, indices]
        return indices, distances

    def get_indices_distances_from_sparse_matrix(self, dists):
        indices = np.zeros((dists.shape[0], self.n_neighbors), dtype=int)
        distances = np.zeros((dists.shape[0], self.n_neighbors), dtype=dists.dtype)
        n_neighbors_m1 = self.n_neighbors - 1
        for i in range(indices.shape[0]):
            neighbors = dists[i].nonzero()  # 'true' and 'spurious' zeros
            indices[i, 0] = i
            distances[i, 0] = 0
            # account for the fact that there might be more than n_neighbors
            # due to an approximate search
            # [the point itself was not detected as its own neighbor during the search]
            if len(neighbors[1]) > n_neighbors_m1:
                sorted_indices = np.argsort(dists[i][neighbors].A1)[:n_neighbors_m1]
                indices[i, 1:] = neighbors[1][sorted_indices]
                distances[i, 1:] = dists[i][
                    neighbors[0][sorted_indices], neighbors[1][sorted_indices]
                ]
            else:
                indices[i, 1:] = neighbors[1]
                distances[i, 1:] = dists[i][neighbors]
        return indices, distances

    def get_parse_distances_numpy(self, indices, distances, n_obs,):
        n_nonzero = n_obs * self.n_neighbors
        indptr = np.arange(0, n_nonzero + 1, self.n_neighbors)
        dists = csr_matrix(
            (
                distances.copy().ravel(),  # copy the data, otherwise strange behavior here
                indices.copy().ravel(),
                indptr,
            ),
            shape=(n_obs, n_obs),
        )
        dists.eliminate_zeros()
        return dists

    def get_parse_distances_umap(self, nn_idx, nn_dist):
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

    def compute_neighbors_umap(
            self,
            x,
            random_state: AnyRandom = None,
            angular: bool = False,
            verbose: bool = False,
            metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    ):
        from umap.umap_ import nearest_neighbors
        random_state = check_random_state(random_state)
        knn_indices, knn_dists, forest = nearest_neighbors(
            x,
            self.n_neighbors,
            random_state=random_state,
            metric=self.metric,
            metric_kwds=metric_kwds,
            angular=angular,
            verbose=verbose,
        )

        return knn_indices, knn_dists, forest

    def find_n_neighbors(self,):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1, algorithm='ball_tree').fit(self.x)
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
        g = ig.Graph()
        g.add_vertices(vertex)
        g.add_edges(edges)
        g.es['weight'] = dist
        return g

    @staticmethod
    def get_igraph_from_adjacency(adjacency, directed=None):
        """Get igraph graph from adjacency matrix."""
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=directed)
        g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es['weight'] = weights
        except KeyError:
            pass
        if g.vcount() != adjacency.shape[0]:
            logger.error(
                f'The constructed graph has only {g.vcount()} nodes. '
                'Your adjacency matrix contained redundant nodes.'
            )
        return g

    def compute_connectivities_umap(self, nn_idx, nn_dist):
        from umap.umap_ import fuzzy_simplicial_set
        n_obs = self.x.shape[0]
        x = coo_matrix(([], ([], [])), shape=(n_obs, 1))
        connectivities = fuzzy_simplicial_set(x, self.n_neighbors, None, None, knn_indices=nn_idx, knn_dists=nn_dist,
                                              set_op_mix_ratio=1.0, local_connectivity=1.0)
        if isinstance(connectivities, tuple):
            connectivities = connectivities[0]
        return connectivities.tocsr()

    def compute_connectivities_diffmap(self, dists,):
        # init distances
        if self.knn:
            dsq = dists.power(2)
            indices, distances_sq = self.get_indices_distances_from_sparse_matrix(dsq,)
        else:
            dsq = np.power(dists, 2)
            indices, distances_sq = self.get_indices_distances_from_dense_matrix(dsq,)

        # exclude the first point, the 0th neighbor
        indices = indices[:, 1:]
        distances_sq = distances_sq[:, 1:]

        # choose sigma, the heuristic here doesn't seem to make much of a difference,
        # but is used to reproduce the figures of Haghverdi et al. (2016)
        if self.knn:
            # as the distances are not sorted
            # we have decay within the n_neighbors first neighbors
            sigmas_sq = np.median(distances_sq, axis=1)
        else:
            # the last item is already in its sorted position through argpartition
            # we have decay beyond the n_neighbors neighbors
            sigmas_sq = distances_sq[:, -1] / 4
        sigmas = np.sqrt(sigmas_sq)

        # compute the symmetric weight matrix
        if not issparse(dists):
            Num = 2 * np.multiply.outer(sigmas, sigmas)
            Den = np.add.outer(sigmas_sq, sigmas_sq)
            W = np.sqrt(Num / Den) * np.exp(-dsq / Den)
            # make the weight matrix sparse
            if not self.knn:
                mask = W > 1e-14
                W[~mask] = 0
            else:
                # restrict number of neighbors to ~k
                # build a symmetric mask
                mask = np.zeros(dsq.shape, dtype=bool)
                for i, row in enumerate(indices):
                    mask[i, row] = True
                    for j in row:
                        if i not in set(indices[j]):
                            W[j, i] = W[i, j]
                            mask[j, i] = True
                # set all entries that are not nearest neighbors to zero
                W[~mask] = 0
        else:
            W = (
                dsq.copy()
            )  # need to copy the distance matrix here; what follows is inplace
            for i in range(len(dsq.indptr[:-1])):
                row = dsq.indices[dsq.indptr[i]: dsq.indptr[i + 1]]
                num = 2 * sigmas[i] * sigmas[row]
                den = sigmas_sq[i] + sigmas_sq[row]
                W.data[dsq.indptr[i]: dsq.indptr[i + 1]] = np.sqrt(num / den) * np.exp(
                    -dsq.data[dsq.indptr[i]: dsq.indptr[i + 1]] / den
                )
            W = W.tolil()
            for i, row in enumerate(indices):
                for j in row:
                    if i not in set(indices[j]):
                        W[j, i] = W[i, j]
            W = W.tocsr()
        connectivities = W
        return connectivities
