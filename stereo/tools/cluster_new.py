#!/usr/bin/env python3
# coding: utf-8
"""
@file: cluster.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran

change log:
    2021/09/07  create file.
"""
import numpy as np
import leidenalg
from ..core.tool_base import ToolBase
from ..log_manager import logger
from stereo.algorithm.neighbors import Neighbors
from .dim_reduce import DimReduce
import pandas as pd
from typing import Optional, Type
from ..plots.scatter import base_scatter, plt, colors
from natsort import natsorted
from sklearn.metrics import pairwise_distances
from typing import Any, Mapping
from types import MappingProxyType
import phenograph

try:
    from leidenalg.VertexPartition import MutableVertexPartition
except ImportError:

    class MutableVertexPartition:
        pass

    MutableVertexPartition.__module__ = 'leidenalg.VertexPartition'


class Cluster(ToolBase):
    """
    clustering bin-cell using nearest neighbor algorithm.

    :param data: expression matrix, pd.Dataframe or StereoExpData object
    :param method: louvain or leiden
    :param pca_x: result of pca, if is None, will run DimReduce tool before clustering
    :param n_neighbors: number of neighbors
    :param normalization: normalize the expression matrix or not, will run normalization before pca dimension reduction
    if pca_x is None and normalization is True.

    """
    def __init__(
            self,
            data=None,
            pca_x: Optional[pd.DataFrame] = None,
            method: str = 'louvain',
            dim_reduce_method: str = 'pca',
            neighbors_method: str = 'umap',
            n_pcs: int = 40,
            n_iter: int = 250,
            n_neighbors: int = 10,
            metric: Optional[str] = 'euclidean',
            knn: bool = True,
            min_dist: float = 0.3,
            phenograpg_k: int = 20
    ):
        super(Cluster, self).__init__(data=data, method=method)
        self.n_neighbors = n_neighbors if n_neighbors < len(self.data.cell_names) else int(len(self.data.cell_names) / 2)
        self.dim_reduce_method = dim_reduce_method
        self.n_pcs = n_pcs
        self.n_iter = n_iter
        self.pca_x = pca_x
        self.metric = metric
        self.neighbors_method = neighbors_method
        self.phenograph_k = phenograpg_k
        self.knn = knn
        self.min_dist = min_dist
        self.nor_x = None

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['leiden', 'louvain']
        self._method_check(method, m_range)

    def run_dim_reduce(self):
        dim_reduce = DimReduce(self.data, method=self.dim_reduce_method, n_pcs=self.n_pcs, n_iter=self.n_iter,
                               n_neighbors=self.n_neighbors, min_dist=self.min_dist)
        dim_reduce.fit(self.nor_x)
        self.pca_x = dim_reduce.result
        return self.pca_x

    def run_neighbors(self,):
        """
        find neighbors
        :return: neighbor object and neighbor cluster info
        """
        # from stereo.algorithm.neighbors import get_indices_distances_from_dense_matrix
        neighbor = Neighbors(self.pca_x, self.n_neighbors, self.n_pcs, self.metric, self.neighbors_method, self.knn)
        neighbor.check_setting()
        x = neighbor.choose_x()
        use_dense_distances = (self.metric == 'euclidean' and x.shape[0] < 8192) or not self.knn
        dists = x
        if use_dense_distances:
            print(x.shape)
            dists = pairwise_distances(x, metric=self.metric,)
            print(dists[0])
            knn_indices, knn_distances = neighbor.get_indices_distances_from_dense_matrix(dists)
            if self.knn:
                dists = neighbor.get_parse_distances_numpy(
                    knn_indices, knn_distances, x.shape[0],
                )
                print(dists[0])
                print("s1")
        else:
            if x.shape[0] < 4096:
                dists = pairwise_distances(x, metric=self.metric)
            self.metric = 'precomputed'
            knn_indices, knn_distances, forest = neighbor.compute_neighbors_umap(dists, )
        if not use_dense_distances or self.neighbors_method in {'umap'}:
            connectivities = neighbor.get_connectivities_umap(knn_indices, knn_distances)
            dists = neighbor.get_parse_distances_umap(knn_indices, knn_distances, )
            print("end")
            print(dists[0])
        if self.neighbors_method == 'gauss':
            connectivities = neighbor.compute_connectivities_diffmap(dists)
        return neighbor, dists, connectivities

    def run_louvain(self, neighbor, nn_idx, nn_dist):
        """
        louvain method

        :param neighbor: neighbor object
        :param nn_idx: n neighbors's ids
        :param nn_dist: n neighbors's data results
        :return:
        """
        g = neighbor.get_igraph_from_knn(nn_idx, nn_dist)
        louvain_partition = g.community_multilevel(weights=g.es['weight'], return_levels=False)
        clusters = np.arange(len(self.data.cell_names))
        for i in range(len(louvain_partition)):
            clusters[louvain_partition[i]] = str(i)
        return clusters

    def run_knn_leiden(
            self,
            neighbor,
            adjacency=None,
            directed: bool = True,
            resolution: float = 1,
            use_weights: bool = True,
            random_state=0,
            n_iterations: int = -1,
            partition_type: Optional[Type[MutableVertexPartition]] = None,
            **partition_kwargs,
    ):
        partition_kwargs = dict(partition_kwargs)
        # convert it to igraph
        g = neighbor.get_igraph_from_adjacency(adjacency, directed=directed)
        # filp to the default partition type if not overriden by the user
        if partition_type is None:
            partition_type = leidenalg.RBConfigurationVertexPartition
        if use_weights:
            partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
            partition_kwargs['n_iterations'] = n_iterations
            partition_kwargs['seed'] = random_state
        if resolution is not None:
            partition_kwargs['resolution_parameter'] = resolution
        # clustering proper
        part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
        # store output
        groups = np.array(part.membership)
        leiden_partition = groups
        cluster = pd.Categorical(
            values=groups.astype('U'),
            categories=natsorted(map(str, np.unique(groups))),
        )
        return cluster

    def run_phenograph(self, phenograph_k):
        communities, _, _ = phenograph.cluster(self.pca_x.matrix, k=phenograph_k)
        cluster = communities.astype(str)
        return cluster

    def reset_dim_reduce_params(self, method, n_pcs, n_iter, n_neighbors, min_dist):
        self.dim_reduce_method = method
        self.n_pcs = n_pcs
        self.n_iter = n_iter
        self.min_dist = min_dist
        self.neighbors = n_neighbors

    def reset_neighbors_params(self, method, n_pcs, n_neighbors, metric):
        self.neighbors_method = method
        self.n_pcs = n_pcs
        self.metric = metric
        self.neighbors = n_neighbors

    def fit(self):
        """
        running and add results
        """
        self.sparse2array()
        #self.logger.info('start to run dim reduce...')
        #self.run_dim_reduce()
        #self.logger.info('start to run neighbors...')
        neighbor, dists, connectivities = self.run_neighbors()
        self.logger.info(f'start to run {self.method} cluster...')
        if self.method == 'leiden':
            cluster = self.run_knn_leiden(neighbor, adjacency=connectivities)
        # elif self.method == 'phenograph':
        #     cluster = self.run_phenograph(self.phenograph_k)
        # else:
        #     cluster = self.run_louvain(neighbor, nn_idx, nn_dist)
        cluster = [str(i) for i in cluster]
        info = {'bins': self.data.cell_names, 'cluster': cluster}
        df = pd.DataFrame(info)
        self.result.matrix = df
        self.logger.info('finish...')
        return df

    def plot_scatter(self, plot_dim_reduce=False, file_path=None):
        """
        plot scatter after
        :param plot_dim_reduce: plot cluster after dimension reduce if true
        :param file_path:
        :return:
        """
        if plot_dim_reduce:
            base_scatter(self.pca_x.values[:, 0], self.pca_x.values[:, 1],
                         color_values=np.array(self.result.matrix['cluster']), color_list=colors)
        else:
            base_scatter(self.data.position[:, 0], self.data.position[:, 1],
                         color_values=np.array(self.result.matrix['cluster']),
                         color_list=colors)
        if file_path:
            plt.savefig(file_path)
