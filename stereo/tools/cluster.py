#!/usr/bin/env python3
# coding: utf-8
"""
@file: cluster.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/07/23  create file.
"""
import numpy as np
import leidenalg as la
from ..core.tool_base import ToolBase
from ..log_manager import logger
from stereo.algorithm.neighbors import Neighbors
from ..preprocess.normalize import Normalizer
from .dim_reduce import DimReduce
import pandas as pd
from typing import Optional
from ..plots.scatter import base_scatter, plt
import colorcet as cc
# import phenograph


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
            method: str = 'louvain',
            normalize_method: str = 'quantile',
            target_sum: Optional[int] = 10000,
            zscore_r: Optional[int] = 20,
            dim_reduce_method: str = 'pca',
            n_pcs: int = 30,
            n_iter: int = 250,
            n_neighbors: int = 10,
            min_dist: float = 0.3,
            phenograpg_k: int = 20
    ):
        super(Cluster, self).__init__(data=data, method=method)
        self._neighbors = n_neighbors if n_neighbors < len(self.data.cell_names) else int(len(self.data.cell_names) / 2)
        self.normalize_method = normalize_method
        self.target_sum = target_sum
        self.zscore_r = zscore_r
        self.dim_reduce_method = dim_reduce_method
        self.n_pcs = n_pcs
        self.n_iter = n_iter
        self.min_dist = min_dist
        self.nor_x = None
        self.pca_x = None
        self.phenograph_k = phenograpg_k

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, n_neighbors: int):
        if n_neighbors > len(self.data.cell_names):
            logger.error(f'n neighbor should be less than {len(self.data.cell_names)}')
            self._neighbors = self.neighbors
        else:
            self._neighbors = n_neighbors

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['leiden', 'louvain']
        self._method_check(method, m_range)

    def run_normalize(self):
        """
        normalize input data

        :return: normalized data
        """
        normakizer = Normalizer(self.data, method=self.normalize_method, target_sum=self.target_sum, r=self.zscore_r)
        self.nor_x = normakizer.fit()
        return self.nor_x

    def run_dim_reduce(self):
        dim_reduce = DimReduce(self.data, method=self.dim_reduce_method, n_pcs=self.n_pcs, n_iter=self.n_iter,
                               n_neighbors=self.neighbors, min_dist=self.min_dist)
        dim_reduce.fit(self.nor_x)
        self.pca_x = dim_reduce.result
        return self.pca_x

    def run_neighbors(self, x):
        """
        find neighbors

        :param x: input data array. [[cell_1, gene_count_1], [cell_1, gene_count_2], ..., [cell_M, gene_count_N]]
        :return: neighbor object and neighbor cluster info
        """
        neighbor = Neighbors(x, self.neighbors)
        nn_idx, nn_dist = neighbor.find_n_neighbors()
        return neighbor, nn_idx, nn_dist

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

    def run_knn_leiden(self, neighbor, nn_idx, nn_dist, diff=1):
        """
        leiden method

        :param neighbor: neighbor object
        :param nn_idx: n neighbors's ids
        :param nn_dist: n neighbors's data results
        :param diff:
        :return:
        """
        g = neighbor.get_igraph_from_knn(nn_idx, nn_dist)
        optimiser = la.Optimiser()
        leiden_partition = la.ModularityVertexPartition(g, weights=g.es['weight'])
        while diff > 0:
            diff = optimiser.optimise_partition(leiden_partition, n_iterations=10)
        clusters = np.arange(len(self.data.cell_names))
        for i in range(len(leiden_partition)):
            clusters[leiden_partition[i]] = str(i)
        return clusters

    def run_phenograph(self, phenograph_k):
        communities, _, _ = phenograph.cluster(self.pca_x, k=phenograph_k)
        cluster = communities.astype(str)
        return cluster

    def reset_normalize_params(self, method, target_sum, zscore_r):
        self.normalize_method = method
        self.target_sum = target_sum
        self.zscore_r = zscore_r

    def reset_dim_reduce_params(self, method, n_pcs, n_iter, n_neighbors, min_dist):
        self.dim_reduce_method = method
        self.n_pcs = n_pcs
        self.n_iter = n_iter
        self.min_dist = min_dist
        self.neighbors = n_neighbors

    def fit(self):
        """
        running and add results
        """
        self.data.sparse2array()
        self.logger.info('start to run normalization...')
        self.run_normalize()
        self.logger.info('start to run dim reduce...')
        self.run_dim_reduce()
        self.logger.info('start to run neighbors...')
        neighbor, nn_idx, nn_dist = self.run_neighbors(self.pca_x)
        self.logger.info(f'start to run {self.method} cluster...')
        if self.method == 'leiden':
            cluster = self.run_knn_leiden(neighbor, nn_idx, nn_dist)
        elif self.method == 'phenograph':
            cluster = self.run_phenograph(self.phenograph_k)
        else:
            cluster = self.run_louvain(neighbor, nn_idx, nn_dist)
        cluster = [str(i) for i in cluster]
        info = {'bins': self.data.cell_names, 'cluster': cluster}
        df = pd.DataFrame(info)
        self.result = df
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
                         color_values=np.array(self.result['cluster']), color_list=cc.glasbey)
        else:
            base_scatter(self.data.position[:, 0], self.data.position[:, 1],
                         color_values=np.array(self.result['cluster']),
                         color_list=cc.glasbey)
        if file_path:
            plt.savefig(file_path)
