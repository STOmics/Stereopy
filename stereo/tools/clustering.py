#!/bin/env python3
"""
Create on 2020-12-04
@author liulin4
@revised on Jan22-2021

change log:
    2021/05/20 rst supplement. by: qindanhua.
    2021/06/20 adjust for restructure base class . by: qindanhua.
"""

import numpy as np
import leidenalg as la
from ..core.tool_base import ToolBase
# from ..log_manager import logger
from .neighbors import Neighbors
from ..preprocess.normalize import Normalizer
from .dim_reduce import DimReduce
import pandas as pd


class Clustering(ToolBase):
    """
    clustering bin-cell using nearest neighbor algorithm.

    :param data: anndata object
    :param method: louvain or leiden\
    :param n_neighbors: number of neighbors
    # :param normalize_key: defined when running 'Normalize' tool by setting 'name' property.
    # :param normalize_method: normalization method, Normalizer will be run before clustering if the param is set.
    # :param nor_target_sum: summary of target
    # :param name: name of this tool and will be used as a key when adding tool result to andata object.
    """
    def __init__(
            self,
            data=None,
            method: str = 'louvain',
            pca_x=None,
            n_neighbors: int = 30,
            normalize=False,
            # normalize_key='cluster_normalize',
            # normalize_method=None,
            # nor_target_sum=10000,
            # name='clustering'
    ):
        super(Clustering, self).__init__(data=data, method=method)
        # self.param = self.get_params(locals())
        # self.dim_reduce_key = dim_reduce_key
        self.neighbors = n_neighbors
        self.normalize = normalize
        self._pca_x = pca_x

    @property
    def pca_x(self):
        return self._pca_x

    @pca_x.setter
    def pca_x(self, pca_x):
        input_df = self.check_input_data(pca_x)
        self._pca_x = input_df

    def run_normalize(self, normalize_method='normalize_total', nor_target_sum=10000):
        """
        normalize input data

        :return: normalized data
        """
        normakizer = Normalizer(self.data, method=normalize_method, inplace=False, target_sum=nor_target_sum)
        nor_x = normakizer.fit()
        return nor_x

    def get_dim_reduce_x(self):
        """
        get dimensionality reduction results

        :return: pca results
        """
        if self.pca_x is None:
            nor_x = self.run_normalize() if self.normalize else self.data.X
            dim_reduce = DimReduce(self.data, method='pca', n_pcs=30)
            dim_reduce.fit(nor_x)
            self.pca_x = dim_reduce.result.x_reduce
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
        clusters = np.arange(len(self.data.obs))
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
        clusters = np.arange(len(self.data.obs))
        for i in range(len(leiden_partition)):
            clusters[leiden_partition[i]] = str(i)
        return clusters

    def fit(self):
        """
        running and add results
        """
        self.get_dim_reduce_x()
        neighbor, nn_idx, nn_dist = self.run_neighbors(self.pca_x)
        if self.method == 'leiden':
            cluster = self.run_knn_leiden(neighbor, nn_idx, nn_dist)
        else:
            cluster = self.run_louvain(neighbor, nn_idx, nn_dist)
        cluster = [str(i) for i in cluster]
        info = {'bins': self.data.obs_names, 'cluster': cluster}
        df = pd.DataFrame(info)
        self.result.matrix = df
        # TODO  added for find marker
        # self.data.obs[self.name] = cluster
        return df
