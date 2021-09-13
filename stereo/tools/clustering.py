#!/bin/env python3
"""
Create on 2020-12-04
@author liulin4
@revised on Jan22-2021

change log:
    2021/05/20 rst supplement. by: qindanhua.
    2021/06/20 adjust for restructure base class . by: qindanhua.
    2021/06/29 last modified by qindanhua.
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
from ..plots.scatter import base_scatter, plt, colors
import colorcet as cc


class Clustering(ToolBase):
    """
    clustering bin-cell using nearest neighbor algorithm.

    :param data: expression matrix, pd.Dataframe or StereoExpData object
    :param method: louvain or leiden
    :param pca_x: result of pca, if is None, will run DimReduce tool before clustering
    :param n_neighbors: number of neighbors
    :param normalization: normalize the expression matrix or not, will run normalization before pca dimension reduction
    if pca_x is None and normalization is True.

    Examples
    --------

    >>> from stereo.tools.clustering import Clustering
    >>> import pandas as pd
    >>> test_exp_matrix = pd.DataFrame({'gene_1': [0, 1, 2, 0, 3, 4], 'gene_2': [1, 3, 2, 0, 3, 0], 'gene_3': [0, 0, 2, 0, 3, 1]}, index=['cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5', 'cell_6'])
    >>> test_exp_matrix
            gene_1  gene_2  gene_3
    cell_1       0       1       0
    cell_2       1       3       0
    cell_3       2       2       2
    cell_4       0       0       0
    cell_5       3       3       3
    cell_6       4       0       1
    >>> ct = Clustering(test_exp_matrix)
    >>> ct.fit()
    >>> ct.result.matrix
    """
    def __init__(
            self,
            data=None,
            method: str = 'louvain',
            pca_x: Optional[pd.DataFrame] = None,
            n_neighbors: int = 30,
            normalization: bool = False,
    ):
        super(Clustering, self).__init__(data=data, method=method)
        self._neighbors = n_neighbors if n_neighbors < len(self.data.cell_names) else int(len(self.data.cell_names) / 2)
        # self.neighbors = n_neighbors
        self.normalization = normalization
        # self._pca_x = pca_x
        self.pca_x = pca_x

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

    @property
    def pca_x(self):
        return self._pca_x

    @pca_x.setter
    def pca_x(self, pca_x):
        input_df = self._check_input_data(pca_x)
        self._pca_x = input_df

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['leiden', 'louvain']
        self._method_check(method, m_range)

    def run_normalize(self, normalize_method='normalize_total', nor_target_sum=10000):
        """
        normalize input data

        :return: normalized data
        """
        normakizer = Normalizer(self.data, method=normalize_method, target_sum=nor_target_sum)
        nor_x = normakizer.fit()
        return nor_x

    def get_dim_reduce_x(self):
        """
        get dimensionality reduction results

        :return: pca results
        """
        if self.pca_x.is_empty:
            # TODO normalize or not if set Clustering().normalization = True and self.pca_x.is_empty if False.
            nor_x = self.run_normalize() if self.normalization else self.data.exp_matrix
            dim_reduce = DimReduce(self.data, method='pca', n_pcs=self.neighbors)
            dim_reduce.fit(nor_x)
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

    def fit(self):
        """
        running and add results
        """
        self.data.sparse2array()
        self.get_dim_reduce_x()
        neighbor, nn_idx, nn_dist = self.run_neighbors(self.pca_x.matrix)
        if self.method == 'leiden':
            cluster = self.run_knn_leiden(neighbor, nn_idx, nn_dist)
        else:
            cluster = self.run_louvain(neighbor, nn_idx, nn_dist)
        cluster = [str(i) for i in cluster]
        info = {'bins': self.data.cell_names, 'cluster': cluster}
        df = pd.DataFrame(info)
        self.result.matrix = df
        # TODO  added for find marker
        # self.data.obs[self.name] = cluster
        return df

    def plot_scatter(self, plot_dim_reduce=False, file_path=None):
        """
        plot scatter after
        :param plot_dim_reduce: plot cluster after dimension reduce if true
        :param file_path:
        :return:
        """
        if plot_dim_reduce:
            base_scatter(self.pca_x.matrix.values[:, 0], self.pca_x.matrix.values[:, 1],
                         color_values=np.array(self.result.matrix['cluster']), color_list=cc.glasbey)
        else:
            base_scatter(self.data.position[:, 0], self.data.position[:, 1],
                         color_values=np.array(self.result.matrix['cluster']),
                         color_list=cc.glasbey)
        if file_path:
            plt.savefig(file_path)
