#!/bin/env python3
"""
Create on 2020-12-04
@author liulin4
@revised on Jan22-2021
"""

import numpy as np
import leidenalg as la
from ..core.tool_base import ToolBase
from ..core.stereo_result import ClusterResult
from .neighbors import Neighbors
from ..preprocess.normalize import Normalizer
from .dim_reduce import DimReduce
import pandas as pd


class Clustering(ToolBase):
    """
    clustering bin-cell using nearest neighbor algorithm.
    """
    def __init__(self, data, method='louvain', outdir=None, dim_reduce_key='dim_reduce', n_neighbors=30,
                 normalize_key='cluster_normalize', normalize_method=None, nor_target_sum=10000, name='clustering'):
        """
        initialization

        :param data: anndata object
        :param method: louvain or leiden
        :param outdir: output directory
        :param dim_reduce_key: defined when running 'DimReduce' tool by setting 'name' property.
        :param n_neighbors: number of neighbors
        :param normalize_key: defined when running 'Normalize' tool by setting 'name' property.
        :param normalize_method: normalization method, Normalizer will be run before clustering if the param is set.
        :param nor_target_sum: summary of target
        :param name: name of this tool and will be used as a key when adding tool result to andata object.
        """
        super(Clustering, self).__init__(data=data, method=method, name=name)
        self.param = self.get_params(locals())
        self.outdir = outdir
        self.normakizer = Normalizer(self.data, method=normalize_method, inplace=False, target_sum=nor_target_sum,
                                     name=normalize_key) if normalize_method is not None else None
        self.dim_reduce_key = dim_reduce_key
        self.neighbors = n_neighbors

    def run_normalize(self):
        """
        normalize input data

        :return: normalized data
        """
        nor_x = self.normakizer.fit() if self.normakizer is not None else None
        return nor_x

    def get_dim_reduce_x(self, nor_x):
        """
        get dimensionality reduction results

        :param nor_x: normalized x array
        :return: pca results
        """
        if self.dim_reduce_key not in self.data.uns.keys():
            dim_reduce = DimReduce(self.data, method='pca', n_pcs=30, name=self.dim_reduce_key)
            dim_reduce.fit(nor_x)
        pca_x = self.data.uns[self.dim_reduce_key].x_reduce
        return pca_x

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
        nor_x = self.sparse2array() if self.normakizer is None else self.run_normalize()
        reduce_x = self.get_dim_reduce_x(nor_x)
        neighbor, nn_idx, nn_dist = self.run_neighbors(reduce_x)
        if self.method == 'leiden':
            cluster = self.run_knn_leiden(neighbor, nn_idx, nn_dist)
        else:
            cluster = self.run_louvain(neighbor, nn_idx, nn_dist)
        cluster = [str(i) for i in cluster]
        info = {'bins': self.data.obs_names, 'cluster': cluster}
        df = pd.DataFrame(info)
        result = ClusterResult(name=self.name, param=self.param, cluster_info=df)
        self.add_result(result, key_added=self.name)
        # TODO  added for find marker
        self.data.obs[self.name] = cluster
