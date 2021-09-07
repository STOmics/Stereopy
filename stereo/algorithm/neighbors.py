#!/usr/bin/env python3
# coding: utf-8
"""
@file: neighbors.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/9/3 create file.
"""

from ..utils.data_helper import select_group
from ..core.tool_base import ToolBase
from ..log_manager import logger
from tqdm import tqdm
from typing import Union, Sequence
import numpy as np
from ..plots.marker_genes import plot_marker_genes_text, plot_marker_genes_heatmap
from ..algorithm.neighbors import Neighbors
from sklearn.metrics import pairwise_distances
from typing import Optional
import pandas as pd

class Neighbors(ToolBase):
    """
    a tool of finding maker gene
    for each group, find statistical test different genes between one group and the rest groups using t-test or wilcoxon_test
    :param data: expression matrix, StereoExpData object
    :param groups: group information matrix, at least two columns, treat first column as sample name, and the second as
    group name e.g pd.DataFrame({'bin_cell': ['cell_1', 'cell_2'], 'cluster': ['1', '2']})
    :param case_groups: default all clusters
    :param control_groups: rest of groups
    :param method: t-test or wilcoxon_test
    :param corr_method: correlation method
    Examples
    --------
    >>> from stereo.tools.find_markers import FindMarker
    >>> fm = FindMarker()
    """

    def __init__(
            self,
            method: str = 'umap',
            pca_x: Optional[pd.DataFrame] = None,
            n_neighbors: Optional[int] = 30,
            n_pcs: Optional[int] = 40,
            metric: Optional[str] = 'euclidean',
            knn: Optional[bool] = True,
            find_neighbors: bool = False,
            clustering: bool = True,
            pca_pcs: Optional[int] = 50,
    ):
        super(Neighbors, self).__init__(pca_x=pca_x)
        self.n_pcs = n_pcs,
        self.metric = metric,
        self.method = method,
        self.knn = knn,
        self.find_neighbors = find_neighbors,
        self.n_neighbors = n_neighbors,

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
        return self.pca_x

    @pca_x.setter
    def pca_x(self, pca_x):
        input_df = self._check_input_data(pca_x)
        self.pca_x = input_df

    def fit(self):
        """
        find neighbors

        :param x: input data array. [[cell_1, gene_count_1], [cell_1, gene_count_2], ..., [cell_M, gene_count_N]]
        :return: neighbor object and neighbor cluster info
        """
        neighbor = Neighbors(self.pca_x, self.n_neighbors, self.n_pcs, self.metric, self.method, self.knn)
        neighbor.check_setting()
        self.pca_x = neighbor.choose_x()
        use_dense_distances = (self.metric == 'euclidean' and self.pca_x.shape[0] < 8192) or not self.knn
        dists = self.pca_x
        if use_dense_distances:
            dists = pairwise_distances(self.pca_x, metric=self.metric, )
            knn_indices, knn_distances = neighbor.get_indices_distances_from_dense_matrix(dists, self.n_neighbors)
            if self.knn:
                dists = neighbor.get_parse_distances_numpy(
                    knn_indices, knn_distances, self.pca_x.shape[0],
                )
        else:
            if self.pca_x.shape[0] < 4096:
                dists = pairwise_distances(self.pca_x, metric=self.metric)
            self.metric = 'precomputed'
            knn_indices, knn_distances, forest = neighbor.compute_neighbors_umap(dists, )
        if not use_dense_distances or self.method in {'umap'}:
            connectivities = neighbor.get_connectivities_umap(knn_indices, knn_distances)
            dists = neighbor.get_parse_distances_umap(knn_indices, knn_distances, )
        if self.method == 'gauss':
            neighbor.compute_connectivities_diffmap(dists)
        return neighbor, dists, connectivities
