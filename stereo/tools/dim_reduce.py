#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:dim_reduce.py
@time:2021/03/17

change log:
    2021/03/17 16:36:00  add filter functions, by Ping Qiu.
    2021/05/20 rst supplement. by: qindanhua.
    2021/06/15 adjust for restructure base class . by: qindanhua.
"""

import pandas as pd
# from stereo.log_manager import logger
from stereo.core.tool_base import ToolBase
import numpy as np
from typing import Optional


class DimReduce(ToolBase):
    """
    bin-cell dimension reduction

    :param data: stereo expression data object
    :param method: default pca, options are pca, tsen, umap, factor_analysis and low_variance
    :param n_pcs: the number of features for a return array after reducing.
    :param min_variance: minus variance
    :param n_iter: number of iteration
    :param n_neighbors: number of neighbors
    :param min_dist: param for UMAP method, the minus value of distance.

    Examples
    --------

    >>> from stereo.tools.dim_reduce import DimReduce
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> dr = DimReduce(pd.DataFrame(X, columns=['gene1', 'gene2'], index=['c1', 'c2', 'c3', 'c4', 'c5', 'c6']))
    >>> dr.fit()
    >>> dr.result.matrix
           0         1
    0  13.754602 -3.000469
    1  12.671173 -2.698209
    2  13.715853 -1.961277
    3  11.762485 -1.426327
    4  12.028677 -0.300320
    5  12.879782 -0.918522
    >>> dr.method = 'umap'
    >>> dr.n_pcs = 3
    >>> dr.fit()

    Or
    >>> dr.u_map(X, n_pcs=3, n_neighbors=5, min_dist=0.3)
    """
    def __init__(
            self,
            data=None,
            method: str = 'pca',
            n_pcs: int = 2,
            min_variance: float = 0.01,
            n_iter: int = 250,
            n_neighbors: int = 5,
            min_dist: float = 0.3,
    ):
        super(DimReduce, self).__init__(data=data, method=method)
        self.n_pcs = n_pcs
        self.min_variance = min_variance
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['pca', 'tsen', 'umap', 'factor_analysis', 'low_variance']
        self._method_check(method, m_range)

    @property
    def n_pcs(self):
        return self._n_pcs

    @n_pcs.setter
    def n_pcs(self, n_pcs):
        if self._params_range_check(n_pcs, 2, len(self.data.gene_names), int):
            self._n_pcs = self.n_pcs
        else:
            self._n_pcs = n_pcs

    def _check_params(self):
        if not isinstance(self.n_iter, int):
            raise ValueError(f'{self.n_iter} should be int type')

    def fit(self, exp_matrix=None):

        from ..algorithm.dim_reduce import low_variance, factor_analysis, pca, t_sne, u_map

        self._check_params()
        exp_matrix = exp_matrix if exp_matrix is not None else self.data.exp_matrix
        if self.method == 'low_variance':
            x_reduce = low_variance(exp_matrix, self.min_variance)
        elif self.method == 'factor_analysis':
            x_reduce = factor_analysis(exp_matrix, self.n_pcs)
        elif self.method == 'tsen':
            x_reduce = t_sne(exp_matrix, self.n_pcs, self.n_iter)
        elif self.method == 'umap':
            x_reduce = u_map(exp_matrix, self.n_pcs, self.n_neighbors, self.min_dist)
        else:
            pca_res = pca(exp_matrix, self.n_pcs)
            x_reduce = pca_res['x_pca']
            # self.result.variance_ratio = pca_res['variance_ratio']
            # self.result.variance_pca = pca_res['variance']
            # self.result.pcs = pca_res['pcs']
        self.result = pd.DataFrame(x_reduce)
        return self.result
        # self.add_result(result=self.result, key_added=self.name)

    def plot_scatter(self,
                     gene_name: Optional[list],
                     file_path=None):
        """
        plot scatter after
        :param gene_name list of gene names
        :param file_path:
        :return:
        """
        # from scipy.sparse import issparse
        # if issparse(self.data.exp_matrix):
        #     self.data.exp_matrix = self.data.exp_matrix.toarray()
        from ..plots.scatter import plt, base_scatter, multi_scatter

        self.data.sparse2array()
        if len(gene_name) > 1:
            multi_scatter(self.result.matrix.values[:, 0], self.result.matrix.values[:, 1],
                          hue=np.array(self.data.sub_by_name(gene_name=gene_name).exp_matrix).T,
                          color_bar=True)
        else:
            base_scatter(self.result.matrix.values[:, 0], self.result.matrix.values[:, 1],
                         hue=np.array(self.data.sub_by_name(gene_name=gene_name).exp_matrix[:, 0]),
                         color_bar=True)
        if file_path:
            plt.savefig(file_path)

