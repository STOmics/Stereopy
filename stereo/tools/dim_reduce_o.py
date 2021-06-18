#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:dim_reduce.py
@time:2021/03/17

change log:
    2021/03/17 16:36:00  add filter functions, by Ping Qiu.
"""

from sklearn.decomposition import PCA
from anndata import AnnData
import numpy as np
import umap
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import TSNE
from ..log_manager import logger
from ..core.tool_base import ToolBase
from ..core.stereo_result import DimReduceResult


class DimReduce(ToolBase):
    """
    bin-cell dimensionality reduction
    """
    def __init__(self, data: AnnData, method='pca', n_pcs=2, min_variance=0.01, n_iter=250,
                 n_neighbors=5, min_dist=0.3, inplace=False, name='dim_reduce'):
        """
        initialization

        :param data: anndata object
        :param method: default pca, options are pca, tsen, umap, factor_analysis and low_variance
        :param n_pcs: the number of features for a return array after reducing.
        :param min_variance: minus variance
        :param n_iter: number of iteration
        :param n_neighbors: number of neighbors
        :param min_dist: param for UMAP method, the minus value of distance.
        :param inplace: inplace the input anndata if True
        :param name: name of this tool and will be used as a key when adding tool result to andata object.
        """
        self.params = self.get_params(locals())
        super(DimReduce, self).__init__(data=data, method=method, name=name)
        self.n_pcs = n_pcs
        self.min_variance = min_variance
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.check_param()
        self.result = DimReduceResult(name=name, param=self.params)

    def check_param(self):
        """
        Check whether the parameters meet the requirements.
        """
        super(DimReduce, self).check_param()
        if self.method.lower() not in ['pca', 'tsen', 'umap', 'factor_analysis', 'low_variance']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')

    def fit(self, exp_matrix=None):
        exp_matrix = exp_matrix if exp_matrix is not None else self.exp_matrix
        if self.method == 'low_variance':
            self.result.x_reduce = low_variance(exp_matrix, self.min_variance)
        elif self.method == 'factor_analysis':
            self.result.x_reduce = factor_analysis(exp_matrix, self.n_pcs)
        elif self.method == 'tsen':
            self.result.x_reduce = t_sne(exp_matrix, self.n_pcs, self.n_iter)
        elif self.method == 'umap':
            self.result.x_reduce = u_map(exp_matrix, self.n_pcs, self.n_neighbors, self.min_dist)
        else:
            pca_res = pca(exp_matrix, self.n_pcs)
            self.result.x_reduce = pca_res['x_pca']
            self.result.variance_ratio = pca_res['variance_ratio']
            self.result.variance_pca = pca_res['variance']
            self.result.pcs = pca_res['pcs']
        self.add_result(result=self.result, key_added=self.name)


def low_variance(x, threshold=0.01):
    """
    filter the features which have low variance between the samples.

    :param x: 2D array, shape (M, N)
    :param threshold: the min threshold of variance.
    :return: a new array which filtered the feature with low variance.
    """
    x_var = np.var(x, axis=0)
    var_index = np.where(x_var > threshold)[0]
    x = x[:, var_index]
    return x


def factor_analysis(x, n_pcs):
    """
    the dim reduce function of factor analysis

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :return:  ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
    """
    fa = FactorAnalysis(n_components=n_pcs)
    fa.fit(x)
    tran_x = fa.transform(x)
    return tran_x


def pca(x, n_pcs):
    """
    the dim reduce function of PCA

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :return:  ndarray of shape (n_samples, n_components) Embedding of the training data in low-dimensional space.
    """
    pca_obj = PCA(n_components=n_pcs)
    x_pca = pca_obj.fit_transform(x)
    variance = pca_obj.explained_variance_
    variance_ratio = pca_obj.explained_variance_ratio_
    pcs = pca_obj.components_.T
    return dict([('x_pca', x_pca), ('variance', variance), ('variance_ratio', variance_ratio), ('pcs', pcs)])


def t_sne(x, n_pcs, n_iter=200):
    """
    the dim reduce function of TSEN

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :param n_iter: the number of iterators.
    :return:  ndarray of shape (n_samples, n_components) Embedding of the training data in low-dimensional space.
    """
    tsen = TSNE(n_components=n_pcs, n_iter=n_iter)
    tsne_x = tsen.fit_transform(x)
    return tsne_x


def u_map(x, n_pcs, n_neighbors=5, min_dist=0.3):
    """
    the dim reduce function of UMAP

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :param n_neighbors: the number of neighbors
    :param min_dist: the min value of distance.
    :return: ndarray of shape (n_samples, n_components) Embedding of the training data in low-dimensional space.
    """
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_pcs, min_dist=min_dist)
    umap_x = umap_obj.fit_transform(x)
    return umap_x
