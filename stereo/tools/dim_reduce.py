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
from ..core.base import Base


class DimReduce(Base):
    def __init__(self, andata: AnnData, method='pca', n_pcs=2, min_variance=0.01, n_iter=200,
                 n_neighbors=5, min_dist=0.3, inplace=False):
        self.params = locals()
        super(DimReduce, self).__init__(data=andata, method=method, inplace=inplace)
        self.n_pcs = n_pcs
        self.min_variance = min_variance
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.check_param()

    def check_param(self):
        """
        Check whether the parameters meet the requirements.
        :return:
        """
        super(DimReduce, self).check_param()
        if self.method.lower() not in ['pca', 'tsen', 'umap', 'factor', 'low_variance']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')

    def fit(self):
        if self.method == 'pca':
            pass


def low_variance_filter(x, threshold=0.01):
    """
    filter the features which have low variance between the samples.
    :param x: 2D array, shape (M, N)
    :param threshold: the min threshold of variance.
    :return: a new array which filtered the feature with low variance.
    """
    x_var = np.var(x, axis=0)
    var_index = np.where(x_var > threshold)
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
    :return:  ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
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
    :return:  ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
    """
    tsen = TSNE(n_components=n_pcs, n_iter=n_iter)
    tsne_x = tsen.fit_transform(x)
    return tsne_x


def u_map(x, n_pcs, n_neighbors=5, min_dist=0.3):
    """
    the dim reduce fanction of UMAP
    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :param n_neighbors: the number of neighbors
    :param min_dist: the min value of distance.
    :return: ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
    """
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_pcs, min_dist=min_dist)
    umap_x = umap_obj.fit_transform(x)
    return umap_x
