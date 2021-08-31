#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
"""

import numpy as np
# import umap
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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
    Principal component analysis.

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
    import umap
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_pcs, min_dist=min_dist)
    umap_x = umap_obj.fit_transform(x)
    return umap_x
