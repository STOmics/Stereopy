#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
"""


import numpy as np
from scipy import stats
import scipy.spatial as spatial
from functools import singledispatch
from scipy.sparse import spmatrix
from sklearn.utils import sparsefuncs


@singledispatch
def normalize_total(x, target_sum):
    """
        total count normalize the data to `target_sum` reads per cell, so that counts become comparable among cells.

        :param x: 2D array, shape (M, N), which row is cells and column is genes.
        :param target_sum: the number of reads per cell after normalization.
        :return: the normalized data.
        """
    pass


@normalize_total.register(np.ndarray)
def _(x, target_sum):
    nor_x = x * target_sum / x.sum(axis=1)[:, np.newaxis]
    return nor_x


@normalize_total.register(spmatrix)
def _(x, target_sum):
    x = x.astype(np.float64)
    counts = target_sum / np.ravel(x.sum(1))
    sparsefuncs.inplace_row_scale(x, counts)
    return x


def quantile_norm(x):
    """
    Normalize the columns of X to each have the same distribution. Given an expression matrix  of M genes by N samples,
    quantile normalization ensures all samples have the same spread of data (by construction).

    :param x: 2D array of float, shape (M, N)
    :return: The normalized data.
    """
    quantiles = np.mean(np.sort(x, axis=0), axis=1)
    ranks = np.apply_along_axis(stats.rankdata, 0, x)
    rank_indices = ranks.astype(int) - 1
    xn = quantiles[rank_indices]
    return xn


def log1p(x):
    """
    Logarithmize the data. log(1 + x)

    :param x: 2D array, shape (M, N).
    :return:
    """
    log_x = np.log1p(x, out=x)
    return log_x


def zscore_disksmooth(x, position, r):
    """
    for each position, given a radius, calculate the z-score within this circle as final normalized value.

    :param x: 2D array, shape (M, N), which row is cells and column is genes.
    :param position: each cell's position , [[x1, y1], [x2, y2], ..., M]
    :param r: radius
    :return: normalized data, shape (M, N), which row is cells and column is genes.
    """
    position = position.astype(np.int32)
    point_tree = spatial.cKDTree(position)
    x = x.astype(np.float32)
    mean_bin = x.mean(1)
    mean_bin = np.array(mean_bin)
    std_bin = np.std(x, axis=1)
    zscore = []
    for i in range(len(position)):
        current_neighbor = point_tree.query_ball_point(position[i], r)
        current_neighbor.remove(i)
        if len(current_neighbor) > 0:
            mean_bins = np.mean(mean_bin[current_neighbor])
            std_bins = np.mean(std_bin[current_neighbor])
            zscore.append((x[i] - mean_bins) / std_bins + 1)
        else:
            mean_bins = mean_bin[i]
            std_bins = std_bin[i]
            zscore.append((x[i] - mean_bins) / std_bins + 1)
    return np.array(zscore)
