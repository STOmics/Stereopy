#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:normalize.py
@time:2021/03/05

change log:
    add basic functions of normalization. by Ping Qiu. 2021/03/05
    Refactor the code and add the quantile_norm function. by Ping Qiu. 2021/03/17
    add the zscore_disksmooth function. by Ping Qiu. 2021/05/28
"""
import numpy as np
from anndata import AnnData
from scipy import stats
from ..log_manager import logger
from ..core.tool_base import ToolBase
import scipy.spatial as spatial


class Normalizer(ToolBase):
    """
    Normalizer of stereo.
    """
    def __init__(self, data, method='normalize_total', inplace=True, target_sum=1, name='normalize', r=20):
        super(Normalizer, self).__init__(data=data, method=method, name=name)
        self.target_num = target_sum
        self.inplace = inplace
        self.check_param()
        self.position = data.obsm['spatial']
        self.r = r

    def check_param(self):
        """
        Check whether the parameters meet the requirements.

        """
        super(Normalizer, self).check_param()
        if self.method.lower() not in ['normalize_total', 'quantile', 'zscore_disksmooth']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')

    @property
    def methods(self):
        return self.method

    @methods.setter
    def methods(self, v):
        if v not in ['normalize_total', 'quantile']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')
        self.method = v

    def fit(self):
        """
        compute the scale value of self.exp_matrix.
        """
        nor_res = None
        self.sparse2array()  # TODO: add  normalize of sparseMatrix
        if self.method == 'normalize_total':
            nor_res = normalize_total(self.exp_matrix, self.target_num)
        elif self.method == 'quantile':
            nor_res = quantile_norm(self.exp_matrix.T)
            nor_res = nor_res.T
        elif self.method == 'zscore_disksmooth':
            nor_res = zscore_disksmooth(self.exp_matrix, self.position, self.r)
        else:
            pass
        if nor_res is not None and self.inplace and isinstance(self.data, AnnData):
            self.data.X = nor_res
        return nor_res


def normalize_total(x, target_sum):
    """
    total count normalize the data to `target_sum` reads per cell, so that counts become comparable among cells.

    :param x: 2D array, shape (M, N), which row is cells and column is genes.
    :param target_sum: the number of reads per cell after normalization.
    :return: the normalized data.
    """
    nor_x = x * target_sum / x.sum(axis=1)[:, np.newaxis]
    return nor_x


def quantile_norm(x):
    """
    Normalize the columns of X to each have the same distribution. Given an expression matrix  of M genes by N samples, quantile normalization ensures all samples have the same spread of data (by construction).

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
