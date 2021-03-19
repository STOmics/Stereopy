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
"""
import numpy as np
from anndata import AnnData
from scipy import stats
from ..log_manager import logger
from ..core.tool_base import ToolBase


class Normalizer(ToolBase):
    """
    Normalizer of stereo.
    """
    def __init__(self, data, method='normalize_total', inplace=True, target_sum=1, name='normalize'):
        """
        :param data:
        :param method:
        :param inplace:
        :param target_sum:
        """
        super(Normalizer, self).__init__(data=data, method=method, inplace=inplace, name=name)
        self.target_num = target_sum
        self.check_param()

    def check_param(self):
        """
        Check whether the parameters meet the requirements.
        :return:
        """
        super(Normalizer, self).check_param()
        if self.method.lower() not in ['normalize_tital', 'quantile']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')

    def fit(self):
        """
        compute the scale value of self.exp_matrix.
        :return:
        """
        nor_res = None
        self.sparse2array()  # TODO: add  normalize of sparseMatrix
        if self.method == 'normalize_total':
            nor_res = normalize_total(self.exp_matrix, self.target_num)
        elif self.method == 'quantile':
            nor_res = quantile_norm(self.exp_matrix.T)
            nor_res = nor_res.T
        else:
            pass
        if nor_res and self.inplace and isinstance(self.data, AnnData):
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
    Normalize the columns of X to each have the same distribution.
    Given an expression matrix  of M genes by N samples, quantile normalization ensures all samples have the same
    spread of data (by construction).
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
