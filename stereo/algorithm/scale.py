#!/usr/bin/env python3
# coding: utf-8
"""
@author: Junhao Xu  xujunhao@genomics.cn
"""


import numba
import numpy as np
from functools import singledispatch
from scipy import sparse
from scipy.sparse import spmatrix
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from ..log_manager import logger


@singledispatch
def scale(x, zero_center, max_value):
    """
        Scale the data to unit variance and zero mean.

        :param x: 2D array, shape (M, N), which row is cells and column is genes.
        :param zero_center: Ignore zero variables if `False`
        :param max_value: Truncate to this value after scaling. If `None`, do not truncate.
        :return: the scaled data.
    """
    return scale_array(x, zero_center=zero_center, max_value=max_value)


@scale.register(np.ndarray)
def scale_array(x, zero_center, max_value):
    if not zero_center and max_value is not None:
        logger.info('Be careful when using `max_value` without `zero_center` is False')
    
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float)
    
    mean, var = _get_mean_var(x)
    std = np.sqrt(var)
    std[std == 0] = 1
    if issparse(x):
        if zero_center:
            logger.error('Cannot zero-center sparse matrix.')
        sparsefuncs.inplace_column_scale(x, 1 / std)
    else:
        if zero_center:
            x -= mean
        x /= std

    if max_value is not None:
        logger.info(f'Truncate at max_value {max_value}')
        x[x > max_value] = max_value

    return x


@scale.register(spmatrix)
def scale_sparse(x, zero_center, max_value):
    if zero_center:
        x = x.toarray()
    return scale_array(x, zero_center=zero_center, max_value=max_value)


def _get_mean_var(x, *, axis=0):
    """
    Calculate mean and var values of array.

    :param x: 2D array of float, shape (M, N)
    :return: means and variances values.
    """
    if sparse.issparse(x):
        datam = x.data
        indices = x.indices
        major_len, minor_len = x.shape[::-1]
        mean, var = sparse_mean_variance(datam, indices, major_len, minor_len)
    else:
        mean = np.mean(x, axis=0, dtype=np.float64)
        mean_sq = np.multiply(x, x).mean(axis=0, dtype=np.float64)
        var = mean_sq - mean**2

    var *= x.shape[0] / (x.shape[0] - 1)

    return mean, var


@numba.njit(cache=True)
def sparse_mean_variance(data, indices, major_len, minor_len):
    """
    Calculate mean and var values of sparse array(csr matrix) for the minor axis.

    :param data: array.
    :param indices: indices.
    :param major_len: major len.
    :param minor_len: minor len.
    :return: means and variances values.
    """
    
    dtype = np.float64
    non_zero = indices.shape[0]

    means = np.zeros(minor_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    counts = np.zeros(minor_len, dtype=np.int64)

    for i in range(non_zero):
        col_ind = indices[i]
        diff = data[i] - means[col_ind]
        variances[col_ind] += diff * diff
        counts[col_ind] += 1

    for i in range(minor_len):
        variances[i] += (major_len - counts[i]) * means[i] ** 2
        variances[i] /= major_len


    return means, variances