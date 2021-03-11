#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:normalize.py
@time:2021/03/05

change log:
    add basic functions of normalization. by Ping Qiu.
"""
from scipy.sparse import issparse
import numpy as np


def normalize(andata, target_sum=1, inplace=True):
    """
    total count normalize the  adata.X to `target_sum` reads per cell, so that counts become comparable among cells.
    :param andata: AnnData object.
    :param target_sum: the number of reads per cell after normalization.
    :param inplace: whether inplace the original adata or return a new anndata.
    :return:
    """
    exp_matrix = andata.X
    if issparse(exp_matrix):
        exp_matrix = exp_matrix.toarray()
    nor_exp_matrix = exp_matrix * target_sum / exp_matrix.sum(axis=1)[:, np.newaxis]
    if inplace:
        andata.X = nor_exp_matrix
    return nor_exp_matrix if not inplace else None


def log1p(andata, inplace=True):
    """
    Logarithmize the data. log(1 + x)
    :param andata: AnnData object.
    :param inplace: whether inplace the original adata or return a new anndata.
    :return:
    """
    exp_matrix = andata.X.copy()
    if issparse(exp_matrix):
        exp_matrix = exp_matrix.toarray()
    log_x = np.log1p(exp_matrix, out=exp_matrix)
    if inplace:
        andata.X = log_x
    return log_x if not inplace else None
