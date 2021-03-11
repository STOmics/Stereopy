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
from typing import Union
from anndata import AnnData
from pandas import DataFrame


def normalize(data: Union[AnnData, DataFrame], target_sum=1, inplace=True):
    """
    total count normalize the data to `target_sum` reads per cell, so that counts become comparable among cells.
    :param data: AnnData object or Dataframe, which row is cells and column is genes.
    :param target_sum: the number of reads per cell after normalization.
    :param inplace: whether inplace the original anndata or return a new anndata. only use for AnnData format.
    :return:
    """
    exp_matrix = data.X if isinstance(data, AnnData) else data.values
    if issparse(exp_matrix):
        exp_matrix = exp_matrix.toarray()
    nor_exp_matrix = exp_matrix * target_sum / exp_matrix.sum(axis=1)[:, np.newaxis]
    if inplace and isinstance(data, AnnData):
        data.X = nor_exp_matrix
        return None
    return nor_exp_matrix


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
