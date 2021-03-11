#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:correlation.py
@time:2021/03/11
"""
import numpy as np
import pandas as pd
from scipy import stats


def pearson(arr1, arr2):
    """
    calculate pearson correlation between two numpy arrays.
    :param arr1: one array, the feature is a column. the shape is  `m * n`
    :param arr2: the other array, the feature is a column. the shape is `m * k`
    :return: a pearson score np.array , the shape is `k * n`
    """
    assert arr1.shape[0] == arr2.shape[0]
    n = arr1.shape[0]
    sums = np.multiply.outer(arr2.sum(0), arr1.sum(0))
    stds = np.multiply.outer(arr2.std(0), arr1.std(0))
    return (arr2.T.dot(arr1) - sums / n) / stds / n


def pearson_corr(df1, df2):
    """
    calculate pearson correlation between two dataframes.
    :param df1: one dataframe
    :param df2: the other dataframe
    :return: a pearson score dataframe, the index is the columns of `df1`, the columns is the columns of `df2`
    """
    v1, v2 = df1.values, df2.values
    corr_matrix = pearson(v1, v2)
    return pd.DataFrame(corr_matrix, df2.columns, df1.columns)


def spearmanr_corr(df1, df2):
    """
    calculate pearson correlation between two dataframes.
    :param df1: one dataframe
    :param df2: the other dataframe
    :return: a spearmanr score dataframe, the index is the columns of `df1`, the columns is the columns of `df2`
    """
    score, pvalue = stats.spearmanr(df1.values, df2.values)
    score = score[df1.shape[1]:, 0:df1.shape[1]]
    return pd.DataFrame(score, df2.columns, df1.columns)
