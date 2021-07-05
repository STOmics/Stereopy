#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:qc.py
@time:2021/03/26
"""
from scipy.sparse import issparse
import numpy as np
from ..core.stereo_exp_data import StereoExpData


def cal_qc(data: StereoExpData):
    """
    calculate three qc index including the number of genes expressed in the count matrix, the total counts per cell
    and the percentage of counts in mitochondrial genes.

    :param data: the StereoExpData object.
    :return: StereoExpData object storing quality control results.
    """
    exp_matrix = data.exp_matrix
    total_count = np.array(exp_matrix.sum(1)).reshape(-1)
    n_gene_by_count = exp_matrix.getnnz(axis=1) if issparse(exp_matrix) else np.count_nonzero(exp_matrix, axis=1)
    mt_index = np.char.startswith(np.char.lower(data.gene_names), prefix='mt-')
    mt_count = np.array(exp_matrix[:, mt_index].sum(1)).reshape(-1)
    data.cells.total_counts = total_count
    data.cells.pct_counts_mt = mt_count / total_count * 100
    data.cells.n_genes_by_counts = n_gene_by_count
    return data
