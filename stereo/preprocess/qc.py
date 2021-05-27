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


def cal_qc(andata):
    """
    calculate three qc index including the number of genes expressed in the count matrix, the total counts per cell and the percentage of counts in mitochondrial genes.

    :param: andata
    :return: anndata object storing quality control results
    """
    exp_matrix = andata.X.toarray() if issparse(andata.X) else andata.X
    total_count = exp_matrix.sum(1)
    n_gene_by_count = np.count_nonzero(exp_matrix, axis=1)
    mt_index = andata.var_names.str.lower().str.startswith('mt-')
    mt_count = np.array(andata.X[:, mt_index].sum(1)).reshape(-1)
    andata.obs['total_counts'] = total_count
    andata.obs['pct_counts_mt'] = mt_count / total_count * 100
    andata.obs['n_genes_by_counts'] = n_gene_by_count
    return andata
