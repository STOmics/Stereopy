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


def cal_qc(data):
    """
    calculate three qc index including the number of genes expressed in the count matrix, the total counts per cell
    and the percentage of counts in mitochondrial genes.

    :param data: the StereoExpData object.
    :return: StereoExpData object storing quality control results.
    """
    exp_matrix = data.exp_matrix
    total_count = cal_total_counts(exp_matrix)
    n_gene_by_count = cal_n_genes_by_counts(exp_matrix)
    pct_counts_mt = cal_pct_counts_mt(data, exp_matrix, total_count)
    data.cells.total_counts = total_count
    data.cells.pct_counts_mt = pct_counts_mt
    data.cells.n_genes_by_counts = n_gene_by_count
    return data


def cal_total_counts(exp_matrix):
    """
    calculate the total gene counts of per cell.

    :param exp_matrix: the express matrix.
    :return:
    """
    total_count = np.array(exp_matrix.sum(1)).reshape(-1)
    return total_count


def cal_per_gene_counts(exp_matrix):
    """
    calculate the total counts of per gene.

    :param exp_matrix: the express matrix.
    :return:
    """
    gene_counts = np.array(exp_matrix.sum(axis=0)).reshape(-1)
    return gene_counts


def cal_n_cells_by_counts(exp_matrix):
    """
    total counts of each gene.

    :param exp_matrix: the express matrix.
    :return:
    """
    n_cells_by_counts = np.array(exp_matrix.sum(0)).reshape(-1)
    return n_cells_by_counts


def cal_n_cells(exp_matrix):
    """
    Number of cells that occur in each gene.

    :param exp_matrix: the express matrix.
    :return:
    """
    n_cells = exp_matrix.getnnz(axis=0) if issparse(exp_matrix) else np.count_nonzero(exp_matrix, axis=0)
    return n_cells


def cal_n_genes_by_counts(exp_matrix):
    n_genes_by_counts = exp_matrix.getnnz(axis=1) if issparse(exp_matrix) else np.count_nonzero(exp_matrix, axis=1)
    return n_genes_by_counts


def cal_pct_counts_mt(data, exp_matrix, total_count):
    if total_count is None:
        total_count = cal_total_counts(exp_matrix)
    mt_index = np.char.startswith(np.char.lower(data.gene_names), prefix='mt-')
    mt_count = np.array(exp_matrix[:, mt_index].sum(1)).reshape(-1)
    pct_counts_mt = mt_count / total_count * 100
    return pct_counts_mt
