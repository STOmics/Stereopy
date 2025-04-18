#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:qc.py
@time:2021/03/26
"""
import numpy as np
from scipy.sparse import issparse

from stereo.core.stereo_exp_data import StereoExpData


def cal_qc(data: StereoExpData, use_raw, layer):
    """
    calculate three QC index including the number of genes expressed in the count matrix, the total counts per cell
    and the percentage of counts in mitochondrial genes.

    :param data: the StereoExpData object.
    :return: StereoExpData object storing quality control results.
    """
    cal_cells_indicators(data, use_raw, layer)
    cal_genes_indicators(data, use_raw, layer)
    return data


def cal_cells_indicators(data: StereoExpData, use_raw, layer):
    exp_matrix = data.get_exp_matrix(use_raw, layer)
    # exp_matrix = data.exp_matrix
    data.cells.total_counts = cal_total_counts(exp_matrix)
    data.cells.n_genes_by_counts = cal_n_genes_by_counts(exp_matrix)
    if data.genes.real_gene_name is not None:
        gene_names = data.genes.real_gene_name
    else:
        gene_names = data.gene_names
    data.cells.pct_counts_mt = cal_pct_counts_mt(exp_matrix, gene_names)
    return data


def cal_genes_indicators(data: StereoExpData, use_raw, layer):
    exp_matrix = data.get_exp_matrix(use_raw, layer)
    # exp_matrix = data.exp_matrix
    data.genes.n_cells = cal_n_cells(exp_matrix)
    data.genes.n_counts = cal_per_gene_counts(exp_matrix)
    data.genes.mean_umi = cal_gene_mean_umi(exp_matrix)
    return data


def cal_total_counts(exp_matrix):
    """
    calculate the total gene counts of per cell.

    :param exp_matrix: the express matrix.
    :return:
    """
    return np.array(exp_matrix.sum(1)).reshape(-1)


def cal_per_gene_counts(exp_matrix):
    """
    calculate the total counts of per gene.

    :param exp_matrix: the express matrix.
    :return:
    """
    return np.array(exp_matrix.sum(axis=0)).reshape(-1)


def cal_n_cells_by_counts(exp_matrix):
    """
    total counts of each gene.

    :param exp_matrix: the express matrix.
    :return:
    """
    return np.array(exp_matrix.sum(0)).reshape(-1)


def cal_n_cells(exp_matrix):
    """
    Number of cells that occur in each gene.

    :param exp_matrix: the express matrix.
    :return:
    """
    return exp_matrix.getnnz(axis=0) if issparse(exp_matrix) else np.count_nonzero(exp_matrix, axis=0)


def cal_gene_mean_umi(exp_matrix):
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    # gene_mean_umi = data.genes.n_counts / data.genes.n_cells
    n_counts = cal_per_gene_counts(exp_matrix)
    n_cells = cal_n_cells(exp_matrix)
    gene_mean_umi = n_counts / n_cells
    flag = np.isnan(gene_mean_umi) | np.isinf(gene_mean_umi)
    gene_mean_umi[flag] = 0
    np.seterr(**old_settings)
    return gene_mean_umi


def cal_n_genes_by_counts(exp_matrix):
    return exp_matrix.getnnz(axis=1) if issparse(exp_matrix) else np.count_nonzero(exp_matrix, axis=1)


def cal_pct_counts_mt(exp_matrix, gene_names):
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    total_counts = cal_total_counts(exp_matrix)
    mt_index = np.char.startswith(np.char.lower(gene_names), prefix='mt-')
    mt_count = np.array(exp_matrix[:, mt_index].sum(1)).reshape(-1)
    pct_counts_mt = mt_count / total_counts * 100
    flag = np.isnan(pct_counts_mt) | np.isinf(pct_counts_mt)
    pct_counts_mt[flag] = 0
    np.seterr(**old_settings)
    return pct_counts_mt
