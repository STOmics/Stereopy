#!/usr/bin/env python3
# coding: utf-8
"""
@file: filter.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/07/06  create file.
"""
import numpy as np
import copy
from .qc import cal_total_counts, cal_pct_counts_mt, cal_n_genes_by_counts, cal_n_cells_by_counts, cal_n_cells


def filter_cells(
        data,
        min_gene=None,
        max_gene=None,
        min_n_genes_by_counts=None,
        max_n_genes_by_counts=None,
        pct_counts_mt=None,
        cell_list=None,
        inplace=True):
    """
    filter cells based on numbers of genes expressed.

    :param data: StereoExpData object
    :param min_gene: Minimum number of genes expressed for a cell pass filtering.
    :param max_gene: Maximum number of genes expressed for a cell pass filtering.
    :param min_n_genes_by_counts: Minimum number of  n_genes_by_counts for a cell pass filtering.
    :param max_n_genes_by_counts: Maximum number of  n_genes_by_counts for a cell pass filtering.
    :param pct_counts_mt: Maximum number of  pct_counts_mt for a cell pass filtering.
    :param cell_list: the list of cells which will be filtered.
    :param inplace: whether inplace the original data or return a new data.
    :return: StereoExpData object.
    """
    data = data if inplace else copy.deepcopy(data)
    if min_gene is None and max_gene is None and cell_list is None and min_n_genes_by_counts is None \
            and max_n_genes_by_counts is None and pct_counts_mt is None:
        raise ValueError('At least one filter must be set.')
    if data.cells.total_counts is None:
        total_counts = cal_total_counts(data.exp_matrix)
        data.cells.total_counts = total_counts
    if min_gene:
        cell_subset = data.cells.total_counts >= min_gene
        data.sub_by_index(cell_index=cell_subset)
    if max_gene:
        cell_subset = data.cells.total_counts <= max_gene
        data.sub_by_index(cell_index=cell_subset)
    if min_n_genes_by_counts:
        if data.cells.n_genes_by_counts is None:
            data.cells.n_genes_by_counts = cal_n_genes_by_counts(data.exp_matrix)
        cell_subset = data.cells.n_genes_by_counts >= min_n_genes_by_counts
        data.sub_by_index(cell_index=cell_subset)
    if max_n_genes_by_counts:
        if data.cells.n_genes_by_counts is None:
            data.cells.n_genes_by_counts = cal_n_genes_by_counts(data.exp_matrix)
        cell_subset = data.cells.n_genes_by_counts <= max_n_genes_by_counts
        data.sub_by_index(cell_index=cell_subset)
    if pct_counts_mt:
        if data.cells.pct_counts_mt is None:
            data.cells.pct_counts_mt = cal_pct_counts_mt(data, data.exp_matrix, data.cells.total_counts)
        cell_subset = data.cells.pct_counts_mt <= pct_counts_mt
        data.sub_by_index(cell_index=cell_subset)
    if cell_list:
        cell_subset = np.isin(data.cells.cell_name, cell_list)
        data.sub_by_index(cell_index=cell_subset)
    return data


def filter_genes(data, min_cell=None, max_cell=None, gene_list=None, inplace=True):
    """
    filter genes based on the numbers of cells.

    :param data: StereoExpData object.
    :param min_cell: Minimum number of cells for a gene pass filtering.
    :param max_cell: Maximun number of cells for a gene pass filtering.
    :param gene_list: the list of genes which will be filtered.
    :param inplace: whether inplace the original data or return a new data.
    :return: StereoExpData object.
    """
    data = data if inplace else copy.deepcopy(data)
    if min_cell is None and max_cell is None and gene_list is None:
        raise ValueError('please set `min_cell` or `max_cell` or `gene_list` or both of them.')
    if data.genes.n_cells is None:
        data.genes.n_cells = cal_n_cells(data.exp_matrix)
    if min_cell:
        gene_subset = data.genes.n_cells >= min_cell
        data.sub_by_index(gene_index=gene_subset)
    if max_cell:
        gene_subset = data.genes.n_cells <= max_cell
        data.sub_by_index(gene_index=gene_subset)
    if gene_list:
        gene_subset = np.isin(data.gene_names, gene_list)
        data.sub_by_index(gene_index=gene_subset)
    return data


def filter_coordinates(data, min_x=None, max_x=None, min_y=None, max_y=None, inplace=True):
    """
    filter cells based on the coordinates of cells.

    :param data: StereoExpData object.
    :param min_x: Minimum of x for a cell pass filtering.
    :param max_x: Maximum of x for a cell pass filtering.
    :param min_y: Minimum of y for a cell pass filtering.
    :param max_y: Maximum of y for a cell pass filtering.
    :param inplace: whether inplace the original data or return a new data.
    :return: StereoExpData object
    """
    data = data if inplace else copy.deepcopy(data)
    none_param = [i for i in [min_x, min_y, max_x, max_y] if i is None]
    if len(none_param) == 4:
        raise ValueError('Only provide one of the optional parameters `min_x`, `min_y`, `max_x`, `max_y` per call.')
    pos = data.position
    obs_subset = np.full(pos.shape[0], True)
    if min_x:
        obs_subset &= pos[:, 0] >= min_x
    if min_y:
        obs_subset &= pos[:, 1] >= min_y
    if max_x:
        obs_subset &= pos[:, 0] <= max_x
    if max_y:
        obs_subset &= pos[:, 1] <= max_y
    data.sub_by_index(cell_index=obs_subset)
    data.genes.n_cells = cal_n_cells(data.exp_matrix)
    return data
