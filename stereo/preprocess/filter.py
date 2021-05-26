#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:filter.py
@time:2021/03/09

change log:
    2021/03/20 16:36:00  add filter functions, by Ping Qiu.
"""
from scipy.sparse import issparse
from anndata import AnnData
import numpy as np


def filter_cells(adata, min_gene=None, max_gene=None, n_genes_by_counts=None, pct_counts_mt=None,
                 cell_list=None, obs_key=None, inplace=True):
    """
    filter cells based on numbers of genes expressed .

    :param adata: AnnData object
    :param min_gene: Minimum number of genes expressed for a cell pass filtering.
    :param max_gene: Maximum number of genes expressed for a cell pass filtering.
    :param n_genes_by_counts: Minimum number of  n_genes_by_counts for a cell pass filtering.
    :param pct_counts_mt: Maximum number of  pct_counts_mt for a cell pass filtering.
    :param cell_list: the list of cells which will be filtered.
    :param obs_key: the key of adata.obs to find the name of cell. if None, adata.obs.index replace.
    :param inplace: whether inplace the original adata or return a new anndata.
    :return: AnnData object if inplace id true else none
    """
    assert isinstance(adata, AnnData)
    adata = adata if inplace else adata.copy()
    if min_gene is None and max_gene is None and cell_list is None:
        raise ValueError('please set `min_gene` or `max_gene` or `cell_list` or all of them.')
    if 'total_counts' not in adata.obs_keys():
        exp_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
        genes_per_cell = exp_matrix.sum(axis=1)
        adata.obs['total_counts'] = genes_per_cell
    if min_gene:
        cell_subset = adata.obs['total_counts'] >= min_gene
        adata._inplace_subset_obs(cell_subset)
    if max_gene:
        cell_subset = adata.obs['total_counts'] <= max_gene
        adata._inplace_subset_obs(cell_subset)
    if n_genes_by_counts:
        cell_subset = adata.obs['n_genes_by_counts'] >= n_genes_by_counts
        adata._inplace_subset_obs(cell_subset)
    if pct_counts_mt:
        if 'pct_counts_mt' not in adata.obs_keys():
            mt_index = adata.var_names.str.startswith('MT-')
            adata.obs['pct_counts_mt'] = adata.X[:, mt_index].sum(1)
        cell_subset = adata.obs['pct_counts_mt'] <= pct_counts_mt
        adata._inplace_subset_obs(cell_subset)
    if cell_list:
        cell_subset = adata.obs.index.isin(cell_list) if not obs_key else adata.obs[obs_key].isin(cell_list)
        adata._inplace_subset_obs(~cell_subset)
    return adata if not inplace else None


def filter_genes(adata, min_cell=None, max_cell=None, gene_list=None, inplace=True):
    """
    filter genes based on the numbers of cells.

    :param adata: AnnData object.
    :param min_cell: Minimum number of cells for a gene pass filtering.
    :param max_cell: Maximun number of cells for a gene pass filtering.
    :param gene_list: the list of genes which will be filtered.
    :param inplace: whether inplace the original adata or return a new anndata.
    :return: AnnData object if inplace id true else none
    """
    assert isinstance(adata, AnnData)
    adata = adata if inplace else adata.copy()
    if min_cell is None and max_cell is None and gene_list is None:
        raise ValueError('please set `min_cell` or `max_cell` or `gene_list` or both of them.')
    if 'n_cells' not in adata.var_keys():
        exp_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
        cells_per_gene = exp_matrix.sum(axis=0)
        adata.var['n_cells'] = cells_per_gene
    if min_cell:
        gene_subset = adata.var['n_cells'] >= min_cell
        adata._inplace_subset_var(gene_subset)
    if max_cell:
        gene_subset = adata.var['n_cells'] <= max_cell
        adata._inplace_subset_var(gene_subset)
    if gene_list:
        gene_subset = adata.var.index.isin(gene_list)
        adata._inplace_subset_var(~gene_subset)
    return adata if not inplace else None


def filter_coordinates(adata, min_x=None, max_x=None, min_y=None, max_y=None, inplace=True):
    """
    filter cells based on the coordinates of cells.

    :param adata: AnnData object.
    :param min_x: Minimum of x for a cell pass filtering.
    :param max_x: Maximum of x for a cell pass filtering.
    :param min_y: Minimum of y for a cell pass filtering.
    :param max_y: Maximum of y for a cell pass filtering.
    :param inplace: whether inplace the original adata or return a new anndata.
    :return: AnnData object if inplace id true else none
    """
    assert isinstance(adata, AnnData)
    assert 'spatial' in adata.obsm_keys()
    adata = adata if inplace else adata.copy()
    none_param = [i for i in [min_x, min_y, max_x, max_y] if i is None]
    if len(none_param) == 4:
        raise ValueError('Only provide one of the optional parameters `min_x`, `min_y`, `max_x`, `max_y` per call.')
    pos = adata.obsm['spatial']
    obs_subset = np.full(pos.shape[0], True)
    if min_x:
        obs_subset &= pos[:, 0] >= min_x
    if min_y:
        obs_subset &= pos[:, 1] >= min_y
    if max_x:
        obs_subset &= pos[:, 0] <= max_x
    if max_y:
        obs_subset &= pos[:, 1] <= max_y
    adata._inplace_subset_obs(obs_subset)
    adata.var['n_cells'] = adata.X.toarray() if issparse(adata.X) else adata.X.sum(axis=0)
    adata.obs['n_genes'] = adata.X.toarray() if issparse(adata.X) else adata.X.sum(axis=1)
    return adata if not inplace else None
