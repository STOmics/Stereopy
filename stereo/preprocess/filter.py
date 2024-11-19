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
import copy
from typing import (
    Union,
    List,
    Tuple
)

import numpy as np
import pandas as pd

from stereo.core.stereo_exp_data import StereoExpData
from .qc import (
    cal_cells_indicators,
    cal_genes_indicators
)


def filter_cells(
        data: StereoExpData,
        min_counts=None,
        max_counts=None,
        min_genes=None,
        max_genes=None,
        pct_counts_mt=None,
        cell_list=None,
        excluded=False,
        filter_raw=True,
        use_raw=True,
        layer=None,
        inplace=True
    ):
    """
    filter cells based on numbers of genes expressed.

    :param data: StereoExpData object
    :param min_counts: Minimum number of genes expressed for a cell pass filtering.
    :param max_counts: Maximum number of genes expressed for a cell pass filtering.
    :param min_genes: Minimum number of  n_genes_by_counts for a cell pass filtering.
    :param max_genes: Maximum number of  n_genes_by_counts for a cell pass filtering.
    :param pct_counts_mt: Maximum number of  pct_counts_mt for a cell pass filtering.
    :param cell_list: the list of cells which will be filtered.
    :param excluded: set it to True to exclude the cells specified by parameter `cell_list` while False to include.
    :param filter_raw: whether filter the raw data.
    :param inplace: whether inplace the original data or return a new data.

    :return: StereoExpData object.
    """
    data = data if inplace else copy.deepcopy(data)
    if min_counts is None and max_counts is None and min_genes is None \
            and max_genes is None and pct_counts_mt is None and cell_list is None:
        # raise ValueError('At least one filter must be set.')
        raise ValueError('Please set any of `min_counts`, `max_counts`, `min_genes`, `max_genes`, `pct_counts_mt` and `cell_list`')
    if min_counts is not None or max_counts is not None \
        or min_genes is not None or max_genes is not None or pct_counts_mt is not None:
        cal_cells_indicators(data, use_raw, layer)
    cell_subset = np.ones(data.cells.size, dtype=bool)
    if min_counts:
        cell_subset &= data.cells.total_counts >= min_counts
    if max_counts:
        cell_subset &= data.cells.total_counts <= max_counts
    if min_genes:
        cell_subset &= data.cells.n_genes_by_counts >= min_genes
    if max_genes:
        cell_subset &= data.cells.n_genes_by_counts <= max_genes
    if pct_counts_mt:
        cell_subset &= data.cells.pct_counts_mt <= pct_counts_mt
    if cell_list is not None:
        if excluded:
            cell_subset &= ~np.isin(data.cells.cell_name, cell_list)
        else:
            cell_subset &= np.isin(data.cells.cell_name, cell_list)
    data.sub_by_index(cell_index=cell_subset, filter_raw=filter_raw)
    return data


def filter_genes(
        data: StereoExpData,
        min_cells=None,
        max_cells=None,
        min_counts=None,
        max_counts=None,
        gene_list=None,
        mean_umi_gt=None,
        excluded=False,
        filter_mt_genes=False,
        filter_raw=True,
        use_raw=True,
        layer=None,
        inplace=True
):
    """
    filter genes based on the numbers of cells.

    :param data: StereoExpData object.
    :param min_cells: Minimum number of cells for a gene pass filtering.
    :param max_cells: Maximun number of cells for a gene pass filtering.
    :param min_counts: minimum number of counts expressed required for a gene to pass filtering.
    :param max_counts: maximum number of counts expressed required for a gene to pass filtering.
    :param mean_umi_gt: Filter genes whose mean umi greater than this value.
    :param gene_list: the list of genes which will be filtered.
    :param excluded: set it to True to exclude the genes specified by parameter `gene_list` while False to include.
    :param filter_mt_genes: whether filter the mt genes.
    :param filter_raw: whether filter the raw data.
    :param inplace: whether inplace the original data or return a new data.

    :return: StereoExpData object.
    """
    data = data if inplace else copy.deepcopy(data)
    if not filter_mt_genes and \
        (min_cells is None and max_cells is None \
        and min_counts is None and max_counts is None \
        and gene_list is None and mean_umi_gt is None):
        raise ValueError('Please set any of `min_cells`, `max_cells`, `min_counts`, `max_counts`, `gene_list` and `mean_umi_gt`')
    if min_cells is not None or max_cells is not None \
        or min_counts is not None or max_counts is not None or mean_umi_gt is not None:
        cal_genes_indicators(data, use_raw, layer)
    gene_subset = np.ones(data.genes.size, dtype=bool)
    if min_cells:
        gene_subset &= data.genes.n_cells >= min_cells
    if max_cells:
        gene_subset &= data.genes.n_cells <= max_cells
    if min_counts:
        gene_subset &= data.genes.n_counts >= min_counts
    if max_counts:
        gene_subset &= data.genes.n_counts <= max_counts
    if gene_list is not None:
        if excluded:
            gene_subset &= ~np.isin(data.gene_names, gene_list)
        else:
            gene_subset &= np.isin(data.gene_names, gene_list)
    if mean_umi_gt is not None:
        gene_subset &= data.genes.mean_umi > mean_umi_gt
    if filter_mt_genes:
        gene_subset &= ~np.char.startswith(np.char.lower(data.gene_names), 'mt-')
    data.sub_by_index(gene_index=gene_subset, filter_raw=filter_raw)
    return data


def filter_coordinates(
        data: StereoExpData,
        min_x=None,
        max_x=None,
        min_y=None,
        max_y=None,
        filter_raw=True,
        inplace=True
):
    """
    filter cells based on the coordinates of cells.

    :param data: StereoExpData object.
    :param min_x: Minimum of x for a cell pass filtering.
    :param max_x: Maximum of x for a cell pass filtering.
    :param min_y: Minimum of y for a cell pass filtering.
    :param max_y: Maximum of y for a cell pass filtering.
    :param filter_raw: whether filter the raw data.
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
    data.sub_by_index(cell_index=obs_subset, filter_raw=filter_raw)
    return data


# def filter_by_clusters(
#         data: StereoExpData,
#         cluster_res: pd.DataFrame,
#         groups: Union[str, np.ndarray, List[str]],
#         excluded: bool = False,
#         inplace: bool = True
# ) -> Tuple[StereoExpData, pd.DataFrame]:
#     """_summary_

#     :param data: StereoExpData object.
#     :param cluster_res: clustering result.
#     :param groups: the groups in clustering result which will be filtered.
#     :param inplace: whether inplace the original data or return a new data.
#     :param excluded: bool type.
#     :return: StereoExpData object
#     """
#     data = data if inplace else copy.deepcopy(data)
#     all_groups = cluster_res['group']
#     if isinstance(groups, str):
#         groups = [groups]
#     is_in_bool = all_groups.isin(groups).to_numpy()
#     if excluded:
#         is_in_bool = ~is_in_bool
#     data.sub_by_index(cell_index=is_in_bool)
#     cluster_res = cluster_res[is_in_bool].copy()
#     cluster_res['group'] = cluster_res['group'].to_numpy()
#     cluster_res['group'] = cluster_res['group'].astype('category')
#     return data, cluster_res

def filter_by_clusters(
        data: StereoExpData,
        cluster_res_key: str,
        groups: Union[str, np.ndarray, List[str]],
        excluded: bool = False,
        filter_raw: bool = True,
        inplace: bool = False
) -> Tuple[StereoExpData, pd.DataFrame]:
    """_summary_

    :param data: StereoExpData object.
    :param cluster_res_key: the key of clustering result in data.cells.
    :param groups: the groups in clustering result which will be filtered.
    :param excluded: whether exclude the groups.
    :param filter_raw: whether filter the raw data.
    :param inplace: whether inplace the original data or return a new data.

    :return: StereoExpData object
    """
    data = data if inplace else copy.deepcopy(data)
    all_groups = data.cells[cluster_res_key]
    if isinstance(groups, str):
        groups = [groups]
    is_in_bool = all_groups.isin(groups).to_numpy()
    if excluded:
        is_in_bool = ~is_in_bool
    data.sub_by_index(cell_index=is_in_bool, filter_raw=filter_raw)

    return data