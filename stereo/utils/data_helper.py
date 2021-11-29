#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: data_helper.py
@time: 2021/3/14 16:11
"""
from scipy.sparse import issparse
import pandas as pd
import numpy as np
from ..core.stereo_exp_data import StereoExpData
from typing import Optional
from datetime import datetime


def select_group(st_data, groups, cluster):
    all_groups = set(cluster['group'].values)
    groups = groups if isinstance(groups, list) else [groups]
    for g in groups:
        if g not in all_groups:
            raise ValueError(f"cluster {g} is not in all cluster.")
    group_index = cluster['group'].isin(groups)
    exp_matrix = st_data.exp_matrix.toarray() if issparse(st_data.exp_matrix) else st_data.exp_matrix
    group_sub = exp_matrix[group_index]
    return group_sub, group_index


def get_cluster_res(adata, data_key='clustering'):
    cluster_data = adata.uns[data_key].cluster
    cluster = cluster_data['cluster'].astype(str).astype('category').values
    return cluster


def get_position_array(data, obs_key='spatial'):
    return np.array(data.obsm[obs_key])[:, 0: 2]


def exp_matrix2df(data: StereoExpData, cell_name: Optional[np.ndarray] = None, gene_name: Optional[np.ndarray] = None):
    if data.tl.raw:
        data = data.tl.raw
    cell_index = [np.argwhere(data.cells.cell_name == i)[0][0] for i in cell_name] if cell_name is not None else None
    gene_index = [np.argwhere(data.genes.gene_name == i)[0][0] for i in gene_name] if gene_name is not None else None
    x = data.exp_matrix[cell_index, :] if cell_index is not None else data.exp_matrix
    x = x[:, gene_index] if gene_index is not None else x
    x = x if isinstance(x, np.ndarray) else x.toarray()
    index = cell_name if cell_name is not None else data.cell_names
    columns = gene_name if gene_name is not None else data.gene_names
    df = pd.DataFrame(data=x, index=index, columns=columns)
    return df


def get_top_marker(g_name: str, marker_res: dict, sort_key: str, ascend: bool = False, top_n: int = 10):
    result = marker_res[g_name]
    top_res = result.sort_values(by=sort_key, ascending=ascend).head(top_n)
    return top_res
