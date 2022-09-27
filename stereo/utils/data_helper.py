#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: data_helper.py
@time: 2021/3/14 16:11
"""
# from scipy.sparse import issparse
import scipy.sparse as sp
import pandas as pd
import numpy as np
from ..core.stereo_exp_data import StereoExpData
from typing import Optional
from datetime import datetime
from stereo.core.cell import Cell
from stereo.core.gene import Gene


def select_group(st_data, groups, cluster):
    all_groups = set(cluster['group'].values)
    groups = groups if isinstance(groups, list) else [groups]
    for g in groups:
        if g not in all_groups:
            raise ValueError(f"cluster {g} is not in all cluster.")
    group_index = cluster['group'].isin(groups)
    exp_matrix = st_data.exp_matrix.toarray() if sp.issparse(st_data.exp_matrix) else st_data.exp_matrix
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

def merge(data1: StereoExpData = None, data2: StereoExpData = None, *args, is_sparse=True, reorganize_coordinate=True, coordinate_offset_additional=0):
    """merge two or more datas to one

    :param data1: the first data to be merged, an object of StereoExpData, defaults to None
    :param data2: the second data to be merged, an object of StereoExpData, defaults to None
        you can also input more than two datas
    :param is_sparse: if True, merge to a sparse array, or merge to a ndarray, defaults to True
    :param reorganize_coordinate: set to true to reorganize the coordinates of cells, defaults to True
                if reorganize_coordinate is set to True, the coordinates of cells will be reorganized like below:
                        ---------------
                        | data1 data2 |
                        | data3 data4 |
                        | data5 ...   |
                        | ...   ...   |
                        ---------------
                if set to False, the coordinates maybe overlap between each data.
    :param coordinate_offset_additional: the offset between left and right or up and down after reorganizing the coordinates of cells
                for example, between data1 and data2, data1 and data3, data2 and data4...
                be ignored if reorganize_coordinate set to False
    :return: a new object of StereoExpData
    """
    assert data1 is not None, 'the first parameter `data1` must be input'
    if data2 is None:
        return data1
    datas = [data1, data2]
    if len(args) > 0:
        datas.extend(args)
    data_count = len(datas)
    new_data = StereoExpData()
    for i in range(data_count):
        data: StereoExpData = datas[i]
        data.cells.batch = i
        cell_names = np.array([f"{cell_name}-{i}" for cell_name in data.cells.cell_name])
        data.array2sparse()
        if i == 0:
            new_data.exp_matrix = data.exp_matrix.copy()
            new_data.cells = Cell(cell_name=cell_names, cell_border=data.cells.cell_boder, batch=data.cells.batch)
            new_data.genes = Gene(gene_name=data.gene_names)
            new_data.position = data.position
            new_data.bin_type = data.bin_type
            new_data.bin_size = data.bin_size
            new_data.offset_x = data.offset_x
            new_data.offset_y = data.offset_y
            new_data.attr = data.attr
        else:
            new_data.cells.cell_name = np.concatenate([new_data.cells.cell_name, cell_names])
            new_data.cells.batch = np.concatenate([new_data.cells.batch, data.cells.batch])
            if new_data.cell_borders is not None and data.cell_borders is not None:
                new_data.cells.cell_boder = np.concatenate([new_data.cells.cell_boder, data.cells.cell_boder])
            new_data.position = np.concatenate([new_data.position, data.position])
            new_data.genes.gene_name, ind1, ind2 = np.intersect1d(new_data.genes.gene_name, data.genes.gene_name, return_indices=True)
            # new_data.exp_matrix = np.concatenate([new_data.exp_matrix[:, ind1], data.exp_matrix[:, ind2]])
            new_data.exp_matrix = sp.vstack([new_data.exp_matrix[:, ind1], data.exp_matrix[:, ind2]])
            if new_data.offset_x is not None and data.offset_x is not None:
                new_data.offset_x = min(new_data.offset_x, data.offset_x)
            if new_data.offset_y is not None and data.offset_y is not None:
                new_data.offset_y = min(new_data.offset_y, data.offset_y)
            if new_data.attr is not None and data.attr is not None:
                new_data.attr = {
                    'minX': min(new_data.attr['minX'], data.attr['minX']),
                    'minY': min(new_data.attr['minY'], data.attr['minY']),
                    'maxX': max(new_data.attr['maxX'], data.attr['maxX']),
                    'maxY': max(new_data.attr['maxY'], data.attr['maxY']),
                    'minExp': new_data.exp_matrix.min(),
                    'maxExp': new_data.exp_matrix.min(),
                    'resolution': 0,
                }
    if reorganize_coordinate:
        coordinate_offset_additional = 0 if coordinate_offset_additional < 0 else coordinate_offset_additional
        batches = np.unique(new_data.cells.batch)
        max_x_list = []
        max_y_list = [0]
        for i in range(0, data_count, 2):
            start = i
            end = min(i + 2, data_count)
            current_row_batches = batches[start:end]
            if len(current_row_batches) == 2:
                idx1 = np.where(new_data.cells.batch == current_row_batches[0])
                idx2 = np.where(new_data.cells.batch == current_row_batches[1])
            else:
                idx1 = np.where(new_data.cells.batch == current_row_batches[0])
                idx2 = None
            current_row_position_1 = new_data.position[idx1]
            current_row_position_2 = new_data.position[idx2] if idx2 is not None else None
            max_x = current_row_position_1[:, 0].max()
            if current_row_position_2 is not None:
                max_y = max(current_row_position_1[:, 1].max(), current_row_position_2[:, 1].max()) + max_y_list[-1]
            else:
                max_y = current_row_position_1[:, 1].max() + max_y_list[-1]
            max_x_list.append(max_x)
            max_y_list.append(max_y)
        max_x = max(max_x_list)
        for i, bno in enumerate(batches):
            idx = np.where(new_data.cells.batch == bno)
            if (i % 2) == 0:
                x_add = 0
            else:
                x_add = max_x + coordinate_offset_additional
            
            y_add = max_y_list[i // 2]
            if y_add > 0:
                y_add += coordinate_offset_additional
            new_data.position[idx] += [x_add, y_add]

    if not is_sparse:
        new_data.sparse2array()
    return new_data