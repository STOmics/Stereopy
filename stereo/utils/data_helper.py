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


def merge(data1: StereoExpData = None, data2: StereoExpData = None, *args, reorganize_coordinate=2,
          coordinate_offset_additional=0):
    """merge two or more datas to one

    :param data1: the first data to be merged, an object of StereoExpData, defaults to None
    :param data2: the second data to be merged, an object of StereoExpData, defaults to None
        you can also input more than two datas
    :param reorganize_coordinate: set it to decide to whether reorganize the coordinates of the cells
                if set it to False, will not reorganize 
                if set it to a number, like 2, the coordinates of cells will be reorganized to 2 columns like below:
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
    new_data = StereoExpData(merged=True)
    new_data.sn = {}
    if reorganize_coordinate:
        from math import ceil
        position_row_count = ceil(data_count / reorganize_coordinate)
        position_column_count = reorganize_coordinate
        max_xs = [0] * (position_column_count + 1)
        max_ys = [0] * (position_row_count + 1)
    for i in range(data_count):
        data: StereoExpData = datas[i]
        data.cells.batch = i
        cell_names = np.array([f"{cell_name}-{i}" for cell_name in data.cells.cell_name])
        data.array2sparse()
        new_data.sn[str(i)] = data.sn
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
            new_data.genes.gene_name, ind1, ind2 = np.intersect1d(new_data.genes.gene_name, data.genes.gene_name,
                                                                  return_indices=True)
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
            position_row_number = i // reorganize_coordinate
            position_column_number = i % reorganize_coordinate
            max_x = data.position[:, 0].max()
            max_y = data.position[:, 1].max()
            if max_x > max_xs[position_column_number + 1]:
                max_xs[position_column_number + 1] = max_x
            if max_y > max_ys[position_row_number + 1]:
                max_ys[position_row_number + 1] = max_y
    if reorganize_coordinate:
        coordinate_offset_additional = 0 if coordinate_offset_additional < 0 else coordinate_offset_additional
        batches = np.unique(new_data.cells.batch).tolist()
        batches.sort(key=lambda x: int(x))
        for i, bno in enumerate(batches):
            idx = np.where(new_data.cells.batch == bno)[0]
            position_row_number = i // reorganize_coordinate
            position_column_number = i % reorganize_coordinate
            x_add = max_xs[position_column_number]
            y_add = max_ys[position_row_number]
            if position_column_number > 0:
                x_add += sum(max_xs[0:position_column_number]) + coordinate_offset_additional * position_column_number
            if position_row_number > 0:
                y_add += sum(max_ys[0:position_row_number]) + coordinate_offset_additional * position_row_number
            # position_offset = np.repeat([[x_add, y_add]], repeats=len(idx), axis=0).astype(np.uint32)
            position_offset = np.array([x_add, y_add], dtype=np.uint32)
            new_data.position[idx] += position_offset
            if new_data.position_offset is None:
                new_data.position_offset = {bno: position_offset}
            else:
                # new_data.position_offset = np.concatenate([new_data.position_offset, position_offset])
                new_data.position_offset[bno] = position_offset

    return new_data


def split(data: StereoExpData = None):
    """
    splitting a data which is merged from different batches base on batch number

    :param data: a merged data, defaults to None
    :return: a split data list
    """

    if data is None:
        return None

    from copy import deepcopy

    all_data = []
    data.array2sparse()
    batch = np.unique(data.cells.batch)
    result = data.tl.result
    for bno in batch:
        cell_idx = np.where(data.cells.batch == bno)[0]
        cell_names = data.cell_names[cell_idx]
        new_data = StereoExpData(bin_type=data.bin_type, bin_size=data.bin_size, cells=deepcopy(data.cells),
                                 genes=deepcopy(data.genes))
        new_data.cells = new_data.cells.sub_set(cell_idx)
        new_data.position = data.position[cell_idx] - data.position_offset[bno]
        new_data.exp_matrix = data.exp_matrix[cell_idx]
        new_data.tl.key_record = deepcopy(data.tl.key_record)
        new_data.sn = data.sn[bno]
        for key, all_res_key in data.tl.key_record.items():
            if len(all_res_key) == 0:
                continue
            if key == 'hvg':
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
            elif key in ['pca', 'cluster', 'umap']:
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key].iloc[cell_idx]
                    new_data.tl.result[res_key].reset_index(drop=True, inplace=True)
            elif key == 'neighbors':
                min_idx = cell_idx.min()
                max_idx = cell_idx.max() + 1
                for res_key in all_res_key:
                    connectivities = result[res_key]['connectivities']
                    nn_dist = result[res_key]['nn_dist']
                    new_data.tl.result[res_key] = {
                        'neighbor': result[res_key]['neighbor'],
                        'connectivities': connectivities[min_idx:max_idx, min_idx:max_idx],
                        'nn_dist': nn_dist[min_idx:max_idx, min_idx:max_idx]
                    }
            elif key == 'marker_genes':
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
            elif key == 'sct':
                for res_key in all_res_key:
                    cells_bool_list = np.isin(result[res_key][0]['umi_cells'], cell_names)
                    # sct `counts` and `data` should have same shape
                    new_data.tl.result[res_key] = (
                        new_data,
                        {
                            'cells': cell_names,
                            'genes': result[res_key][0]['umi_genes'],
                            'filtered_corrected_counts': result[res_key][0]['counts'][cells_bool_list, :],
                            'filtered_normalized_counts': result[res_key][0]['data'][cells_bool_list, :]
                        }
                    )
            elif key == 'tsne':
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
            else:
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
        if data.tl.raw is not None:
            new_data.tl.raw = data.tl.raw.tl.filter_cells(cell_list=cell_names, inplace=False)
        all_data.append(new_data)

    return all_data
