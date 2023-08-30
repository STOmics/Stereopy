#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: data_helper.py
@time: 2021/3/14 16:11
"""
from typing import Optional
from typing import Union
from natsort import natsorted
from math import ceil

import numpy as np
import pandas as pd
import scipy.sparse as sp

from stereo.core.cell import Cell
from stereo.core.gene import Gene
from ..core.stereo_exp_data import StereoExpData


def select_group(groups, cluster, all_groups):
    groups = [groups] if isinstance(groups, str) else groups
    for g in groups:
        if g not in all_groups:
            raise ValueError(f"cluster {g} is not in all cluster.")
    group_index = cluster['group'].isin(groups)
    return group_index


def get_cluster_res(adata, data_key='clustering'):
    cluster_data = adata.uns[data_key].cluster
    cluster = cluster_data['cluster'].astype(str).astype('category').values
    return cluster


def get_position_array(data, obs_key='spatial'):
    return np.array(data.obsm[obs_key])[:, 0: 2]


def exp_matrix2df(data: StereoExpData, cell_name: Optional[np.ndarray] = None, gene_name: Optional[np.ndarray] = None):
    if data.tl.raw:
        cell_isin = np.isin(data.tl.raw.cell_names, data.cell_names)
        gene_isin = np.isin(data.tl.raw.gene_names, data.gene_names)
        exp_matrix = data.tl.raw.exp_matrix[cell_isin, :][:, gene_isin]
        # data = data.tl.raw
    else:
        exp_matrix = data.exp_matrix
    cell_index = [np.argwhere(data.cells.cell_name == i)[0][0] for i in cell_name] if cell_name is not None else None
    gene_index = [np.argwhere(data.genes.gene_name == i)[0][0] for i in gene_name] if gene_name is not None else None
    # x = data.exp_matrix[cell_index, :] if cell_index is not None else data.exp_matrix
    x = exp_matrix[cell_index, :] if cell_index is not None else exp_matrix
    x = x[:, gene_index] if gene_index is not None else x
    x = x if isinstance(x, np.ndarray) else x.toarray()
    index = cell_name if cell_name is not None else data.cell_names
    columns = gene_name if gene_name is not None else data.gene_names
    df = pd.DataFrame(data=x, index=index, columns=columns)
    return df


def get_top_marker(g_name: str, marker_res: dict, sort_key: str, ascend: bool = False, top_n: int = 10):
    result: pd.DataFrame = marker_res[g_name]
    # top_res = result.sort_values(by=sort_key, ascending=ascend).head(top_n).dropna(axis=0, how='any')
    top_res = result.sort_values(by=sort_key, ascending=ascend).head(top_n).dropna(axis=0, subset=[sort_key])
    return top_res


def _union_merge(arr1: np.ndarray, arr2: np.ndarray, new_col: np.ndarray, old_col1: np.ndarray, old_col2: np.ndarray):
    """
    Merge two array:
               a  b        b  c
            0  1  3     0  1  3
            1  2  4     1  2  4

    To:  a  b  c
       [[1. 3. 0.]
       [2. 4. 0.]
       [0. 1. 3.]
       [0. 2. 4.]]
    """
    merged_arr = np.zeros([arr1.shape[0] + arr2.shape[0], len(new_col)])
    merged_arr[:arr1.shape[0], np.where(old_col1 == new_col[:, None])[0]] = arr1
    merged_arr[arr1.shape[0]:arr1.shape[0] + arr2.shape[0], np.where(old_col2 == new_col[:, None])[0]] = arr2
    return merged_arr


def reorganize_data_coordinates(
        cells_batch: np.ndarray,
        data_position: np.ndarray,
        data_position_offset: dict = None,
        reorganize_coordinate: Union[bool, int] = 2,
        horizontal_offset_additional: Union[int, float] = 0,
        vertical_offset_additional: Union[int, float] = 0,
):
    if not reorganize_coordinate:
        return data_position, data_position_offset

    batches = natsorted(np.unique(cells_batch))
    data_count = len(batches)
    position_row_count = ceil(data_count / reorganize_coordinate)
    position_column_count = reorganize_coordinate
    max_xs = [0] * (position_column_count + 1)
    max_ys = [0] * (position_row_count + 1)
    for i, bno in enumerate(batches):
        idx = np.where(cells_batch == bno)[0]
        data_position[idx] -= data_position_offset[bno] if data_position_offset is not None else 0
        position_row_number = i // reorganize_coordinate
        position_column_number = i % reorganize_coordinate
        max_x = data_position[idx][:, 0].max() - data_position[idx][:, 0].min() + 1
        max_y = data_position[idx][:, 1].max() - data_position[idx][:, 1].min() + 1
        if max_x > max_xs[position_column_number + 1]:
            max_xs[position_column_number + 1] = max_x
        if max_y > max_ys[position_row_number + 1]:
            max_ys[position_row_number + 1] = max_y

    data_position_offset = {}
    for i, bno in enumerate(batches):
        idx = np.where(cells_batch == bno)[0]
        position_row_number = i // reorganize_coordinate
        position_column_number = i % reorganize_coordinate
        x_add = max_xs[position_column_number]
        y_add = max_ys[position_row_number]
        if position_column_number > 0:
            x_add += sum(max_xs[0:position_column_number]) + horizontal_offset_additional * position_column_number
        if position_row_number > 0:
            y_add += sum(max_ys[0:position_row_number]) + vertical_offset_additional * position_row_number
        # position_offset = np.repeat([[x_add, y_add]], repeats=len(idx), axis=0).astype(np.uint32)
        position_offset = np.array([x_add, y_add], dtype=data_position.dtype)
        data_position[idx] += position_offset
        data_position_offset[bno] = position_offset
    return data_position, data_position_offset


def merge(
        *data_list: StereoExpData,
        reorganize_coordinate: Union[bool, int] = 2,
        horizontal_offset_additional: Union[int, float] = 0,
        vertical_offset_additional: Union[int, float] = 0,
        space_between: Optional[str] = '0',
        var_type="intersect",
):
    """
    Merge several slices of data.

    :param data_list: several slices of data to be merged, at least two slices. 
    :param reorganize_coordinate: whether to reorganize the coordinates of the obs(cells), 
            if set it to a number, like 2, the coordinates will be reorganized to 2 columns on coordinate system as below:
                            ---------------
                            | data1 data2
                            | data3 data4
                            | data5 ...  
                            | ...   ...  
                            ---------------
            if set to `False`, the coordinates maybe overlap between slices.
    :param horizontal_offset_additional: the additional offset between each slice on horizontal direction while reorganizing coordinates.
    :param vertical_offset_additional: the additional offset between each slice on vertical direction while reorganizing coordinates.
    :param space_between: the distance between each slice, like '10nm', '1um', ..., it will be used for calculating the z-coordinate of each slice.

    :return: A merged StereoExpData object.
    """
    if data_list is None or len(data_list) < 2:
        raise Exception("At least two slices of data need to be input.")

    def _parse_space_between(space_between: str):
        import re
        if space_between == '0':
            return 0.0
        pattern = r"^\d+(\.\d)*(nm|um|mm|cm|dm|m)$"
        match = re.match(pattern, space_between)
        if match is None:
            raise ValueError(f"Invalid space between: '{space_between}'")
        unit = match.groups()[1]
        space_between = float(space_between.replace(unit, ''))
        if unit == 'um':
            space_between *= 1e3
        elif unit == 'mm':
            space_between *= 1e6
        elif unit == 'cm':
            space_between *= 1e7
        elif unit == 'dm':
            space_between *= 1e8
        elif unit == 'm':
            space_between *= 1e9
        return space_between

    space_between = _parse_space_between(space_between)
    data_count = len(data_list)
    new_data = StereoExpData(merged=True)
    new_data.sn = {}
    current_position_z = 0
    for i in range(data_count):
        data: StereoExpData = data_list[i]
        data.cells.batch = i
        # cell_names = np.array([f"{cell_name}-{i}" for cell_name in data.cells.cell_name])
        cell_names = np.char.add(data.cells.cell_name, f"-{i}")
        data.array2sparse()
        new_data.sn[str(i)] = data.sn
        if i == 0:
            new_data.exp_matrix = data.exp_matrix.copy()
            new_data.cells = Cell(cell_name=cell_names, cell_border=data.cells.cell_border, batch=data.cells.batch)
            new_data.genes = Gene(gene_name=data.gene_names)
            new_data.cells._obs = data.cells._obs.copy(deep=True)
            new_data.cells._obs.index = cell_names
            new_data.position = data.position
            if data.position_z is None:
                new_data.position_z = np.repeat([[0]], repeats=data.position.shape[0], axis=0).astype(
                    data.position.dtype)
            else:
                new_data.position_z = data.position_z
            new_data.bin_type = data.bin_type
            new_data.bin_size = data.bin_size
            new_data.offset_x = data.offset_x
            new_data.offset_y = data.offset_y
            new_data.attr = data.attr
        else:
            current_obs = data.cells._obs.copy()
            current_obs.index = cell_names
            new_data.cells._obs = pd.concat([new_data.cells._obs, current_obs])
            if new_data.cell_borders is not None and data.cell_borders is not None:
                new_data.cells.cell_border = np.concatenate([new_data.cells.cell_border, data.cells.cell_border])
            new_data.position = np.concatenate([new_data.position, data.position])
            if data.position_z is None:
                current_position_z += space_between / data.attr['resolution']
                new_data.position_z = np.concatenate(
                    [new_data.position_z, np.repeat([[current_position_z]], repeats=data.position.shape[0], axis=0)])
            else:
                new_data.position_z = np.concatenate([new_data.position_z, data.position_z])
            if var_type == "intersect":
                new_data.genes.gene_name, ind1, ind2 = \
                    np.intersect1d(new_data.genes.gene_name, data.genes.gene_name, return_indices=True)
                new_data.exp_matrix = sp.vstack([new_data.exp_matrix[:, ind1], data.exp_matrix[:, ind2]])
            elif var_type == "union":
                old_gene_name = new_data.genes.gene_name
                new_data.genes.gene_name = np.union1d(new_data.genes.gene_name, data.genes.gene_name)
                new_data.exp_matrix = _union_merge(
                    new_data.exp_matrix.toarray(), data.exp_matrix.toarray(), new_data.genes.gene_name, old_gene_name,
                    data.genes.gene_name
                )
            else:
                raise Exception(f"got an unexpected var_type: {var_type}")
            if new_data.offset_x is not None and data.offset_x is not None:
                new_data.offset_x = min(new_data.offset_x, data.offset_x)
            if new_data.offset_y is not None and data.offset_y is not None:
                new_data.offset_y = min(new_data.offset_y, data.offset_y)
            if new_data.attr is not None and data.attr is not None:
                for key, value in data.attr.items():
                    if key in ('minX', 'minY'):
                        new_data.attr[key] = min(new_data.attr[key], value)
                    elif key in ('maxX', 'maxY'):
                        new_data.attr[key] = max(new_data.attr[key], value)
                    elif key == 'minExp':
                        new_data.attr['minExp'] = new_data.exp_matrix.min()
                    elif key == 'maxExp':
                        new_data.attr['maxExp'] = new_data.exp_matrix.max()
                    elif key == 'resolution':
                        new_data.attr['resolution'] = value
    if reorganize_coordinate:
        new_data.position, new_data.position_offset = reorganize_data_coordinates(
            new_data.cells.batch, new_data.position, new_data.position_offset,
            reorganize_coordinate, horizontal_offset_additional, vertical_offset_additional
        )

    return new_data


def split(data: StereoExpData = None):
    """
    Split a data object which is merged from different batches of data, according to the batch number.

    :param data: a merged data object.

    :return: A split data list.
    """

    if data is None:
        return None

    from copy import deepcopy
    from .pipeline_utils import cell_cluster_to_gene_exp_cluster

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
        if data.position_offset is not None:
            new_data.position = data.position[cell_idx] - data.position_offset[bno]
        else:
            new_data.position = data.position[cell_idx]
        new_data.position_z = data.position_z[cell_idx]
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
            elif key == 'gene_exp_cluster':
                continue
            else:
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
        if data.tl.raw is not None:
            new_data.tl.raw = data.tl.raw.tl.filter_cells(cell_list=cell_names, inplace=False)
        if 'gene_exp_cluster' in data.tl.key_record:
            for cluster_res_key in data.tl.key_record['cluster']:
                gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(new_data, cluster_res_key)
                if gene_exp_cluster_res is not False:
                    new_data.tl.result[f"gene_exp_{cluster_res_key}"] = gene_exp_cluster_res
        all_data.append(new_data)

    return all_data
