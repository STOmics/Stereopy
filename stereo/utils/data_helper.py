#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: data_helper.py
@time: 2021/3/14 16:11
"""
from math import ceil
from typing import Optional
from typing import Union
from functools import singledispatch
from copy import deepcopy

import anndata as ad
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from natsort import natsorted

from stereo.core.cell import Cell
from stereo.core.gene import Gene
from stereo.core.stereo_exp_data import StereoExpData, AnnBasedStereoExpData
from stereo.log_manager import logger


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


# def exp_matrix2df(data: StereoExpData, cell_name: Optional[np.ndarray] = None, gene_name: Optional[np.ndarray] = None):
#     # if data.tl.raw:
#     #     cell_isin = np.isin(data.tl.raw.cell_names, data.cell_names)
#     #     gene_isin = np.isin(data.tl.raw.gene_names, data.gene_names)
#     #     exp_matrix = data.tl.raw.exp_matrix[cell_isin, :][:, gene_isin]
#     # else:
#     #     exp_matrix = data.exp_matrix
#     cell_index = [np.argwhere(data.cells.cell_name == i)[0][0] for i in cell_name] if cell_name is not None else None
#     gene_index = [np.argwhere(data.genes.gene_name == i)[0][0] for i in gene_name] if gene_name is not None else None
#     # x = exp_matrix[cell_index, :] if cell_index is not None else exp_matrix
#     x = data.exp_matrix[cell_index, :] if cell_index is not None else data.exp_matrix
#     x = x[:, gene_index] if gene_index is not None else x
#     x = x if isinstance(x, np.ndarray) else x.toarray()
#     index = cell_name if cell_name is not None else data.cell_names
#     columns = gene_name if gene_name is not None else data.gene_names
#     df = pd.DataFrame(data=x, index=index, columns=columns)
    return df

def exp_matrix2df(
    data: StereoExpData,
    use_raw=False,
    layer: Optional[str] = None,
    cell_name: Optional[np.ndarray] = None,
    gene_name: Optional[np.ndarray] = None
):
    x = data.get_exp_matrix(use_raw=use_raw, layer=layer, cell_list=cell_name, gene_list=gene_name)
    x = x if isinstance(x, np.ndarray) else x.toarray()
    index = cell_name if cell_name is not None else data.cell_names
    columns = gene_name if gene_name is not None else data.gene_names
    df = pd.DataFrame(data=x, index=index, columns=columns)
    return df


def get_top_marker(g_name: str, marker_res: dict, sort_key: str, ascend: bool = False, top_n: int = 10):
    result: pd.DataFrame = marker_res[g_name]
    top_res = result.sort_values(by=sort_key, ascending=ascend).head(top_n).dropna(axis=0, subset=[sort_key])
    return top_res

@singledispatch
def _union_merge(arr1, arr2, col1, col2):
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
    pass

@_union_merge.register(np.ndarray)
def _union_merge_array(arr1: np.ndarray, arr2: np.ndarray, col1: np.ndarray, col2: np.ndarray):
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
    if (col1.size == col2.size) and np.all(col1 == col2):
        return col1, np.concatenate([arr1, arr2])

    new_col = np.union1d(col1, col2)
    new_col_index = pd.Index(new_col)
    ind1_in_new_col = new_col_index.get_indexer(col1)
    ind2_in_new_col = new_col_index.get_indexer(col2)
    merged_arr = np.zeros([arr1.shape[0] + arr2.shape[0], new_col.size], dtype=arr1.dtype)
    merged_arr[0:arr1.shape[0], ind1_in_new_col] = arr1
    merged_arr[arr1.shape[0]:(arr1.shape[0] + arr2.shape[0]), ind2_in_new_col] = arr2
    return new_col, merged_arr


@_union_merge.register(sp.csr_matrix)
def _union_merge_csr_matrix(mtx1: sp.csr_matrix, mtx2: sp.csr_matrix, col1: np.ndarray, col2: np.ndarray):
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
    if (col1.size == col2.size) and np.all(col1 == col2):
        return col1, sp.vstack([mtx1, mtx2])

    @nb.njit(cache=True)
    def __merge(
        new_col_size: int, 
        mtx1_shape: tuple,
        mtx1_indptr: np.ndarray,
        mtx1_indices: np.ndarray,
        mtx1_data: np.ndarray,
        ind1_in_new_col: np.ndarray,
        mtx2_shape: np.ndarray,
        mtx2_indptr: np.ndarray,
        mtx2_indices: np.ndarray,
        mtx2_data: np.ndarray,
        ind2_in_new_col: np.ndarray
    ):
        row_count = mtx1_shape[0] + mtx2_shape[0]
        
        data = np.zeros(mtx1_data.size + mtx2_data.size, dtype=mtx1_data.dtype)
        indices = np.zeros(mtx1_indices.size + mtx2_indices.size, dtype=mtx1_indices.dtype)
        indptr = np.zeros(row_count + 1, dtype=mtx1_indptr.dtype)
        
        row_new = np.zeros(new_col_size, dtype=data.dtype)
        
        for i in range(row_count):
            if i < mtx1_shape[0]:
                col_ind_start, col_ind_end = mtx1_indptr[i], mtx1_indptr[i + 1]
                col_ind = mtx1_indices[col_ind_start:col_ind_end]
                row_old = np.zeros(mtx1_shape[1], dtype=data.dtype)
                row_old[col_ind] = mtx1_data[col_ind_start:col_ind_end]
                row_new[ind1_in_new_col] = row_old
            else:
                j = i - mtx1_shape[0]
                col_ind_start, col_ind_end = mtx2_indptr[j], mtx2_indptr[j + 1]
                col_ind = mtx2_indices[col_ind_start:col_ind_end]
                row_old = np.zeros(mtx2_shape[1], dtype=data.dtype)
                row_old[col_ind] = mtx2_data[col_ind_start:col_ind_end]
                row_new[ind2_in_new_col] = row_old
            nonzero_ind = np.nonzero(row_new)[0]
            indptr[i + 1] = indptr[i] + nonzero_ind.size
            data[indptr[i]:indptr[i + 1]] = row_new[nonzero_ind]
            indices[indptr[i]:indptr[i + 1]] = nonzero_ind
            row_new[:] = 0
        return data, indices, indptr
    
    new_col = np.union1d(col1, col2)
    
    new_col_index = pd.Index(new_col)
    ind1_in_new_col = new_col_index.get_indexer(col1)
    ind2_in_new_col = new_col_index.get_indexer(col2)

    data, indices, indptr = __merge(
        new_col.size,
        mtx1.shape, mtx1.indptr, mtx1.indices, mtx1.data, ind1_in_new_col,
        mtx2.shape, mtx2.indptr, mtx2.indices, mtx2.data, ind2_in_new_col
    )

    return new_col, sp.csr_matrix((data, indices, indptr), shape=(mtx1.shape[0] + mtx2.shape[0], new_col.size))

def _merge_matrix(data1: StereoExpData, data2: StereoExpData, var_type: str):
    layer_keys = list(data1.layers.keys())
    if var_type == "intersect":
        data1.genes.gene_name, ind1, ind2 = \
            np.intersect1d(data1.genes.gene_name, data2.genes.gene_name, return_indices=True)
        if data1.issparse():
            data1.exp_matrix = sp.vstack([data1.exp_matrix[:, ind1], data2.exp_matrix[:, ind2]])
        else:
            data1.exp_matrix = np.concatenate([data1.exp_matrix[:, ind1], data2.exp_matrix[:, ind2]])
        for key in data1.genes_matrix.keys():
            if isinstance(data1.genes_matrix[key], pd.DataFrame):
                data1.genes_matrix[key] = deepcopy(data1.genes_matrix[key].iloc[ind1])
            else:
                data1.genes_matrix[key] = deepcopy(data1.genes_matrix[key][ind1])
        for key in data2.genes_matrix.keys():
            if key in data1.genes_matrix:
                continue
            if isinstance(data2.genes_matrix[key], pd.DataFrame):
                data1.genes_matrix[key] = deepcopy(data2.genes_matrix[key].iloc[ind2])
            else:
                data1.genes_matrix[key] = deepcopy(data2.genes_matrix[key][ind2])
        for key in layer_keys:
            if key not in data2.layers:
                del data1.layers[key]
                continue
            if type(data1.layers[key]) != type(data2.layers[key]):
                del data1.layers[key]
                continue
            if isinstance(data1.layers[key], np.ndarray):
                data1.layers[key] = np.concatenate([data1.layers[key][:, ind1], data2.layers[key][:, ind2]])
            elif isinstance(data1.layers[key], sp.csr_matrix):
                data1.layers[key] = sp.vstack([data1.layers[key][:, ind1], data2.layers[key][:, ind2]])

    elif var_type == "union":
        original_var_index_1 = data1.genes.var.index
        original_var_index_2 = data2.genes.var.index
        data1.genes.gene_name, data1.exp_matrix = _union_merge(
            data1.exp_matrix, data2.exp_matrix, 
            data1.genes.gene_name, data2.genes.gene_name
        )
        for key in data1.genes_matrix.keys():
            if isinstance(data1.genes_matrix[key], np.ndarray):
                tmep_matrix = np.empty((data1.n_genes, data1.genes_matrix[key].shape[1]), dtype=data1.genes_matrix[key].dtype)
                tmep_matrix[:] = np.nan
                indexer = data1.genes.var.index.get_indexer(original_var_index_1)
                tmep_matrix[indexer] = data1.genes_matrix[key]
                data1.genes_matrix[key] = tmep_matrix
            elif isinstance(data1.genes_matrix[key], pd.DataFrame):
                data1.genes_matrix[key] = data1.genes_matrix[key].reindex(data1.genes.gene_name)
            elif isinstance(data1.genes_matrix[key], sp.csr_matrix):
                matrix: sp.csr_matrix = data1.genes_matrix[key]
                indexer = original_var_index_1.get_indexer(data1.genes.gene_name)
                new_indptr = np.zeros(data1.n_genes + 1, dtype=matrix.indptr.dtype)
                new_indices = np.zeros(matrix.indices.size, dtype=matrix.indices.dtype)
                new_data = np.zeros(matrix.data.size, dtype=matrix.data.dtype)
                for i, j in enumerate(indexer):
                    if j == -1:
                        new_indptr[i + 1] = new_indptr[i]
                        continue
                    new_indptr[i + 1] = new_indptr[i] + matrix.indptr[j + 1] - matrix.indptr[j]
                    new_indices[new_indptr[i]:new_indptr[i + 1]] = matrix.indices[matrix.indptr[j]:matrix.indptr[j + 1]]
                    new_data[new_indptr[i]:new_indptr[i + 1]] = matrix.data[matrix.indptr[j]:matrix.indptr[j + 1]]
                data1.genes_matrix[key] = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(data1.n_genes, matrix.shape[1]))
        for key in layer_keys:
            if key not in data2.layers:
                del data1.layers[key]
                continue
            if type(data1.layers[key]) != type(data2.layers[key]):
                del data1.layers[key]
                continue
            _, data1.layers[key] = _union_merge(
                data1.layers[key], data2.layers[key], 
                original_var_index_1, original_var_index_2
            )
    else:
        raise Exception(f"got an unexpected var_type: {var_type}")
    return data1

def reorganize_data_coordinates(
        cells_batch: np.ndarray,
        data_position: np.ndarray,
        data_position_offset: dict = None,
        data_position_min: dict = None,
        reorganize_coordinate: Union[bool, int] = 2,
        horizontal_offset_additional: Union[int, float] = 0,
        vertical_offset_additional: Union[int, float] = 0
):
    if not reorganize_coordinate:
        return data_position, data_position_offset, data_position_min

    batches = natsorted(np.unique(cells_batch))
    data_count = len(batches)
    position_row_count = ceil(data_count / reorganize_coordinate)
    position_column_count = reorganize_coordinate
    max_xs = [0] * (position_column_count + 1)
    max_ys = [0] * (position_row_count + 1)

    if data_position_min is None:
        data_position_min = {}
        for i, bno in enumerate(batches):
            idx = np.where(cells_batch == bno)[0]
            position_min = np.min(data_position[idx], axis=0)
            data_position[idx] -= position_min
            data_position_min[bno] = position_min

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
        position_offset = np.array([x_add, y_add], dtype=data_position.dtype)
        data_position[idx] += position_offset
        data_position_offset[bno] = position_offset
    return data_position, data_position_offset, data_position_min

def __parse_space_between(space_between: str):
    import re
    if isinstance(space_between, (int, float, np.number)):
        return float(space_between)
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

@singledispatch
def merge(
    *data_list: Union[StereoExpData, AnnBasedStereoExpData],
    reorganize_coordinate: Union[bool, int] = False,
    horizontal_offset_additional: Union[int, float] = 0,
    vertical_offset_additional: Union[int, float] = 0,
    space_between: Optional[str] = '0',
    var_type: str = "intersect",
    batch_tags: Union[list, np.ndarray, pd.Series] = None
) -> Union[StereoExpData, AnnBasedStereoExpData]:
    """
    Merge several slices of data.

    :param data_list: several slices of data to be merged, at least two slices.
    :param reorganize_coordinate: whether to reorganize the coordinates of the obs(cells), 
            if set it to a number, like 2, the coordinates will be reorganized to 2 columns on coordinate system as below
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
    :param var_type: how to merge the var(genes), 'intersect' or 'union', default 'intersect'.

    :return: A merged StereoExpData object.
    """  # noqa
    pass

@merge.register(StereoExpData)
def __merge_for_stereo_exp_data(
    *data_list: StereoExpData,
    reorganize_coordinate: Union[bool, int] = False,
    horizontal_offset_additional: Union[int, float] = 0,
    vertical_offset_additional: Union[int, float] = 0,
    space_between: Optional[str] = '0',
    var_type: str = "intersect",
    batch_tags: Union[list, np.ndarray, pd.Series] = None
):
    if data_list is None or len(data_list) < 2:
        raise Exception("At least two slices of data need to be input.")
    
    all_issparse = np.array([data.issparse() for data in data_list])

    if not np.all(all_issparse) and not np.all(~all_issparse):
        raise Exception("All slices of data should be in the same format, either sparse or ndarray.")

    space_between = __parse_space_between(space_between)
    data_count = len(data_list)
    new_data = StereoExpData(merged=True)
    new_data.sn = {}
    current_position_z = 0
    issparse = data_list[0].issparse()
    merge_raw = True
    raw_list = []
    for i in range(data_count):
        data: StereoExpData = data_list[i]
        if data.raw is None:
            merge_raw = False
        else:
            raw_list.append(data.raw)
        batch = i if batch_tags is None or i >= len(batch_tags) else batch_tags[i]
        data.cells.batch = batch
        cell_names = np.char.add(data.cells.cell_name, f"-{batch}")
        new_data.sn[str(batch)] = data.sn
        if i == 0:
            new_data.exp_matrix = data.exp_matrix.copy()
            new_data.cells = Cell(
                obs=data.cells.obs.set_index(cell_names),
                # cell_name=cell_names,
                # cell_border=data.cells.cell_border,
                batch=data.cells.batch
            )
            new_data.genes = Gene(var=data.genes.var.copy(deep=True))
            # position = data.position
            if data.position_z is None:
                position_z = np.repeat([[0]], repeats=data.n_cells, axis=0).astype(data.position.dtype)
            else:
                position_z = data.position_z
            # new_data.spatial = np.concatenate([position, position_z], axis=1)
            new_data.file_format = data.file_format
            new_data.spatial_key = data.spatial_key
            new_data.bin_type = data.bin_type
            new_data.bin_size = data.bin_size
            new_data.offset_x = data.offset_x
            new_data.offset_y = data.offset_y
            new_data.attr = data.attr
            cells_matrix_keys = []
            # genes_matrix_keys = list(data.genes_matrix.keys())
            for key, value in data.cells_matrix.items():
                if isinstance(value, (np.ndarray, sp.csr_matrix)):
                    new_data.cells_matrix[key] = value.copy()
                    cells_matrix_keys.append(key)
                elif isinstance(value, pd.DataFrame):
                    new_data.cells_matrix[key] = value.copy(deep=True)
                    if np.all(value.index == data.cell_names):
                        new_data.cells_matrix[key].index = cell_names
                    cells_matrix_keys.append(key)
                else:
                    logger.warning(f"got an unexpected type of cells_matrix: {type(value)}")
            for key, value in data.genes_matrix.items():
                new_data.genes_matrix[key] = deepcopy(value)
                if isinstance(value, pd.DataFrame):
                    new_data.genes_matrix[key].index = data.genes.gene_name
            
            # layer_keys = list(data.layers.keys())
            for key, value in data.layers.items():
                new_data.layers[key] = deepcopy(value)
            
            new_data.tl.key_record = deepcopy(data.tl.key_record)
            for key, result in data.tl.result.items():
                dict.__setitem__(new_data.tl.result, key, deepcopy(result))
        else:
            current_obs = data.cells.obs.set_index(cell_names)
            new_data.cells._obs = pd.concat([new_data.cells.obs, current_obs])
            # if new_data.cell_borders is not None and data.cell_borders is not None:
            #     new_data.cells.cell_border = np.concatenate([new_data.cells.cell_border, data.cells.cell_border])
            # position = data.position
            if data.position_z is None:
                current_position_z += space_between / data.attr['resolution']
                position_z = np.concatenate([position_z, np.repeat([[current_position_z]], repeats=data.n_cells, axis=0)], axis=0)
            else:
                position_z = np.concatenate([position_z, data.position_z], axis=0)
            # current_spatial = np.concatenate([position, position_z], axis=1)
            # new_data.spatial = np.concatenate([new_data.spatial, current_spatial], axis=0)
            # if var_type == "intersect":
            #     new_data.genes.gene_name, ind1, ind2 = \
            #         np.intersect1d(new_data.genes.gene_name, data.genes.gene_name, return_indices=True)
            #     if issparse:
            #         new_data.exp_matrix = sp.vstack([new_data.exp_matrix[:, ind1], data.exp_matrix[:, ind2]])
            #     else:
            #         new_data.exp_matrix = np.concatenate([new_data.exp_matrix[:, ind1], data.exp_matrix[:, ind2]])
            #     for key in new_data.genes_matrix.keys():
            #         if isinstance(new_data.genes_matrix[key], pd.DataFrame):
            #             new_data.genes_matrix[key] = deepcopy(new_data.genes_matrix[key].iloc[ind1])
            #         else:
            #             new_data.genes_matrix[key] = deepcopy(new_data.genes_matrix[key][ind1])
            #     for key in data.genes_matrix.keys():
            #         if key in new_data.genes_matrix:
            #             continue
            #         if isinstance(data.genes_matrix[key], pd.DataFrame):
            #             new_data.genes_matrix[key] = deepcopy(data.genes_matrix[key].iloc[ind2])
            #         else:
            #             new_data.genes_matrix[key] = deepcopy(data.genes_matrix[key][ind2])
            # elif var_type == "union":
            #     new_data.genes.gene_name, new_data.exp_matrix = _union_merge(
            #         new_data.exp_matrix, data.exp_matrix, 
            #         new_data.genes.gene_name, data.genes.gene_name
            #     )
            # else:
            #     raise Exception(f"got an unexpected var_type: {var_type}")
            _merge_matrix(new_data, data, var_type)
            
            for key in cells_matrix_keys:
                if key not in data.cells_matrix:
                    del new_data.cells_matrix[key]
                    continue
                if type(new_data.cells_matrix[key]) != type(data.cells_matrix[key]):
                    del new_data.cells_matrix[key]
                    continue
                if isinstance(new_data.cells_matrix[key], np.ndarray):
                    new_data.cells_matrix[key] = np.concatenate([new_data.cells_matrix[key], data.cells_matrix[key]])
                elif isinstance(new_data.cells_matrix[key], sp.spmatrix):
                    new_data.cells_matrix[key] = sp.vstack([new_data.cells_matrix[key], data.cells_matrix[key]])
                elif isinstance(new_data.cells_matrix[key], pd.DataFrame):
                    if np.all(data.cells_matrix[key].index == data.cell_names):
                        new_data.cells_matrix[key] = pd.concat([new_data.cells_matrix[key], data.cells_matrix[key]], axis=0)
                        new_data.cells_matrix[key].index = new_data.cell_names
                    else:
                        new_data.cells_matrix[key] = pd.concat([new_data.cells_matrix[key], data.cells_matrix[key]], axis=0)
            cells_matrix_keys = list(new_data.cells_matrix.keys())
            
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
                    else:
                        new_data.attr[key] = value
            
            for key, result in data.tl.result.items():
                if not dict.__contains__(new_data.tl.result, key):
                    dict.__setitem__(new_data.tl.result, key, deepcopy(result))
            
            for key, key_list in data.tl.key_record.items():
                if key not in new_data.tl.key_record:
                    new_data.tl.key_record[key] = deepcopy(key_list)
                elif len(key_list) > 0:
                    for k in key_list:
                        if k not in new_data.tl.key_record[key]:
                            new_data.tl.key_record[key].append(k)

    new_data.tl.review_key_record()

    new_data.position_z = position_z
    if reorganize_coordinate:
        new_data.position, new_data.position_offset, new_data.position_min = reorganize_data_coordinates(
            new_data.cells.batch, new_data.position, new_data.position_offset, new_data.position_min,
            reorganize_coordinate, horizontal_offset_additional, vertical_offset_additional
        )
    
    for column in new_data.cells.obs.columns:
        if column in data_list[0].cells.obs.columns and \
                data_list[0].cells.obs[column].dtype.name != new_data.cells.obs[column].dtype.name:
            new_data.cells.obs[column] = new_data.cells.obs[column].astype(data_list[0].cells.obs[column].dtype.name)
    
    if merge_raw:
        new_data.tl._raw = __merge_for_stereo_exp_data(
            *raw_list,
            reorganize_coordinate=reorganize_coordinate,
            horizontal_offset_additional=horizontal_offset_additional,
            vertical_offset_additional=vertical_offset_additional,
            space_between=space_between,
            var_type=var_type,
            batch_tags=batch_tags
        )

    return new_data


@merge.register(AnnBasedStereoExpData)
def __merge_for_ann_based_stereo_exp_data(
        *data_list: AnnBasedStereoExpData,
        reorganize_coordinate: Union[bool, int] = False,
        horizontal_offset_additional: Union[int, float] = 0,
        vertical_offset_additional: Union[int, float] = 0,
        space_between: Optional[str] = '0',
        var_type: str = "intersect",
        batch_tags: Union[list, np.ndarray, pd.Series] = None
):
    if data_list is None or len(data_list) < 2:
        raise Exception("At least two slices of data need to be input.")

    space_between = __parse_space_between(space_between)
    current_position_z = 0
    batches = []
    sn = {}
    adata_list = []
    position_z_list = []
    offset_x = None
    offset_y = None
    attr = None
    for i, data in enumerate(data_list):
        if batch_tags is None or i >= len(batch_tags):
            batch = str(i)
        else:
            batch = str(batch_tags[i])
        data.cells.batch = batch
        batches.append(batch)
        sn[batch] = data.sn
        adata_list.append(data.adata)

        if data.position_z is None:
            if i == 0:
                position_z = np.repeat([[0]], repeats=data.position.shape[0], axis=0).astype(
                    data.position.dtype)
            else:
                current_position_z += space_between / data.attr['resolution']
                position_z = np.repeat([[current_position_z]], repeats=data.position.shape[0], axis=0)
        else:
            position_z = data.position_z
        position_z_list.append(position_z)
        if i == 0:
            offset_x = data.offset_x
            offset_y = data.offset_y
            attr = data.attr
        else:
            if offset_x is not None and data.offset_x is not None:
                offset_x = min(offset_x, data.offset_x)
            if offset_y is not None and data.offset_y is not None:
                offset_y = min(offset_y, data.offset_y)
            if attr is not None and data.attr is not None:
                for key, value in data.attr.items():
                    if key in ('minX', 'minY'):
                        attr[key] = min(attr[key], value)
                    elif key in ('maxX', 'maxY'):
                        attr[key] = max(attr[key], value)
                    elif key == 'minExp':
                        attr['minExp'] = min(attr['minExp'], data.exp_matrix.min())
                    elif key == 'maxExp':
                        attr['maxExp'] = max(attr['maxExp'], data.exp_matrix.max())
                    elif key == 'resolution':
                        attr['resolution'] = value

    adata_merged = ad.concat(
        adata_list,
        join='inner' if var_type != 'union' else 'outer',
        axis=0,
        label='batch',
        keys=batches,
        index_unique='-',
        merge='first',
        uns_merge='first'
    )
    bin_type = data_list[0].bin_type
    bin_size = data_list[0].bin_size
    spatial_key = data_list[0].spatial_key
    new_data = AnnBasedStereoExpData(
        based_ann_data=adata_merged,
        bin_type=bin_type, bin_size=bin_size,
        spatial_key=spatial_key
    )
    new_data.merged = True
    new_data.offset_x = offset_x
    new_data.offset_y = offset_y
    new_data.attr = attr
    new_data.sn = sn
    new_data.file_format = data_list[0].file_format

    if new_data.adata.obsm[spatial_key].shape[1] == 2:
        position_z = np.concatenate(position_z_list, axis=0)
        new_data.position_z = position_z
    
    new_data.tl.review_key_record()
    
    if reorganize_coordinate:
        new_data.position, new_data.position_offset, new_data.position_min = reorganize_data_coordinates(
            new_data.cells.batch, new_data.position, new_data.position_offset, new_data.position_min,
            reorganize_coordinate, horizontal_offset_additional, vertical_offset_additional
        )
    return new_data

@singledispatch
def split(data: Union[StereoExpData, AnnBasedStereoExpData] = None):
    """
    Split a data object which is merged from different batches of data, according to the batch number.

    :param data: a merged data object.

    :return: A split data list.
    """
    pass

@split.register(StereoExpData)
def split_for_stereo_exp_data(data: StereoExpData = None):

    if data is None:
        return None

    from copy import deepcopy
    from .pipeline_utils import cell_cluster_to_gene_exp_cluster

    all_data = []
    # data.array2sparse()
    batch = np.unique(data.cells.batch)
    result = data.tl.result
    for bno in batch:
        cell_idx = np.where(data.cells.batch == bno)[0]
        # cell_names = data.cell_names[cell_idx]
        new_data = StereoExpData(
            file_format=data.file_format,
            bin_type=data.bin_type,
            bin_size=data.bin_size,
            exp_matrix=deepcopy(data.exp_matrix),
            cells=deepcopy(data.cells),
            genes=deepcopy(data.genes),
            # offset_x=data.offset_x,
            # offset_y=data.offset_y,
            attr=deepcopy(data.attr),
            spatial_key=data.spatial_key,
        )
        for key, value in data.layers.items():
            new_data.layers[key] = deepcopy(value)
        new_data.tl.raw = data.tl.raw
        new_data.position_offset = data.position_offset
        new_data.position_min = data.position_min
        # new_data.cells = new_data.cells.sub_set(cell_idx)
        # if data.position_offset is not None:
        #     new_data.position = data.position[cell_idx] - data.position_offset[bno]
        # else:
        #     new_data.position = data.position[cell_idx]
        # new_data.position_z = data.position_z[cell_idx]
        # new_data.exp_matrix = data.exp_matrix[cell_idx]
        new_data.sub_by_index(cell_index=cell_idx)
        new_data.cells.obs.index = new_data.cells.obs.index.str.replace(f'-{bno}$', '', regex=True)
        new_data.raw.cells.obs.index = new_data.raw.cells.obs.index.str.replace(f'-{bno}$', '', regex=True)
        new_data.reset_position()
        # if data.position_offset is not None:
        #     new_data.position = new_data.position - data.position_offset[bno]
        new_data.tl.key_record = deepcopy(data.tl.key_record)
        new_data.sn = data.sn[bno]
        for key, all_res_key in data.tl.key_record.items():
            if len(all_res_key) == 0:
                continue
            if key == 'hvg':
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
            elif key in ['pca', 'cluster', 'umap', 'totalVI']:
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
                    umi_cells = pd.Index(result[res_key][1]['umi_cells'])
                    umi_cells = umi_cells.str.replace('-\d+$', '', regex=True).to_numpy()
                    cells_bool_list = np.isin(umi_cells, new_data.cell_names)
                    # genes_bool_list = np.isin(result[res_key][1]['umi_genes'], new_data.gene_names)
                    res1 = {
                        'counts': result[res_key][0]['counts'][:, cells_bool_list],
                        'data': result[res_key][0]['data'][:, cells_bool_list]
                    }
                    if isinstance(result[res_key][0]['scale.data'], pd.DataFrame):
                        columns = result[res_key][0]['scale.data'].columns[cells_bool_list]
                        scale_data = result[res_key][0]['scale.data'][columns].copy()
                        scale_data.columns = columns.str.replace('-\d+$', '', regex=True)
                    else:
                        scale_data = result[res_key][0]['scale.data'][:, cells_bool_list]
                    res1['scale.data'] = scale_data

                    res2 = {
                        'umi_cells': pd.Index(umi_cells[cells_bool_list]).str.replace('-\d+$', '', regex=True).to_numpy(),
                        'umi_genes': result[res_key][1]['umi_genes'].copy(),
                        'top_features': result[res_key][1]['top_features'].copy()
                    }

                    # sct `counts` and `data` should have same shape
                    new_data.tl.result[res_key] = (res1, res2)
            elif key == 'tsne':
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
            elif key == 'gene_exp_cluster':
                continue
            else:
                for res_key in all_res_key:
                    new_data.tl.result[res_key] = result[res_key]
        # if data.tl.raw is not None:
        #     new_data.tl.raw = data.tl.raw.tl.filter_cells(cell_list=cell_names, inplace=False)
        if 'gene_exp_cluster' in data.tl.key_record:
            for cluster_res_key in data.tl.key_record['cluster']:
                gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(new_data, cluster_res_key)
                if gene_exp_cluster_res is not False:
                    new_data.tl.result[f"gene_exp_{cluster_res_key}"] = gene_exp_cluster_res
        all_data.append(new_data)

    return all_data


@split.register(AnnBasedStereoExpData)
def split_for_ann_based_stereo_exp_data(data: AnnBasedStereoExpData = None):

    if data is None:
        return None

    from copy import deepcopy
    from .pipeline_utils import cell_cluster_to_gene_exp_cluster

    all_data = []
    # data.array2sparse()
    batch = np.unique(data.cells.batch)
    for bno in batch:
        adata = data.adata[data.adata.obs['batch'] == bno].copy()
        adata.obs_names = adata.obs_names.str.replace(f'-{bno}$', '', regex=True)
        # adata.uns = adata.uns.copy()
        new_data = AnnBasedStereoExpData(based_ann_data=adata, spatial_key=data.spatial_key)
        new_data.tl.key_record = deepcopy(data.tl.key_record)
        new_data.sn = data.sn[bno]
        new_data.position_offset = data.position_offset
        new_data.position_min = data.position_min
        new_data.reset_position()
        # if data.position_offset is not None:
        #     new_data.position = new_data.position - data.position_offset[bno]

        # if data.tl.raw is not None:
        #     new_data.tl.raw = data.tl.raw.tl.filter_cells(cell_list=new_data.cells.cell_name, inplace=False)
        for key, res_keys in new_data.tl.key_record.items():
            if key == 'sct' and res_keys is not None and len(res_keys) > 0:
                for res_key in res_keys:
                    sct_key = new_data.adata.uns[res_key]['sct_key']
                    sct_id = new_data.adata.uns[res_key]['id']
                    sct_key_1 = f'{sct_key}_{sct_id}_1'
                    sct_key_2 = f'{sct_key}_{sct_id}_2'
                    umi_cells = pd.Index(new_data.adata.uns[sct_key_2]['umi_cells'])
                    umi_cells = umi_cells.str.resplace('-\d+$', '', regex=True).to_numpy()
                    cells_bool_list = np.isin(umi_cells, new_data.cell_names)
                    new_data.adata.uns[sct_key_1]['counts'] = new_data.adata.uns[sct_key_1]['counts'][:, cells_bool_list]
                    new_data.adata.uns[sct_key_1]['data'] = new_data.adata.uns[sct_key_1]['data'][:, cells_bool_list]
                    if isinstance(new_data.adata.uns[sct_key_1]['scale.data'], pd.DataFrame):
                        columns = new_data.adata.uns[sct_key_1]['scale.data'].columns[cells_bool_list]
                        scale_data = new_data.adata.uns[sct_key_1]['scale.data'][columns].copy()
                        scale_data.columns = columns.str.replace('-\d+$', '', regex=True)
                    else:
                        scale_data = new_data.adata.uns[sct_key_1]['scale.data'][:, cells_bool_list]
                    new_data.adata.uns[sct_key_1]['scale.data'] = scale_data
                    new_data.adata.uns[sct_key_2]['umi_cells'] = pd.Index(umi_cells[cells_bool_list]).str.replace('-\d+$', '', regex=True).to_numpy()

        if 'gene_exp_cluster' in data.tl.key_record:
            for cluster_res_key in data.tl.key_record['cluster']:
                gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(new_data, cluster_res_key)
                if gene_exp_cluster_res is not False:
                    new_data.tl.result[f"gene_exp_{cluster_res_key}"] = gene_exp_cluster_res
        all_data.append(new_data)
    return all_data