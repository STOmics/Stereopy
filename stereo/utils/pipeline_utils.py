from typing import (
    Sequence,
    Optional,
    Union
)

import numba as nb
import numpy as np
import pandas as pd

from stereo.core.stereo_exp_data import StereoExpData
from stereo.log_manager import logger


def cell_cluster_to_gene_exp_cluster(
        data: StereoExpData,
        cluster_res_key: str = None,
        groups: Union[Optional[Sequence[str]], str] = None,
        genes: Union[Optional[Sequence[str]], str] = None,
        kind: str = 'sum',
        filter_raw: bool = True
):
    use_raw = False
    if data.raw is not None:
        use_raw = True
    if not use_raw:
        logger.info("Can not find raw data, the data which may have been normalized will be used.")

    if cluster_res_key is None:
        logger.warning("The parameter cluster_res_key of the function cell_cluster_to_gene_exp_cluster must be input")
        return False

    if cluster_res_key not in data.tl.result:
        logger.warning(f"The cluster_res_key '{cluster_res_key}' is not exists")
        return False

    cluster_result: pd.DataFrame = data.tl.result[cluster_res_key].copy()
    cluster_result.reset_index(inplace=True)
    cluster_result.sort_values(by=['group', 'index'], inplace=True)
    group_index = cluster_result.groupby('group').agg(cell_index=('index', list))
    if groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        group_index = group_index.loc[groups]
    tmp = []
    if use_raw:
        data.raw.array2sparse()
        if filter_raw:
            raw_cells_isin_data = np.isin(data.raw.cell_names, data.cell_names)
            raw_genes_isin_data = np.isin(data.raw.gene_names, data.gene_names)
        else:
            raw_cells_isin_data = np.ones(data.raw.cell_names.shape, dtype=bool)
            raw_genes_isin_data = np.ones(data.raw.gene_names.shape, dtype=bool)
        exp_matrix = data.raw.exp_matrix[raw_cells_isin_data][:, raw_genes_isin_data]
        gene_names = data.raw.gene_names[raw_genes_isin_data]
    else:
        data.array2sparse()
        exp_matrix = data.exp_matrix
        gene_names = data.gene_names

    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        all_genes_isin = np.isin(data.gene_names, genes)
        exp_matrix = exp_matrix[:, all_genes_isin]
        gene_names = gene_names[all_genes_isin]

    if kind != 'mean':
        kind = 'sum'
    for _, cell_index in group_index.iterrows():
        cell_index = cell_index.to_numpy()[0]
        if kind == 'sum':
            exp_tmp = exp_matrix[cell_index].sum(axis=0).A[0]
        else:
            exp_tmp = exp_matrix[cell_index].mean(axis=0).A[0]
        tmp.append(exp_tmp)
    cluster_exp_matrix = np.vstack(tmp)
    group_index.index.name = None
    return pd.DataFrame(cluster_exp_matrix, columns=gene_names, index=group_index.index).T


def calc_pct_and_pct_rest(
        data: StereoExpData,
        cluster_res_or_key: Union[str, pd.DataFrame],
        gene_names: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        filter_raw: bool = True
):
    if data.raw is None:
        logger.warning(
            """
            The function calc_pct_and_pct_rest must be based on raw data.
            Please run data.tl.raw_checkpoint() before Normalization.
            """
        )
        return False
    if isinstance(cluster_res_or_key, str):
        if cluster_res_or_key not in data.tl.result:
            logger.warning(f"Can not find the cluster result in data.tl.result by key {cluster_res_or_key}")
            return False

    if filter_raw:
        raw_cells_isin_data = np.isin(data.raw.cell_names, data.cell_names)
        raw_genes_isin_data = np.isin(data.raw.gene_names, data.gene_names)
    else:
        raw_cells_isin_data = np.ones(data.raw.cell_names.shape, dtype=bool)
        raw_genes_isin_data = np.ones(data.raw.gene_names.shape, dtype=bool)
    if gene_names is not None:
        genes_isin_all = np.isin(data.raw.gene_names, gene_names)
    else:
        genes_isin_all = True
    raw_exp_matrix = data.raw.exp_matrix[raw_cells_isin_data][:, genes_isin_all & raw_genes_isin_data]
    gene_names = data.raw.gene_names[genes_isin_all & raw_genes_isin_data]
    exp_matrix_one_hot = (raw_exp_matrix > 0).astype(np.uint8)
    if isinstance(cluster_res_or_key, str):
        cluster_result: pd.DataFrame = data.tl.result[cluster_res_or_key].copy()
    else:
        cluster_result: pd.DataFrame = cluster_res_or_key.copy()
    if 'bins' not in cluster_result.columns:
        cluster_result.reset_index(drop=True, inplace=True)
    cluster_result.reset_index(inplace=True)
    cluster_result.sort_values(by=['group', 'index'], inplace=True)
    group_index = cluster_result.groupby('group').agg(cell_index=('index', list))
    group_check = group_index.apply(lambda x: 1 if len(x[0]) <= 0 else 0, axis=1, result_type='broadcast')
    group_empty_index_list = group_check[group_check['cell_index'] == 1].index.tolist()
    group_index.drop(index=group_empty_index_list, inplace=True)
    if groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        group_index = group_index.loc[groups]

    def _calc(a, exp_matrix_one_hot):
        cell_index = a[0]
        if isinstance(exp_matrix_one_hot, np.ndarray):
            sub_exp = exp_matrix_one_hot[cell_index].sum(axis=0)
            sub_exp_rest = exp_matrix_one_hot.sum(axis=0) - sub_exp
        else:
            sub_exp = exp_matrix_one_hot[cell_index].sum(axis=0).A[0]
            sub_exp_rest = exp_matrix_one_hot.sum(axis=0).A[0] - sub_exp
        sub_pct = sub_exp / len(cell_index)
        sub_pct_rest = sub_exp_rest / (data.raw.cell_names.size - len(cell_index))
        return sub_pct, sub_pct_rest

    pct_all = np.apply_along_axis(_calc, 1, group_index.values, exp_matrix_one_hot)
    pct = pd.DataFrame(pct_all[:, 0], columns=gene_names, index=group_index.index).T
    pct_rest = pd.DataFrame(pct_all[:, 1], columns=gene_names, index=group_index.index).T
    pct.columns.name = None
    pct.reset_index(inplace=True)
    pct.rename(columns={'index': 'genes'}, inplace=True)
    pct_rest.columns.name = None
    pct_rest.reset_index(inplace=True)
    pct_rest.rename(columns={'index': 'genes'}, inplace=True)
    return pct, pct_rest


def cluster_bins_to_cellbins(
        bins_data: StereoExpData,
        cellbins_data: StereoExpData,
        bins_cluster_res_key: str,
):
    """
    Mapping cluster result of bins to corresponding cellbins.

    The cluster of a cell will be mapped to the cluster of a bin if this cell's coordinate is within this bin.

    :param bins_data: StereoExpData object of bins.
    :param cellbins_data: StereoExpData object of cellbins.
    :param bins_cluster_res_key: cluster result key in `bins_data.tl.result`, the mapped result will be named as
                                `{bins_cluster_res_key}_from_bin` and added into `cellbins_data.tl.result`.

    :return: The object of StereoExpData assigned to parameter `cellbins_data`.
    """
    if bins_cluster_res_key not in bins_data.tl.result:
        raise ValueError(f"the key {bins_cluster_res_key} is not in the bins' result.")

    @nb.njit(cache=True, nogil=True, parallel=True)
    def __locate_cellbins_to_bins(bins_position, bin_size, bins_groups_idx, cellbins_names, cellbins_position):
        cells_count = cellbins_position.shape[0]
        cells_groups_idx = np.empty((cells_count,), dtype=bins_groups_idx.dtype)
        cells_bool_list = np.zeros((cells_count,)).astype(np.bool8)
        bins_position_end = bins_position + bin_size
        cellbins_position = cellbins_position.astype(bins_position.dtype)
        for i in nb.prange(cells_count):
            cell_position = cellbins_position[i]
            flag = (cell_position >= bins_position) & (cell_position <= bins_position_end)
            bool_list = flag[:, 0] & flag[:, 1]
            bins_groups_idx_selected = bins_groups_idx[bool_list]
            if bins_groups_idx_selected.size == 0:
                cells_groups_idx[i] = -1
                cells_bool_list[i] = False
                continue
            cells_groups_idx[i] = bins_groups_idx_selected[0]
            cells_bool_list[i] = True
        return cells_groups_idx[cells_bool_list], cellbins_names[cells_bool_list], cellbins_names[~cells_bool_list]

    bins_groups_idx = np.arange(bins_data.cell_names.shape[0], dtype=np.int64)
    cells_groups_idx, cells_located, cells_filtered = \
        __locate_cellbins_to_bins(bins_data.position, bins_data.bin_size, bins_groups_idx, cellbins_data.cell_names,
                                  cellbins_data.position)
    if len(cells_located) == 0:
        logger.warning("All cells can not be located to any bins!")
        return cellbins_data
    if len(cells_filtered) > 0:
        logger.warning(f"{len(cells_filtered)} cells can not be located to any bins.")
        cellbins_data.tl.filter_cells(cell_list=cells_located)
    if bins_cluster_res_key in bins_data.cells._obs:
        cells_groups = bins_data.cells._obs[bins_cluster_res_key][cells_groups_idx].reset_index(drop=True)
    else:
        cells_groups = bins_data.tl.result[bins_cluster_res_key]['group'][cells_groups_idx].reset_index(drop=True)
    cellbins_cluster_res_key = f'{bins_cluster_res_key}_from_bins'
    cellbins_cluster_result = pd.DataFrame(data={'bins': cellbins_data.cell_names, 'group': cells_groups})
    cellbins_data.tl.result[cellbins_cluster_res_key] = cellbins_cluster_result
    cellbins_data.tl.reset_key_record('cluster', cellbins_cluster_res_key)

    if cellbins_data.tl.raw is not None:
        gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(cellbins_data, cellbins_cluster_res_key)
        if gene_exp_cluster_res is not False:
            cellbins_data.tl.result[f"gene_exp_{cellbins_cluster_res_key}"] = gene_exp_cluster_res
            cellbins_data.tl.reset_key_record('gene_exp_cluster', f"gene_exp_{cellbins_cluster_res_key}")
    return cellbins_data
