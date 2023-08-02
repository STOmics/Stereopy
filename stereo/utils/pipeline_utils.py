from natsort import natsorted
from typing import Sequence, Optional, Union
import pandas as pd
import numpy as np
import numba as nb
from stereo.log_manager import logger
from stereo.core.stereo_exp_data import StereoExpData
# from stereo.core.st_pipeline import StPipeline

def cell_cluster_to_gene_exp_cluster(
    tl,
    cluster_res_key: str = None,
    groups: Union[Optional[Sequence[str]], str] = None,
    genes: Union[Optional[Sequence[str]], str] = None,
    kind: str = 'sum'
):
    if  tl.raw is None:
        logger.warning(
            """
            The function cell_cluster_to_gene_exp_cluster must be based on raw data.
            Please run data.tl.raw_checkpoint() before Normalization.
            """
        )
        return False
    if cluster_res_key is None:
        logger.warning("The parameter cluster_res_key of the function cell_cluster_to_gene_exp_cluster must be input")
        return False
    
    if cluster_res_key not in tl.result:
        logger.warning(f"The cluster_res_key '{cluster_res_key}' is not exists")
        return False

    cluster_result: pd.DataFrame = tl.result[cluster_res_key].copy()
    cluster_result.reset_index(inplace=True)
    cluster_result.sort_values(by=['group', 'index'], inplace=True)
    group_index = cluster_result.groupby('group').agg(cell_index=('index', list))
    if groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        group_index = group_index.loc[groups]
    tmp = []
    tl.raw.array2sparse()
    raw_cells_isin_data = np.isin(tl.raw.cell_names, tl.data.cell_names)
    raw_genes_isin_data = np.isin(tl.raw.gene_names, tl.data.gene_names)
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        all_genes_isin = np.isin(tl.raw.gene_names, genes)
    else:
        all_genes_isin = True
    exp_matrix = tl.raw.exp_matrix[raw_cells_isin_data][:, (raw_genes_isin_data & all_genes_isin)]
    gene_names = tl.raw.gene_names[(raw_genes_isin_data & all_genes_isin)]

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
    return pd.DataFrame(cluster_exp_matrix, columns=gene_names, index=group_index.index).T


def cluster_bins_to_cellbins(
        bins_data: StereoExpData,
        cellbins_data: StereoExpData,
        bins_cluster_res_key: str,
):
    """
    Mapping clustering result of bins to conresponding cellbins.
    
    The clustering of a cell will be mapped to the clustering of a bin if this cell's coordinate is within this bin.

    :param bins_data: StereoExpData object of bins.
    :param cellbins_data: StereoExpData object of cellbins.
    :param bins_cluster_res_key: cluster result key in bins' result.
    :return: cellbins_data
    """
    if bins_cluster_res_key not in bins_data.tl.result:
        raise ValueError(f"the key {bins_cluster_res_key} is not in the bins' result.")

    @nb.njit(cache=True, nogil=True, parallel=True)
    def __locate_cellbins_to_bins(bins_position, bin_size, bins_groups_idx, cellbins_names, cellbins_position):
        cells_count = cellbins_position.shape[0]
        cells_groups_idx = np.empty((cells_count, ), dtype=bins_groups_idx.dtype)
        cells_bool_list = np.zeros((cells_count, )).astype(np.bool8)
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
        __locate_cellbins_to_bins(bins_data.position, bins_data.bin_size, bins_groups_idx, cellbins_data.cell_names, cellbins_data.position)
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
        gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(cellbins_data.tl, cellbins_cluster_res_key)
        if gene_exp_cluster_res is not False:
            cellbins_data.tl.result[f"gene_exp_{cellbins_cluster_res_key}"] = gene_exp_cluster_res
            cellbins_data.tl.reset_key_record('gene_exp_cluster', f"gene_exp_{cellbins_cluster_res_key}")
    return cellbins_data
