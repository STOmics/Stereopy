from typing import Sequence, Optional, Union
import pandas as pd
import numpy as np
from stereo.log_manager import logger

def cell_cluster_to_gene_exp_cluster(
    tl,
    cluster_res_key: str = None,
    groups: Union[Optional[Sequence[str]], str] = None,
    genes: Union[Optional[Sequence[str]], str] = None,
    kind='sum'):
    if  tl.raw is None:
        logger.warn(
            """
            The function cell_cluster_to_gene_exp_cluster must be based on raw data.
            Please run data.tl.raw_checkpoint() before Normalization.
            """
        )
        return False
    if cluster_res_key is None:
        logger.warn("The parameter cluster_res_key of the function cell_cluster_to_gene_exp_cluster must be input")
        return False
    
    if cluster_res_key not in tl.result:
        logger.warn(f"The cluster_res_key '{cluster_res_key}' is not exists")
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
