import pandas as pd
import numpy as np
from stereo.core.st_pipeline import StPipeline
from stereo.log_manager import logger

def cell_cluster_to_gene_exp_cluster(tl: StPipeline, cluster_res_key: str=None):
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
    tmp = []
    for _, cell_index in group_index.iterrows():
        cell_index = cell_index.to_numpy()[0]
        sub_exp_matrix = tl.raw.exp_matrix[cell_index].toarray() if tl.raw.issparse() else data.tl.raw.exp_matrix[cell_index]
        tmp.append(sub_exp_matrix.sum(axis=0))
    cluster_exp_matrix = np.vstack(tmp)
    return pd.DataFrame(cluster_exp_matrix, columns=tl.data.gene_names, index=group_index.index).T 