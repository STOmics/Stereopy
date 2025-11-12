import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from stereo.algorithm.sctransform import SCTransform
from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.filter import filter_genes


def sc_transform(
        data: StereoExpData,
        n_cells=5000,
        n_genes=2000,
        filter_hvgs=True,
        var_features_n=3000,
        do_correct_umi=False,
        exp_matrix_key='scale.data',
        seed_use=1448145,
        filter_raw=True,
        layer=None,
        n_jobs=8,
        **kwargs
):
    exp_matrix = data.get_exp_matrix(use_raw=False, layer=layer)
    if not issparse(exp_matrix):
        exp_matrix = csr_matrix(exp_matrix)

    # set do_correct_umi as False for less memory cost
    res = SCTransform(
        exp_matrix.T.tocsr(),
        data.gene_names,
        data.cell_names,
        n_genes=n_genes,
        n_cells=n_cells,
        do_correct_umi=do_correct_umi,
        return_only_var_genes=filter_hvgs,
        variable_features_n=var_features_n,
        seed_use=seed_use,
        n_jobs=n_jobs,
        **kwargs
    )
    new_exp_matrix = res[0][exp_matrix_key]
    if issparse(new_exp_matrix):
        filter_genes(data, gene_list=res[1]['umi_genes'], filter_raw=filter_raw, inplace=True)
        new_exp_matrix = new_exp_matrix.T.tocsr()
    else:
        filter_genes(data, gene_list=new_exp_matrix.index.to_numpy(), filter_raw=filter_raw, inplace=True)
        new_exp_matrix = new_exp_matrix.T.to_numpy()
    return res[0], res[1], new_exp_matrix
