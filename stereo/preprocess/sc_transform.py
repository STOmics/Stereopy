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
        **kwargs
):
    if not issparse(data.exp_matrix):
        data.exp_matrix = csr_matrix(data.exp_matrix)

    # set do_correct_umi as False for less memory cost
    res = SCTransform(
        data.exp_matrix.T.tocsr(),
        data.gene_names,
        data.cell_names,
        n_genes=n_genes,
        n_cells=n_cells,
        do_correct_umi=do_correct_umi,
        return_only_var_genes=filter_hvgs,
        variable_features_n=var_features_n,
        seed_use=seed_use,
        **kwargs
    )
    new_exp_matrix = res[0][exp_matrix_key]
    if issparse(new_exp_matrix):
        # data.sub_by_index(gene_index=res[1]['umi_genes'])
        filter_genes(data, gene_list=res[1]['umi_genes'], inplace=True)
        data.exp_matrix = new_exp_matrix.T.tocsr()
        # gene_index = np.isin(data.gene_names, res[1]['umi_genes'])
        # data.genes = data.genes.sub_set(gene_index)
    else:
        # data.sub_by_index(gene_index=new_exp_matrix.index.to_numpy())
        filter_genes(data, gene_list=new_exp_matrix.index.to_numpy(), inplace=True)
        data.exp_matrix = new_exp_matrix.T.to_numpy()
        # gene_index = np.isin(data.gene_names, new_exp_matrix.index.values)
        # data.genes = data.genes.sub_set(gene_index)
    return res[0], res[1]
