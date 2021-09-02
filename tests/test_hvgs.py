import sys
sys.path.append('/ldfssz1/ST_BI/USER/qindanhua/projects/st/stereopy/')

from stereo.tools.highly_variable_genes import HighlyVariableGenes
from stereo.algorithm.highly_variable_genes import highly_variable_genes_seurat_v3, highly_variable_genes_single_batch
from stereo.utils.hvg_utils import materialize_as_ndarray, get_mean_var, check_nonnegative_integers
# from stereo.io.reader import read_stereo

#
# path = '/ldfssz1/ST_BI/USER/qindanhua/projects/st/data/FP200000381BL_B2.bin1.Lasso.gem'
# data = read_stereo(path, 'bins', 200)
# df = data.to_df().reindex
# df.to_csv('/ldfssz1/ST_BI/USER/qindanhua/projects/st/data/exp.csv', index=False)
import pickle
# with open('/ldfssz1/ST_BI/USER/qindanhua/projects/st/data/test_data', 'wb') as w:
#     pickle.dump(data, w)

with open('/ldfssz1/ST_BI/USER/qindanhua/projects/st/data/b2_bin200.pk', 'rb') as r:
    data = pickle.load(r)

from anndata import AnnData
adata = AnnData(data.to_df())


def test_highly_variable_genes_basic(data):
    import numpy as np
    import pandas as pd
    hvg = HighlyVariableGenes(data, method='cell_ranger')
    hvg.fit()
    hvg.method = 'seurat_v3'
    hvg.fit()
    hvg.method = 'seurat'
    hvg.fit()

    # add group
    groups = pd.DataFrame({
        'group': np.random.binomial(3, 0.5, size=(len(data.cell_names)))
    }, index=data.cell_names)

    hvg = HighlyVariableGenes(data, groups=groups, method='cell_ranger', n_top_genes=200)
    hvg.fit()
    hvg.method = 'seurat_v3'
    hvg.fit()
    hvg.method = 'seurat'
    hvg.fit()
#
    adata.obs['batch'] = groups['group']
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    import scanpy as sc
    sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=200)

    mr = hvg.result[hvg.result['highly_variable']]
    mr_sc = adata.var[adata.var['highly_variable']]
    len(set(mr.index))
    len(set(mr.index))
    len(set(mr.index) & set(mr_sc.index))


# def test_single_batch():
#     import pandas as pd
#     matrix = pd.read_csv('/ldfssz1/ST_BI/USER/qindanhua/projects/st/data/exp_matrix.csv')
#     data = matrix.values()
#     highly_variable_genes_single_batch(data, method='seurat', n_top_genes=None)
