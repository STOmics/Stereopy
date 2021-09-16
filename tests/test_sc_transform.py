from stereo.algorithm.pysctransform import get_hvg_residuals, vst, SCTransform
from stereo.preprocess.sc_transform import sc_transform
from stereo.algorithm.pysctransform.plotting import compare_with_sct

import pickle
with open('/ldfssz1/ST_BI/USER/qindanhua/projects/st/data/test_data.pickle', 'rb') as r:
    data = pickle.load(r)

method_list = ['offset', "theta_ml", "theta_lbfgs", "alpha_lbfgs"]  # 'fix-slope', 'glmgp'需要用到R


def sc_st():
    data_nor = sc_transform(data, method='theta_ml', filter_hvgs=True)


def sc_pkg():
    vst_out = vst(
        data.exp_matrix.T,
        gene_names=data.gene_names.tolist(),
        cell_names=data.cell_names.tolist(),
        method='theta_ml',
        n_cells=5000,
        n_genes=2000,
        exclude_poisson=False,
    )

