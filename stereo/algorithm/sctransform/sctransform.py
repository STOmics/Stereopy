import random
import time
from typing import Optional
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix

from stereo.log_manager import logger
from .scale_data import ScaleData
from .vst import vst


def SCTransform(
        umi: Optional[csr_matrix],
        genes,
        cells,
        reference_sct_model=None,
        do_correct_umi: Optional[bool] = True,
        n_cells: Union[int, None] = 5000,
        residual_features: Union[list, np.ndarray, None] = None,
        variable_features_n: Union[int, None] = 3000,
        variable_features_rv_th: float = 1.3,
        vars_to_regress: Union[str, list, None] = None,
        do_scale: Optional[bool] = False,
        do_center: Optional[bool] = True,
        clip_range=lambda umi: [-np.sqrt(umi.shape[1] / 30), np.sqrt(umi.shape[1] / 30)],
        conserve_memory: Optional[bool] = False,
        return_only_var_genes: Optional[bool] = True,
        seed_use: Union[int, None] = 1448145,
        **kwargs
):
    """
    A Single-Cell RNA Sequencing Transform Method:
        Use regularized negative binomial regression to normalize UMI count data.

    :param umi:
        A csr_matrix of UMI counts with genes as rows and cells as columns
    :param genes:
        Genes describe the `umi`
    :param cells:
        Cells describe the `umi`
    :param reference_sct_model: TODO: only `None` support now.
        If not None, compute residuals for the object using the provided SCT model; supports only log_umi as the latent
        variable. If `residual_features` are not specified, compute for the top variable_features_n specified in the
        model which are also present in the object. If residual_features are specified, the variable features of the
        resultingSCT assay are set to the top variable_features_n in the model.
    :param do_correct_umi: Place corrected UMI matrix in res[0]'s key `counts; default is True
    :param n_cells: Number of sub-sampling cells used to build NB regression; default is 5000
    :param residual_features: TODO not yet finished
        Genes to calculate residual features for; default is None (all genes). If specified, will be set to
        `top_features` of the returned res[1].
    :param variable_features_n:
        Use this many features as variable features after ranking by residual variance; default is 3000. Only applied if
        `residual_features` is not set.
    :param variable_features_rv_th:
        Instead of setting a fixed number of variable features, use this residual variance cutoff; this is only used
        when `variable_features_n` is set to None; default is 1.3. Only applied if `residual_features` is not set.
    :param vars_to_regress: TODO not yet finished
        Variables to regress out in a second non-regularized linear regression. Default is None
    :param do_scale: TODO not yet finished
        Whether to scale residuals to have unit variance; default is False
    :param do_center:
        Whether to center residuals to have mean zero; default is True
    :param clip_range:
        Range to clip the residuals to; default is [-np.sqrt(umi.shape[1] / 30), np.sqrt(umi.shape[1] / 30)], where n
        is the number of cells.
    :param conserve_memory: TODO not yet finished
        If set to True the residual matrix for all genes is never created in full; useful for large data sets, but will
        take longer to run; this will also set `return_only_var_genes` to True; default is False
    :param return_only_var_genes:
        If set to True the scale.data matrices in output res are subset to contain only the variable genes;
        default is True
    :param seed_use:
        Set a random seed. By default, sets the seed to 1448145. Setting None will not set a seed.
    :param kwargs:
        Other arguments, such as `n_genes` defined for `vst`.
    :return:
            {
                'counts':     csr_matrix,       # describe `umi` corrected by pearson residual
                'data':       csr_matrix,       # counts after `log1p`
                'scale.data': pandas.DataFrame, # pearson residual after `scale`
            },
            dict # vst_out
    """
    if seed_use:
        np.random.seed(seed_use)
        random.seed(seed_use)
        logger.info(f'using default random_seed {seed_use}, will run SCT without randomness')

    # different with seurat-SCT, we don't need to choose assay by using exp_matrix directly
    umi = umi.astype(dtype=np.double, copy=False)

    # FIXME: ignore `batch_var`
    if 'batch_var' in kwargs:
        logger.warning('`batch_var` not implemented yet, will add the feature in the future')
        raise NotImplementedError

    if callable(clip_range):
        clip_range = clip_range(umi)
    else:
        logger.warning(f'clip_range type: {type(clip_range)} is not callable')
        raise TypeError

    vst_args = kwargs
    vst_args['umi'] = umi
    vst_args['genes'] = genes
    vst_args['cells'] = cells
    vst_args['return_cell_attr'] = True
    vst_args['return_gene_attr'] = True
    vst_args['return_corrected_umi'] = do_correct_umi
    vst_args['n_cells'] = min(n_cells, umi.shape[1])
    vst_args['seed_use'] = seed_use
    # TODO: ignore `res_clip_range` used in 'conserve.memory' sct-method
    # res_clip_range = vst_args['res_clip_range'] if 'res_clip_range' in vst_args else [-math.sqrt(umi.shape[1]),
    #                                                                                  math.sqrt(umi.shape[1])]

    # TODO: ignore `reference_sct_model`、`residual_features` and `conserve.memory`,
    #     we will finish these features in the future (~_~)
    if reference_sct_model:
        # sct_method = 'reference.model'
        raise NotImplementedError
    elif residual_features:
        # sct_method = 'residual.features'
        raise NotImplementedError
    elif conserve_memory:
        # sct_method = 'conserve.memory'
        raise NotImplementedError
    else:
        sct_method = 'default'

    if sct_method == 'default':
        vst_out = vst(**vst_args)
    else:
        raise NotImplementedError

    feature_variance = vst_out['gene_attr']["residual_variance"]
    feature_variance = feature_variance.sort_values(ascending=False)
    vst_out['feature_variance'] = feature_variance
    if variable_features_n:
        top_features = feature_variance[:min(variable_features_n, len(feature_variance))].index.values
    else:
        top_features = feature_variance[feature_variance >= variable_features_rv_th].index.values
    vst_out['top_features'] = top_features

    if sct_method == 'default':
        if return_only_var_genes:
            vst_out['y']['level_0'] = range(0, vst_out['y'].shape[0])
            vst_out['y'] = vst_out['y'].loc[top_features].sort_values('level_0')
            del vst_out['y']['level_0']
    else:
        raise NotImplementedError

    # create output assay
    assay_out = {
        "counts": None,  # sparse matrix
        "data": None,  # sparse matrix
        "scale.data": None  # too dense to generate as sparse matrix
    }

    residual_type = vst_args['residual_type'] if 'residual_type' in vst_args else 'pearson'
    if do_correct_umi and residual_type == 'pearson':
        assay_out["counts"] = vst_out['umi_corrected']
    else:
        assay_out["counts"] = umi

    log1p_counts = assay_out['counts']
    assay_out["data"] = log1p_counts.log1p()

    genes_original_order = vst_out['y'].index.values
    scale_data = vst_out.pop('y')
    scale_data[scale_data < clip_range[0]] = clip_range[0]
    scale_data[scale_data > clip_range[1]] = clip_range[1]

    start_time = time.time()
    scale_data = ScaleData(
        scale_data,
        features=vst_out['top_features'] if return_only_var_genes else None,
        vars_to_regress=vars_to_regress,
        latent_data=vst_out['cell_attr'].T.loc[vars_to_regress] if vars_to_regress else None,
        model_use='linear',
        use_umi=False,
        do_scale=do_scale,
        do_center=do_center,
        scale_max=np.Inf,
        block_size=750,
        min_cells_to_block=3000,
    )
    logger.info(f'scale data cost {time.time() - start_time} seconds')

    if return_only_var_genes:
        assay_out["scale.data"] = scale_data.reindex(genes_original_order, copy=False)
    else:
        assay_out["scale.data"] = scale_data
    return assay_out, vst_out
