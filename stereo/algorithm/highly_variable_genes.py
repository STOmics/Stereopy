import pandas as pd
import numpy as np
from typing import Optional
from ..utils.hvg_utils import materialize_as_ndarray, get_mean_var, check_nonnegative_integers
from ..log_manager import logger
import warnings
import scipy.sparse as sp_sparse


def highly_variable_genes_single_batch(
    data: Optional[sp_sparse.spmatrix],
    min_disp: Optional[float] = 0.5,
    max_disp: Optional[float] = np.inf,
    min_mean: Optional[float] = 0.0125,
    max_mean: Optional[float] = 3,
    n_top_genes: Optional[int] = None,
    n_bins: int = 20,
    method: Optional[str] = 'seurat',
) -> pd.DataFrame:
    """\
    See `highly_variable_genes`.

    Returns
    -------
    A DataFrame that contains the columns
    `highly_variable`, `means`, `dispersions`, and `dispersions_norm`.
    """
    # TODO how to deal with log data
    # if method == 'seurat':
        # if 'log1p' in adata.uns_keys() and adata.uns['log1p']['base'] is not None:
        #     X *= np.log(adata.uns['log1p']['base'])
        # data = np.expm1(data)

    mean, var = materialize_as_ndarray(get_mean_var(data))
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    if method == 'seurat':  # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
    # all of the following quantities are "per-gene" here
    df = pd.DataFrame()
    df['means'] = mean
    df['dispersions'] = dispersion
    if method == 'seurat':
        df['mean_bin'] = pd.cut(df['means'], bins=n_bins)
        disp_grouped = df.groupby('mean_bin')['dispersions']
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1
        one_gene_per_bin = disp_std_bin.isnull()
        gen_indices = np.where(one_gene_per_bin[df['mean_bin'].values])[0].tolist()
        if len(gen_indices) > 0:
            logger.debug(
                f'Gene indices {gen_indices} fell into a single bin: their '
                'normalized dispersion was set to 1.\n    '
                'Decreasing `n_bins` will likely avoid this effect.'
            )
        # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
        # but there’s still a dtype error without “.value”.
        disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[
            one_gene_per_bin.values
        ].values
        disp_mean_bin[one_gene_per_bin.values] = 0
        # actually do the normalization
        df['dispersions_norm'] = (
            df['dispersions'].values  # use values here as index differs
            - disp_mean_bin[df['mean_bin'].values].values
        ) / disp_std_bin[df['mean_bin'].values].values
    elif method == 'cell_ranger':
        from statsmodels import robust

        df['mean_bin'] = pd.cut(
            df['means'],
            np.r_[-np.inf, np.percentile(df['means'], np.arange(10, 105, 5)), np.inf],
        )
        disp_grouped = df.groupby('mean_bin')['dispersions']
        disp_median_bin = disp_grouped.median()
        # the next line raises the warning: "Mean of empty slice"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            disp_mad_bin = disp_grouped.apply(robust.mad)
            df['dispersions_norm'] = (
                df['dispersions'].values - disp_median_bin[df['mean_bin'].values].values
            ) / disp_mad_bin[df['mean_bin'].values].values
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
    dispersion_norm = df['dispersions_norm'].values
    if n_top_genes is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[
            ::-1
        ].sort()  # interestingly, np.argpartition is slightly slower
        if n_top_genes > data.shape[1]:
            logger.info('`n_top_genes` > `adata.n_var`, returning all genes.')
            n_top_genes = data.shape[1]
        disp_cut_off = dispersion_norm[n_top_genes - 1]
        gene_subset = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
        logger.debug(
            f'the {n_top_genes} top genes correspond to a '
            f'normalized dispersion cutoff of {disp_cut_off}'
        )
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    df['highly_variable'] = gene_subset
    return df


def highly_variable_genes_seurat_v3(
    data: Optional[sp_sparse.spmatrix],
    n_top_genes: int = 2000,
    batch_info: Optional[np.ndarray] = None,
    check_values: bool = True,
    span: float = 0.3,
) -> Optional[pd.DataFrame]:
    """\
    See `highly_variable_genes`.

    For further implemenation details see https://www.overleaf.com/read/ckptrbgzzzpg

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields

    highly_variable : bool
        boolean indicator of highly-variable genes
    **means**
        means per gene
    **variances**
        variance per gene
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches
    highly_variable_rank : float
        Rank of the gene according to normalized variance, median rank in the case of multiple batches
    highly_variable_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as HVG
    """

    try:
        from skmisc.loess import loess
    except ImportError:
        raise ImportError(
            'Please install skmisc package via `pip install --user scikit-misc'
        )
    df = pd.DataFrame()
    # df = pd.DataFrame(index=adata.var_names)
    # X = adata.layers[layer] if layer is not None else adata.X

    if check_values and not check_nonnegative_integers(data):
        warnings.warn(
            "`flavor='seurat_v3'` expects raw count data, but non-integers were found.",
            UserWarning,
        )

    df['means'], df['variances'] = get_mean_var(data)

    if batch_info is None:
        batch_info = pd.Categorical(np.zeros(data.shape[0], dtype=int))
    # else:
    #     batch_info = batch_info

    norm_gene_vars = []
    for b in np.unique(batch_info):
        X_batch = data[batch_info == b]

        mean, var = get_mean_var(X_batch)
        not_const = var > 0
        estimat_var = np.zeros(data.shape[1], dtype=np.float64)

        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        batch_counts = X_batch.astype(np.float64).copy()
        # clip large values as in Seurat
        N = X_batch.shape[0]
        vmax = np.sqrt(N)
        clip_val = reg_std * vmax + mean
        if sp_sparse.issparse(batch_counts):
            batch_counts = sp_sparse.csr_matrix(batch_counts)
            mask = batch_counts.data > clip_val[batch_counts.indices]
            batch_counts.data[mask] = clip_val[batch_counts.indices[mask]]

            squared_batch_counts_sum = np.array(batch_counts.power(2).sum(axis=0))
            batch_counts_sum = np.array(batch_counts.sum(axis=0))
        else:
            clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
            np.putmask(
                batch_counts,
                batch_counts > clip_val_broad,
                clip_val_broad,
            )

            squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
            batch_counts_sum = batch_counts.sum(axis=0)

        norm_gene_var = (1 / ((N - 1) * np.square(reg_std))) * (
            (N * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)

    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df['highly_variable_nbatches'] = num_batches_high_var
    df['highly_variable_rank'] = median_ranked
    df['variances_norm'] = np.mean(norm_gene_vars, axis=0)

    sorted_index = (
        df[['highly_variable_rank', 'highly_variable_nbatches']]
        .sort_values(
            ['highly_variable_rank', 'highly_variable_nbatches'],
            ascending=[True, False],
            na_position='last',
        )
        .index
    )
    df['highly_variable'] = False
    df.loc[sorted_index[: int(n_top_genes)], 'highly_variable'] = True

    if batch_info is None:
        df = df.drop(['highly_variable_nbatches'], axis=1)
    return df
