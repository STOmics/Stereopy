import time
from random import sample
from typing import Optional

import numba
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from patsy.highlevel import dmatrix
from scipy.sparse import csr_matrix

from stereo.log_manager import logger
from .bw import bwSJ
from .ksmooth import ksmooth
from .utils import (
    make_cell_attr,
    cpu_count,
    dds,
    multi_pearson_residual,
    is_outlier,
    fit_poisson,
    row_gmean_sparse
)


def vst(
        umi: Optional[csr_matrix],
        genes,
        cells,
        latent_var='log_umi',
        batch_var=None,
        latent_var_nonreg=None,
        n_genes=2000,
        n_cells=None,
        method='poisson',
        do_regularize=True,
        theta_regularization='od_factor',
        res_clip_range=lambda umi: [-np.sqrt(umi.shape[1]), np.sqrt(umi.shape[1])],
        bin_size=500,
        min_cells=5,
        residual_type='pearson',
        return_cell_attr=False,
        return_gene_attr=True,
        return_corrected_umi=False,
        min_variance=-np.inf,
        bw_adjust=3,
        gmean_eps=1,
        theta_estimation_fun='theta.ml',
        theta_given=None,
        exclude_poisson=False,
        use_geometric_mean=True,
        use_geometric_mean_offset=False,
        fix_intercept=False,
        fix_slope=False,
        scale_factor=None,
        vst_flavor=None,
        seed_use=1448145
):
    # TODO: `vst.flavor` not completed
    if vst_flavor is not None:
        logger.warning("`vst.flavor` not completed")
        raise NotImplementedError

    # TODO: `method` not completed, only `poisson`
    if method != 'poisson':
        logger.warning("`method` not completed, only `poisson`")
        raise NotImplementedError

    res_clip_range = res_clip_range(umi)
    cell_attr = make_cell_attr(umi, cells, latent_var, batch_var, latent_var_nonreg)
    genes_cell_count = np.asarray((umi >= 0.01).sum(1), dtype=np.double)
    genes_cell_count_pd = pd.DataFrame(genes_cell_count, index=genes, dtype=np.double)
    genes_cell_bool_list = (genes_cell_count.T >= min_cells)[0]
    umi = umi[genes_cell_bool_list, :]
    genes = genes[genes_cell_bool_list,]

    if use_geometric_mean:
        genes_log_gmean = pd.DataFrame(np.log10(row_gmean_sparse(umi, gmean_eps)), index=genes)[0]
    else:
        genes_log_gmean = pd.DataFrame(np.log10(umi.mean(1)), index=genes)[0]

    if do_regularize is False and n_genes:
        n_genes = None
        logger.info("do_regularize is set to False, will use all genes")

    if n_cells and n_cells < umi.shape[1]:
        # down sample cells to speed up the first step
        cells_step1 = sample(cells.tolist(), n_cells)
        cells_step1_bool_list = np.isin(cells, cells_step1)
        genes_cell_count_step1 = umi[:, cells_step1_bool_list].sum(1)
        genes_step1 = genes[(genes_cell_count_step1 >= min_cells).T.tolist()[0],]
        genes_step1_bool_list = np.isin(genes, genes_step1)
        if use_geometric_mean:
            genes_log_gmean_step1 = pd.DataFrame(np.log10(row_gmean_sparse(umi[genes_step1_bool_list, :], gmean_eps)),
                                                 index=genes_step1)[0]
        else:
            genes_log_gmean_step1 = pd.DataFrame(np.log10(umi[genes_step1_bool_list, :].mean(1)), index=genes_step1)[0]
    else:
        cells_step1 = cells
        genes_step1 = genes
        genes_log_gmean_step1 = genes_log_gmean

    genes_amean, genes_var = None, None
    # TODO: `do_regularize` and `exclude_poisson` is True have some code to do something
    #    find over-dispersive genes, this is not default, will be finished
    if do_regularize and exclude_poisson:
        logger.warning('finding over-dispersive genes has not finished yet')
        raise NotImplementedError

    if n_genes and n_genes < len(genes_step1):
        sampling_prob = dds(genes_log_gmean_step1)
        genes_step1 = np.array(
            pd.DataFrame(genes_step1).sample(n_genes, weights=sampling_prob.T,
                                             random_state=np.random.RandomState(seed_use))).T[
            0]
        genes_step1_bool_list = np.isin(genes, genes_step1)
        if use_geometric_mean:
            genes_log_gmean_step1 = \
                pd.DataFrame(np.log10(row_gmean_sparse(umi[genes_step1_bool_list, :], gmean_eps)), index=genes_step1)[0]
        else:
            genes_log_gmean_step1 = pd.DataFrame(np.log10(umi[genes_step1_bool_list, :].mean(1)), index=genes_step1)[0]

    logger.info(f'gene-cell umi shape {umi.shape}, n_genes {n_genes} n_cells {n_cells}')

    # TODO: `model_str` need to be recheck, `latent_var` logic not completed
    model_str = "y ~ log_umi"

    data_step1 = cell_attr.loc[cells_step1,]
    cells_step1_bool_list = np.isin(cells, cells_step1)
    genes_step1_bool_list = np.isin(genes, genes_step1)
    start_time = time.time()
    model_pars = get_model_pars(
        genes_step1, bin_size, umi[genes_step1_bool_list,][:, cells_step1_bool_list],
        model_str, cells_step1, method, data_step1, theta_given,
        theta_estimation_fun, exclude_poisson, fix_intercept, fix_slope, use_geometric_mean,
        use_geometric_mean_offset)
    logger.info(f'get_model_pars finished, cost {time.time() - start_time} seconds')

    min_theta = 1e-07
    model_pars['theta'][model_pars['theta'] < min_theta] = min_theta
    model_pars.set_index(genes_step1, inplace=True)

    start_time = time.time()
    if do_regularize:
        model_pars_fit, outliers = reg_model_pars(
            model_pars, genes_log_gmean_step1,
            genes_log_gmean, cell_attr, batch_var, cells_step1,
            genes_step1, umi, bw_adjust, gmean_eps, theta_regularization,
            genes_amean, genes_var, exclude_poisson, fix_intercept,
            fix_slope, use_geometric_mean, use_geometric_mean_offset
        )
    else:
        model_pars_fit, outliers = model_pars, None

    logger.info(f'reg_model_pars finished, cost {time.time() - start_time} seconds')

    regressor_data = dmatrix("~log_umi", cell_attr, return_type='dataframe')
    # TODO: `latent_var_nonreg` not completed yet
    if latent_var_nonreg:
        raise NotImplementedError
    else:
        model_str_nonreg = ""
        model_pars_nonreg = None
        model_pars_final = model_pars_fit
        regressor_data_final = regressor_data

    start_time = time.time()
    if residual_type:
        # TODO `min_variance` not completed related to `vst_flavor`
        bin_ind = np.ceil(np.array(range(1, len(genes) + 1)) / bin_size)
        max_bin = int(np.max(bin_ind))
        res = np.vstack(Parallel(n_jobs=cpu_count(), backend='threading')(
            delayed(multi_pearson_residual)(i, model_pars_final, regressor_data_final, umi, residual_type, min_variance,
                                            genes, bin_ind)
            for i in range(1, max_bin + 1)
        ))
        res = pd.DataFrame(res, index=genes, columns=regressor_data_final.index.values)
    else:
        res = None
    logger.info(f'pearson_residual cost {time.time() - start_time} seconds')

    rv = {
        "y": res,
        "model_str": model_str,
        "model_pars": model_pars,
        "model_outlier": outliers,
        "model_pars_fit": model_pars_fit,
        "model_str_nonreg": model_str_nonreg,
        "model_pars_nonreg": model_pars_nonreg,
        "genes_log_gmean_step1": genes_log_gmean_step1,
        "cells_step1": cells_step1,
        "cell_attr": cell_attr,
        "umi_genes": genes,
        "umi_cells": regressor_data_final.index.values,
    }

    if return_corrected_umi:
        if residual_type == "pearson":
            start_time = time.time()
            rv["umi_corrected"] = correct(rv, genes, do_round=True, do_pos=True, scale_factor=scale_factor,
                                          bin_size=bin_size)
            logger.info(f'umi_corrected cost {time.time() - start_time} seconds')
        else:
            logger.info("will not return corrected UMI because residual type is not set to `pearson`")

    rv["y"][rv["y"] < res_clip_range[0]] = res_clip_range[0]
    rv["y"][rv["y"] > res_clip_range[1]] = res_clip_range[1]
    if not return_cell_attr:
        rv["cell_attr"] = None

    if return_gene_attr:
        gene_attr = pd.DataFrame()
        gene_attr["detection_rate"] = genes_cell_count_pd.loc[genes] / umi.shape[1]
        gene_attr["gmean"] = np.power(10, genes_log_gmean)
        umi_genes_mean = umi.mean(axis=1)
        gene_attr["amean"] = pd.DataFrame(umi_genes_mean, index=genes)
        gene_attr["variance"] = \
            pd.DataFrame(np.power((umi - umi_genes_mean), 2).sum(1) / (umi.shape[1] - 1), index=genes)[0]
        if rv['y'].shape[1] > 0:
            gene_attr["residual_mean"] = rv['y'].mean(1)
            gene_attr["residual_variance"] = rv['y'].var(1)
        rv["gene_attr"] = gene_attr
    return rv


def get_model_pars(genes_step1, bin_size, umi, model_str, cells_step1, method, data_step1, theta_given,
                   theta_estimation_fun, exclude_poisson, fix_intercept, fix_slope, use_geometric_mean,
                   use_geometric_mean_offset) -> pd.DataFrame:
    # TODO: ignore `fix_intercept`、`fix_slope`
    if fix_intercept or fix_slope:
        raise NotImplementedError
    if method.startswith('offset'):
        raise NotImplementedError
    if method == 'poisson':
        model_pars = fit_poisson(umi=umi, model_str=model_str, data=data_step1,
                                 theta_estimation_fun=theta_estimation_fun)
    else:
        raise NotImplementedError
    if exclude_poisson:
        raise NotImplementedError
    return model_pars


def reg_model_pars(
        model_pars,
        genes_log_gmean_step1,
        genes_log_gmean,
        cell_attr,
        batch_var,
        cells_step1,
        genes_step1,
        umi,
        bw_adjust,
        gmean_eps,
        theta_regularization,
        genes_amean=None,
        genes_var=None,
        exclude_poisson=False,
        fix_intercept=False,
        fix_slope=False,
        use_geometric_mean=True,
        use_geometric_mean_offset=False
):
    genes = genes_log_gmean.index.values
    if exclude_poisson or fix_slope or fix_intercept:
        raise NotImplementedError
    dispersion_par = pd.DataFrame()
    if theta_regularization == 'theta':
        dispersion_par[theta_regularization] = np.log10(model_pars["theta"])
    elif theta_regularization == 'od_factor':
        dispersion_par[theta_regularization] = np.log10(1 + np.power(10, genes_log_gmean_step1) / model_pars["theta"])
    else:
        raise Exception(f"{theta_regularization} unknown - only log_theta and od_factor supported at the moment")

    model_pars = model_pars[model_pars.columns[model_pars.columns.values != "theta"]]
    model_pars['dispersion_par'] = dispersion_par.values

    outliers = Parallel(n_jobs=min(cpu_count(), 3), backend="threading")(
        delayed(is_outlier)(col, genes_log_gmean_step1)
        for _, col in model_pars.T.iterrows()
    )

    outliers = np.any(np.array(outliers).T, axis=1)
    if exclude_poisson:
        raise NotImplementedError

    outliers_pd = pd.DataFrame(outliers, dtype=bool, index=model_pars.index.values)
    if np.any(outliers):
        model_pars = model_pars[outliers != True]  # noqa
        # TODO tag genes step1
        # genes_step1 = model_pars.index.values
        genes_log_gmean_step1 = genes_log_gmean_step1[outliers != True]  # noqa

    min_step1, max_step1 = genes_log_gmean_step1.min(), genes_log_gmean_step1.max()
    x_points = np.where(genes_log_gmean > min_step1, genes_log_gmean, min_step1)
    x_points = np.where(x_points < max_step1, x_points, max_step1)
    o = np.array(sorted(range(0, len(x_points)), key=lambda x: x_points[x]))
    model_pars_fit = pd.DataFrame(np.zeros(shape=(len(genes), 3)), index=genes, columns=model_pars.columns.values)
    model_pars_fit = model_pars_fit.reset_index()
    model_pars_fit = model_pars_fit.iloc[o,]

    # TODO: ignore `batch_var`
    start = time.time()
    bw = bwSJ(genes_log_gmean_step1) * bw_adjust
    for col_name, col_value in model_pars.T.iterrows():
        model_pars_fit[col_name] = ksmooth(genes_log_gmean_step1, col_value, x_points.copy(), 2, bw)[1]
    logger.info(f'ksmooth finished, cost {time.time() - start} seconds')

    model_pars_fit = model_pars_fit.sort_index().set_index('index')
    if theta_regularization == 'theta':
        theta = np.power(10, model_pars_fit["dispersion_par"])
    elif theta_regularization == 'od_factor':
        theta = np.power(10, genes_log_gmean) / (np.power(10, model_pars_fit["dispersion_par"]) - 1)
    else:
        raise Exception(f"theta_regularization {theta_regularization} unknown - only log_theta and od_factor supported")
    if exclude_poisson:
        raise NotImplementedError
    if fix_intercept:
        raise NotImplementedError
    if fix_slope:
        raise NotImplementedError
    model_pars_fit = model_pars_fit.loc[:, model_pars_fit.columns != "dispersion_par"]
    model_pars_fit['theta'] = theta
    return model_pars_fit, outliers_pd


def correct(x, genes, as_is=False, do_round=True, do_pos=True, scale_factor=None, bin_size=500):
    if not as_is:
        cell_attr = x['cell_attr'].copy()
        cell_attr['log_umi'] = [np.median(cell_attr['log_umi'])] * len(cell_attr['log_umi'])
    else:
        cell_attr = x['cell_attr']
    regressor_data = dmatrix("~log_umi", cell_attr, return_type='dataframe')
    bin_ind = np.ceil(np.array(range(1, len(genes) + 1)) / bin_size)
    max_bin = int(np.max(bin_ind))
    corrected_data = pd.concat(Parallel(n_jobs=cpu_count(), backend='threading')(
        delayed(multi_correct_data)(x, genes, bin_ind, x['y'], i, regressor_data)
        for i in range(1, max_bin + 1)
    ))
    if do_round:
        corrected_data = np.round(corrected_data, 0)
    if do_pos:
        corrected_data[corrected_data < 0] = 0
    return csr_matrix(corrected_data)


@numba.jit(cache=True, forceobj=True, nogil=True)
def multi_correct_data(x, genes, bin_ind, data, i, regressor_data):
    genes_bin = genes[bin_ind == i]
    pearson_residual_ = data.loc[genes_bin]
    coefs = x['model_pars_fit'].loc[genes_bin, ["Intercept", "log_umi"]]
    theta = x['model_pars_fit'].loc[genes_bin, 'theta']
    return get_correct_data(coefs, regressor_data, theta, pearson_residual_)


@numba.jit(cache=True, forceobj=True, nogil=True)
def get_correct_data(coefs, regressor_data, theta, pearson_residual_):
    mu = np.exp(np.dot(coefs, regressor_data.T))
    variance = mu + np.power(mu, 2) / theta.to_numpy().reshape(-1, 1)
    return mu + pearson_residual_.multiply(np.sqrt(variance))
