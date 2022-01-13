"""Main module."""
import time
import warnings

from KDEpy import FFTKDE
from scipy import interpolate
from scipy import sparse
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)
import concurrent.futures
import logging

import numpy as npy
import pandas as pd
import statsmodels.discrete.discrete_model as dm
from joblib import Parallel
from joblib import delayed
from patsy import dmatrix
from scipy import stats
from scipy.sparse import csr_matrix
from statsmodels.nonparametric.kernel_regression import KernelReg
from tqdm import tqdm
from sklearn.utils.sparsefuncs import mean_variance_axis

logging.captureWarnings(True)


from .fit import alpha_lbfgs
from .fit import estimate_mu_poisson
from .fit import theta_lbfgs
from .fit import theta_ml
from .fit_glmgp import fit_glmgp
from .fit_glmgp import fit_glmgp_offset

from .r_bw import bw_SJr
from .r_bw import is_outlier_r
from .r_bw import ksmooth


def is_outlier_naive(x, snr_threshold=25):
    """
    Mark points as outliers
    """
    if len(x.shape) == 1:
        x = x[:, None]
    median = npy.median(x, axis=0)
    diff = npy.sum((x - median) ** 2, axis=-1)
    diff = npy.sqrt(diff)
    med_abs_deviation = npy.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > snr_threshold


def sparse_var(X, axis=None):
    # X2 = X.copy()
    # X2.data **= 2
    # return X2.mean(axis) - npy.square(X2.mean(axis))
    mean, var = mean_variance_axis(X, axis)
    return var


def bwSJ(genes_log10_gmean_step1, bw_adjust=3):
    # See https://kdepy.readthedocs.io/en/latest/bandwidth.html
    fit = FFTKDE(kernel="gaussian", bw="ISJ").fit(npy.asarray(genes_log10_gmean_step1))
    _ = fit.evaluate()
    bw = fit.bw * bw_adjust
    return npy.array([bw], dtype=float)


def robust_scale(x):
    return (x - npy.median(x)) / (
        stats.median_absolute_deviation(x) + npy.finfo(float).eps
    )


def robust_scale_binned(y, x, breaks):
    bins = pd.cut(x=x, bins=breaks, ordered=True)
    bins = pd.Series(data=[i for i in bins], dtype='category')
    # categories = bins.categories
    # bins = npy.digitize(x=x, bins=breaks)
    df = pd.DataFrame({"x": y, "bins": bins})
    tmp = df.groupby(["bins"]).apply(robust_scale)
    order = df["bins"].argsort()
    tmp = tmp.loc[order]  # sort_values(by=["bins"])
    score = tmp["x"]
    return score


def is_outlier(y, x, th=10):
    bin_width = (npy.nanmax(x) - npy.nanmin(x)) * bwSJ(x, bw_adjust=1 / 2)
    eps = npy.finfo(float).eps * 10
    bin_width = bin_width[0]
    breaks1 = npy.arange(
        start=npy.nanmin(x) - eps, stop=npy.nanmax(x) + bin_width, step=bin_width
    )
    breaks2 = npy.arange(
        start=npy.nanmin(x) - eps - bin_width / 2,
        stop=npy.nanmax(x) + bin_width,
        step=bin_width,
    )
    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)
    return npy.vstack((npy.abs(score1), npy.abs(score2))).min(0) > th


def make_cell_attr(umi, cell_names):
    assert umi.shape[1] == len(cell_names)
    total_umi = npy.squeeze(npy.asarray(umi.sum(0)))
    log10_umi = npy.log10(total_umi)
    expressed_genes = npy.squeeze(npy.asarray((umi > 0).sum(0)))
    log10_expressed_genes = npy.log10(expressed_genes)
    cell_attr = pd.DataFrame({"umi": total_umi, "log10_umi": log10_umi})
    cell_attr.index = cell_names
    cell_attr["n_expressed_genes"] = expressed_genes
    # this is referrred to as gene in SCTransform
    cell_attr["log10_gene"] = log10_expressed_genes
    cell_attr["umi_per_gene"] = log10_umi / expressed_genes
    cell_attr["log10_umi_per_gene"] = npy.log10(cell_attr["umi_per_gene"])
    return cell_attr


def row_gmean(umi, gmean_eps=1):
    gmean = npy.exp(npy.log(umi + gmean_eps).mean(1)) - gmean_eps
    return gmean


def row_gmean_sparse(umi, gmean_eps=1):

    gmean = npy.asarray(npy.array([row_gmean(x.todense(), gmean_eps)[0] for x in umi]))
    gmean = npy.squeeze(gmean)
    return gmean


def _process_y(y):
    if not isinstance(y, npy.ndarray):
        y = npy.array(y)
    y = npy.asarray(y, dtype=int)
    y = npy.squeeze(y)
    return y


def get_model_params_pergene(
    gene_umi,
    model_matrix,
    method="theta_ml",
    offset_intercept=None,
    cell_umi=None,
    fix_slope=False,
):  # latent_var, cell_attr):
    gene_umi = _process_y(gene_umi)
    if method == "sm_nb":
        model = dm.NegativeBinomial(gene_umi, model_matrix, loglike_method="nb2")
        params = model.fit(maxiter=50, tol=1e-3, disp=0).params
        theta = 1 / params[-1]
        if theta >= 1e5:
            theta = npy.inf
        params = dict(zip(model_matrix.design_info.column_names, params[:-1]))
        params["theta"] = theta
    elif method == "theta_ml":
        if fix_slope:
            params = pd.DataFrame(index=[0])
            params["theta"] = npy.nan
            params["Intercept"] = offset_intercept
            params["log10_umi"] = npy.log(10)
            mu = npy.exp(offset_intercept + npy.log(10) * cell_umi)
            assert len(gene_umi) == len(mu)
            theta = theta_ml(y=gene_umi, mu=mu)
        else:
            params = estimate_mu_poisson(gene_umi, model_matrix)
            coef = params["coef"]
            mu = params["mu"]
            params = dict(zip(model_matrix.design_info.column_names, coef))
            theta = theta_ml(y=gene_umi, mu=mu)
        if theta >= 1e5:
            theta = npy.inf
        params["theta"] = theta
    elif method == "alpha_lbfgs":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = alpha_lbfgs(y=gene_umi, mu=mu)
        params["theta"] = theta
    elif method == "theta_lbfgs":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = theta_lbfgs(y=gene_umi, mu=mu)
        params["theta"] = theta
    return params


def get_model_params_pergene_glmgp(gene_umi, coldata, design="~ log10_umi"):
    gene_umi = gene_umi.todense()
    params = fit_glmgp(y=gene_umi, coldata=coldata, design=design)
    return params


def get_model_params_pergene_glmgp_offset(gene_umi, coldata, log_umi, design="~ 1"):
    gene_umi = gene_umi.todense()
    params = fit_glmgp_offset(
        y=gene_umi, coldata=coldata, design=design, log_umi=log_umi
    )
    return params


def get_model_params_allgene_glmgp(
    umi, coldata, bin_size=500, threads=4, use_offset=False, verbosity=0
):

    results = []
    log_umi = npy.log(npy.ravel(umi.sum(0)))
    if use_offset:
        results = Parallel(n_jobs=threads, backend="multiprocessing", batch_size=500)(
            delayed(get_model_params_pergene_glmgp_offset)(row, coldata, log_umi)
            for row in umi
        )
    else:
        results = Parallel(n_jobs=threads, backend="multiprocessing", batch_size=500)(
            delayed(get_model_params_pergene_glmgp)(row, coldata) for row in umi
        )
    params_df = pd.DataFrame(results)

    return params_df


def get_model_params_allgene(
    umi, model_matrix, method="fit", threads=4, fix_slope=False, verbosity=0
):

    results = []
    if fix_slope:
        gene_mean = umi.mean(1)
        cell_umi = npy.log10(npy.ravel(umi.sum(0)))
        mean_cell_sum = npy.mean(cell_umi)
        offset_intercept = npy.log(gene_mean) - npy.log(mean_cell_sum)
        offset_intercept = npy.ravel(offset_intercept)
    else:
        offset_intercept = [npy.nan] * umi.shape[0]
        cell_umi = [npy.nan] * umi.shape[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # TODO this should remain sparse
        feed_list = [
            (
                row.todense().reshape((-1, 1)),
                model_matrix,
                method,
                offset_intercept[i],
                cell_umi,
                fix_slope,
            )
            for i, row in enumerate(umi)
        ]

        if verbosity:
            results = list(
                tqdm(
                    executor.map(lambda p: get_model_params_pergene(*p), feed_list),
                    total=len(feed_list),
                )
            )
        else:
            results = list(
                executor.map(lambda p: get_model_params_pergene(*p), feed_list)
            )

    params_df = pd.DataFrame(results)

    return params_df


def dds(genes_log10_gmean_step1, grid_points=2 ** 10):
    # density dependent downsampling
    # print(genes_log10_gmean_step1.shape)
    # if genes_log10_gmean_step1.ndim <2:
    #    genes_log10_gmean_step1 = genes_log10_gmean_step1[:, npy.newaxis]
    x, y = (
        FFTKDE(kernel="gaussian", bw="silverman")
        .fit(npy.asarray(genes_log10_gmean_step1))
        .evaluate(grid_points=grid_points)
    )
    density = interpolate.interp1d(x=x, y=y, assume_sorted=False)
    sampling_prob = 1 / (density(genes_log10_gmean_step1) + npy.finfo(float).eps)

    # sampling_prob = 1 / (density + npy.finfo(float).eps)
    return sampling_prob / sampling_prob.sum()


def get_regularized_params(
    model_parameters,
    genes,
    genes_step1,
    genes_log10_gmean_step1,
    genes_log10_gmean,
    cell_attr,
    umi,
    batch_var=None,
    bw_adjust=3,
    gmean_eps=1,
    theta_regularization="od_factor",
    exclude_poisson=False,
    poisson_genes=None,
    method="theta_ml",
):
    model_parameters = model_parameters.copy()

    model_parameters_fit = pd.DataFrame(
        npy.nan, index=genes, columns=model_parameters.columns
    )

    x_points_df = pd.DataFrame({"gene_log10_gmean": genes_log10_gmean})
    x_points_df["min_gene_log10_gmean_step1"] = genes_log10_gmean_step1.min()

    x_points_df["x_points"] = npy.nanmax(x_points_df, axis=1)
    x_points_df["max_gene_log10_gmean_step1"] = npy.nanmax(genes_log10_gmean_step1)
    x_points_df["x_points"] = x_points_df[
        ["x_points", "max_gene_log10_gmean_step1"]
    ].min(1)
    x_points = x_points_df["x_points"].values
    for column in model_parameters.columns:
        if column == "theta":
            continue
        endog = model_parameters.loc[genes_step1, column].values
        exog_fit = genes_log10_gmean_step1  # .values
        if method == "glgmp":
            bw = bw_SJr(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
            params = ksmooth(genes_log10_gmean, genes_log10_gmean_step1, endog, bw[0])
            index = model_parameters_fit.index.values[params["order"] - 1]
            model_parameters_fit.loc[index, column] = params["smoothed"]
        else:
            bw = bwSJ(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
            reg = KernelReg(endog=endog, exog=exog_fit, var_type="c", reg_type="ll", bw=bw)
            fit = reg.fit(x_points)
            model_parameters_fit[column] = npy.squeeze(fit[0])
        # bw = bwSJ(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
        # reg = KernelReg(endog=endog, exog=exog_fit, var_type="c", reg_type="ll", bw=bw)
        # fit = reg.fit(x_points)
        # model_parameters_fit[column] = npy.squeeze(fit[0])
        # # print(bw)
        # bw = bw_SJr(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
        # params = ksmooth(genes_log10_gmean, genes_log10_gmean_step1, endog, bw[0])
        # index = model_parameters_fit.index.values[params["order"] - 1]
        # model_parameters_fit.loc[index, column] = params["smoothed"]

    if theta_regularization == "theta":
        theta = npy.power(10, (model_parameters_fit["od_factor"]))
    else:
        theta = npy.power(10, genes_log10_gmean) / (
            npy.power(10, model_parameters_fit["od_factor"]) - 1
        )
    model_parameters_fit["theta"] = theta
    if exclude_poisson:
        # relace theta by inf
        if poisson_genes is not None:
            print("len poisson genes", len(poisson_genes))
            model_parameters_fit.loc[poisson_genes, "theta"] = npy.inf
            # model_parameters_fit["is_poisson"]= False
            # model_parameters_fit.loc[poisson_genes, "is_poisson"] = True
            if theta_regularization == "theta":
                model_parameters_fit.loc[poisson_genes, "od_factor"] = npy.inf
            else:
                model_parameters_fit.loc[poisson_genes, "od_factor"] = 0

            model_parameters_fit.loc[poisson_genes, "log10_umi"] = npy.log(10)
            gene_mean = pd.Series(npy.ravel(umi.mean(1)), index=genes)
            mean_cell_sum = npy.mean(npy.ravel(umi.sum(0)))
            model_parameters_fit.loc[poisson_genes, "Intercept"] = npy.log(
                gene_mean[poisson_genes]
            ) - npy.log(mean_cell_sum)

    return model_parameters_fit


def pearson_residual(y, mu, theta, min_var=-npy.inf):
    variance = mu + npy.divide(mu ** 2, theta.reshape(-1, 1))
    variance[variance < min_var] = min_var
    pearson_residuals = npy.divide(y - mu, npy.sqrt(variance))
    return pearson_residuals


def deviance_residual(y, mu, theta, weight=1):
    theta = npy.tile(theta.reshape(-1, 1), y.shape[1])
    L = npy.multiply((y + theta), npy.log((y + theta) / (mu + theta)))
    log_mu = npy.log(mu)
    log_y = npy.log(y.maximum(1).todense())
    r = npy.multiply(y.todense(), log_y - log_mu)
    r = 2 * weight * (r - L)
    return npy.multiply(npy.sqrt(r), npy.sign(y - mu))


def get_residuals(
    umi,
    model_matrix,
    model_parameters_fit,
    residual_type="pearson",
    res_clip_range="default",
):

    """Get residuals for a fit model.

    Parameters
    ----------

    model_matrix: matrix
                  GLM model matrix
    model_parameters_fit: DataFrame
                          dataframe of model fit parameters
    res_clip_range: string or list
                    options: 1)"seurat": Clips residuals to -sqrt(ncells/30), sqrt(ncells/30)
                             2)"default": Clips residuals to -sqrt(ncells), sqrt(ncells)

    Returns
    -------
    residuals: matrix
               gene x cell matrix of residuals for genes in model_parameters_fit
    """

    subset = npy.asarray(
        model_parameters_fit[model_matrix.design_info.column_names].values
    )
    theta = npy.asarray(model_parameters_fit["theta"].values)

    mu = npy.exp(npy.dot(subset, model_matrix.T))
    # variance = mu + npy.divide(mu ** 2, theta.reshape(-1, 1))
    # pearson_residuals = npy.divide(umi - mu, npy.sqrt(variance))
    if residual_type == "pearson":
        residuals = pearson_residual(umi, mu, theta)
    elif residual_type == "deviance":
        residuals = deviance_residual(umi, mu, theta)
    if res_clip_range == "default":
        res_clip_range = npy.sqrt(umi.shape[1])
        residuals = npy.clip(residuals, a_min=-res_clip_range, a_max=res_clip_range)
    if res_clip_range == "seurat":
        res_clip_range = npy.sqrt(umi.shape[1] / 30)
        residuals = npy.clip(residuals, a_min=-res_clip_range, a_max=res_clip_range)
    return residuals


def correct(residuals, cell_attr, latent_var, model_parameters_fit, umi):
    # replace value of latent variables with its median
    cell_attr = cell_attr.copy()
    for column in latent_var:
        cell_attr.loc[:, column] = cell_attr.loc[:, column].median()
    model_matrix = dmatrix(" + ".join(latent_var), cell_attr)
    non_theta_columns = [
        x for x in model_matrix.design_info.column_names if x != "theta"
    ]
    coefficients = model_parameters_fit[non_theta_columns]
    theta = model_parameters_fit["theta"].values

    mu = npy.exp(coefficients.dot(model_matrix.T))
    mu = npy.exp(npy.dot(coefficients.values, model_matrix.T))
    variance = mu + npy.divide(mu ** 2, npy.tile(theta.reshape(-1, 1), mu.shape[1]))
    corrected_data = mu + residuals.values * npy.sqrt(variance)
    corrected_data[corrected_data < 0] = 0
    corrected_counts = sparse.csr_matrix(corrected_data.astype(int))

    return corrected_counts


def vst(
    umi,
    gene_names=None,
    cell_names=None,
    n_cells=5000,
    latent_var=["log10_umi"],
    batch_var=None,
    gmean_eps=1,
    min_cells=5,
    n_genes=2000,
    threads=4,
    method="theta_ml",
    theta_given=10,
    theta_regularization="od_factor",
    residual_type="pearson",
    correct_counts=False,
    exclude_poisson=False,
    fix_slope=False,
    verbosity=0,
):
    """Perform variance stabilizing transformation.

    Residuals are currently stored for all genes (might be memory intensive for larger datasets).

    Parameters
    ----------
    umi: matrix
         Sparse or dense matrix with genes as rows and cells as columns
         (same as Seurat)
    gene_names: list
                List of gene names for umi matrix
    cell_names: list
                List of cell names for umi matrix
    n_cells: int
             Number of cells to use for estimating parameters in Step1: default is 5000
    n_genes: int
             Number of genes to use for estimating parameters in Step1; default is 2000
    threads: int
             Number of threads to use (caveat: higher threads require higher memory)
    theta_given: int
                 Used only when method == "offset", for fixing the value of inverse overdispersion parameter
                 following Lause et al. (2021) offset model; default is 10
    theta_regularization: string
                         "od_factor" or "theta": parameter to run smoothing operation on for theta,
                         od_factor = 1 +mu/theta; default is od_factor

    residual_type: string
                  "pearson" or "deviance" residuals; default is "pearson"
    correct_counts: bool
                    Whether to correct counts by reversing the GLM with median values
    exclude_poisson: bool
                     To exclude poisson genes from regularization and set final parameters based on offset model; default is False
    fix_slope: bool
               Whether to fix the slope; default is False
    verbosity: bool
               Print verbose messages
    """
    umi = umi.copy()
    if n_cells is None:
        n_cells = umi.shape[1]
    if n_genes is None:
        n_genes = umi.shape[0]
    n_cells = min(n_cells, umi.shape[1])
    n_genes = min(n_genes, umi.shape[0])
    if gene_names is None:
        if not isinstance(umi, pd.DataFrame):
            raise RuntimeError(
                "`gene_names` and `cell_names` are required when umi is not a dataframe"
            )
        else:
            gene_names = umi.index.tolist()
            cell_names = umi.columns.tolist()
            umi = csr_matrix(umi.values)
            # umi.to_numpy()
    if cell_names is None:
        cell_names = [x for x in range(umi.shape[1])]

    gene_names = npy.asarray(gene_names, dtype="U")
    cell_names = npy.asarray(cell_names, dtype="U")
    genes_cell_count = npy.asarray((umi >= 0.01).sum(1))
    min_cells_genes_index = npy.squeeze(genes_cell_count >= min_cells)
    genes = gene_names[min_cells_genes_index]
    cell_attr = make_cell_attr(umi, cell_names)
    if isinstance(umi, pd.DataFrame):
        umi = umi.loc[genes]
    else:
        umi = umi[min_cells_genes_index, :]
    genes_log10_gmean = npy.log10(row_gmean_sparse(umi, gmean_eps=gmean_eps))
    genes_log10_amean = npy.log10(npy.ravel(umi.mean(1)))

    if n_cells is None and n_cells < umi.shape[1]:
        # downsample cells to speed up the first step
        npy.random.seed(1448145)
        cells_step1_index = npy.random.choice(
            a=npy.arange(len(cell_names), dtype=int), size=n_cells, replace=False
        )
        cells_step1 = cell_names[cells_step1_index]
        genes_cell_count_step1 = (umi[:, cells_step1_index] > 0).sum(1)
        genes_step1 = genes[genes_cell_count_step1 >= min_cells]
        genes_log10_gmean_step1 = npy.log10(
            row_gmean_sparse(
                umi[
                    genes_step1,
                ],
                gmean_eps=gmean_eps,
            )
        )
        genes_log10_amean_step1 = npy.log10(
            npy.ravel(
                umi[
                    genes_step1,
                ].mean(1)
            )
        )
        umi_step1 = umi[:, cells_step1_index]
    else:
        cells_step1_index = npy.arange(len(cell_names), dtype=int)
        cells_step1 = cell_names
        genes_step1 = genes
        genes_log10_gmean_step1 = genes_log10_gmean
        genes_log10_amean_step1 = genes_log10_amean
        umi_step1 = umi

    data_step1 = cell_attr.loc[cells_step1]
    if (n_genes is not None) and (n_genes < len(genes_step1)):
        # density-sample genes to speed up the first step
        sampling_prob = dds(genes_log10_gmean_step1)
        npy.random.seed(14)
        genes_step1_index = npy.random.choice(
            a=npy.arange(len(genes_step1)), size=n_genes, replace=False, p=sampling_prob
        )
        genes_step1 = gene_names[genes_step1_index]
        umi_step1 = umi_step1[genes_step1_index, :]  # [:, cells_step1_index]
        genes_log10_gmean_step1 = npy.log10(
            row_gmean_sparse(umi_step1, gmean_eps=gmean_eps)
        )
        genes_log10_amean_step1 = npy.log10(umi_step1.mean(1))

    if method == "offset":
        cells_step1_index = npy.arange(len(cell_names), dtype=int)
        cells_step1 = cell_names
        genes_step1 = genes
        genes_log10_gmean_step1 = genes_log10_gmean
        genes_log10_amean_step1 = genes_log10_amean
        umi_step1 = umi
    # Step 1: Estimate theta

    if verbosity:
        print("Running Step1")
    start = time.time()
    if batch_var is None:
        model_matrix = dmatrix(" + ".join(latent_var), data_step1)
    else:
        cross_term = "(" + " + ".join(latent_var) + "):" + batch_var
        model_matrix = dmatrix(
            " + ".join(latent_var) + cross_term + " + ".join(batch_var) + " + 0",
            data_step1,
        )

    if method == "offset":
        gene_mean = npy.ravel(umi.mean(1))
        mean_cell_sum = npy.mean(umi.sum(0))
        model_parameters = pd.DataFrame(index=genes)
        model_parameters["theta"] = theta_given
        model_parameters["Intercept"] = npy.log(gene_mean) - npy.log(mean_cell_sum)
        model_parameters["log10_umi"] = [npy.log(10)] * len(genes)
    elif method == "glmgp":
        model_parameters = get_model_params_allgene_glmgp(umi_step1, data_step1, threads=4)
        model_parameters.index = genes_step1
    elif method == "fix-slope":
        model_parameters = get_model_params_allgene_glmgp(
            umi_step1, data_step1, threads=4, use_offset=True
        )
        model_parameters.index = genes_step1
    elif method in ["theta_ml", "theta_lbfgs", "alpha_lbfgs"]:
        model_parameters = get_model_params_allgene(
            umi_step1, model_matrix, method, threads, fix_slope
        )
        model_parameters.index = genes_step1
    else:
        raise RuntimeError("Unknown method {}".format(method))
    gene_attr = pd.DataFrame(index=genes)
    gene_attr["gene_amean"] = npy.power(10, genes_log10_amean)
    gene_attr["gene_gmean"] = npy.power(10, genes_log10_gmean)
    gene_attr["gene_detectation_rate"] = (
        npy.squeeze(npy.asarray((umi > 0).sum(1))) / umi.shape[1]
    )
    gene_attr["theta"] = model_parameters["theta"]
    gene_attr["gene_variance"] = sparse_var(umi, 1)

    poisson_genes = None
    if exclude_poisson:
        poisson_genes1 = gene_attr.loc[
            gene_attr["gene_amean"] >= gene_attr["gene_variance"]
        ].index.tolist()
        poisson_genes2 = gene_attr.loc[gene_attr["gene_amean"] <= 1e-3].index.tolist()
        poisson_genes = set(poisson_genes1).union(poisson_genes2)

        poisson_genes_step1 = set(poisson_genes).intersection(genes_step1)

        if verbosity:
            print("Found ", len(poisson_genes1), " genes with var <= mean")
            print("Found ", len(poisson_genes2), " genes with mean < 1e-3")
            print("Found ", len(poisson_genes), " poisson genes")
            print("Setting there estimates to Inf")
        if poisson_genes_step1:
            model_parameters.loc[poisson_genes_step1, "theta"] = npy.inf

    end = time.time()
    step1_time = npy.ceil(end - start)
    if verbosity:
        print("Step1 done. Took {} seconds.".format(npy.ceil(end - start)))
    # Step 2: Do regularization

    if verbosity:
        print("Running Step2")
    start = time.time()
    genes_log10_gmean_step1_to_return = genes_log10_gmean_step1.copy()
    genes_log10_amean_step1_to_return = genes_log10_amean_step1.copy()
    outliers_df = pd.DataFrame(index=genes_step1)
    for col in model_parameters.columns:
        if method == "glmgp":
            col_outliers = is_outlier_r(
                model_parameters[col].values, genes_log10_gmean_step1
            )
        else:
            col_outliers = is_outlier(
                model_parameters[col].values, genes_log10_gmean_step1
            )
        outliers_df[col] = col_outliers

    if exclude_poisson:
        outliers_df.loc[poisson_genes_step1, "theta"] = True
    if theta_regularization == "theta":
        model_parameters["od_factor"] = npy.log10(model_parameters["theta"])
    else:
        model_parameters["od_factor"] = npy.log10(
            1 + npy.power(10, genes_log10_gmean_step1) / model_parameters["theta"]
        )

    model_parameters_to_return = model_parameters.copy()
    non_outliers = outliers_df.sum(1) == 0
    outliers = outliers_df.sum(1) > 0
    if verbosity:
        print("Total outliers: {}".format(npy.sum(outliers)))

    genes_non_outliers = genes_step1[non_outliers]
    genes_step1 = genes_step1[non_outliers]
    genes_log10_gmean_step1 = genes_log10_gmean_step1[non_outliers]
    if method == "offset":
        model_parameters_fit = model_parameters.copy()
    else:
        model_parameters = model_parameters.loc[genes_non_outliers]
        if exclude_poisson:
            non_poisson_genes = set(model_parameters.index.tolist()).difference(
                poisson_genes
            )
            model_parameters = model_parameters.loc[non_poisson_genes]
        model_parameters_fit = get_regularized_params(
            model_parameters,
            genes,
            genes_step1,
            genes_log10_gmean_step1,
            genes_log10_gmean,
            cell_attr,
            umi,
            theta_regularization=theta_regularization,
            exclude_poisson=exclude_poisson,
            poisson_genes=poisson_genes,
            method=method,
        )
    end = time.time()
    step2_time = npy.ceil(end - start)
    if verbosity:
        print("Step2 done. Took {} seconds.".format(npy.ceil(end - start)))

    # Step 3: Calculate residuals
    if verbosity:
        print("Running Step3")

    start = time.time()
    residuals = pd.DataFrame(
        get_residuals(umi, model_matrix, model_parameters_fit, residual_type)
    )
    residuals.index = genes
    residuals.columns = cell_names
    end = time.time()
    step3_time = npy.ceil(end - start)
    if verbosity:
        print("Step3 done. Took {} seconds.".format(npy.ceil(end - start)))

    gene_attr["theta_regularized"] = model_parameters_fit["theta"]
    gene_attr["residual_mean"] = residuals.mean(1)
    gene_attr["residual_variance"] = residuals.var(1)

    corrected_counts = None
    if correct_counts:
        corrected_counts = correct(
            residuals, cell_attr, latent_var, model_parameters_fit, umi
        )

    return {
        "residuals": residuals,
        "model_parameters": model_parameters_to_return,
        "model_parameters_fit": model_parameters_fit,
        "corrected_counts": corrected_counts,
        "genes_log10_gmean_step1": genes_log10_gmean_step1_to_return,
        "genes_log10_gmean": genes_log10_gmean,
        "genes_log10_amean_step1": genes_log10_amean_step1_to_return,
        "genes_log10_amean": genes_log10_amean,
        "cell_attr": cell_attr,
        "model_matrix": model_matrix,
        "gene_attr": gene_attr,
        "step1_time": step1_time,
        "step2_time": step2_time,
        "step3_time": step3_time,
        "total_cells": len(cell_names),
    }


def get_hvg_residuals(vst_out, var_features_n=3000, res_clip_range="seurat"):
    """Get residuals for highly variable genes (hvg)
    Get residuals for n highly variable genes (sorted by decreasing residual variance)

    Parameters
    -----------

    vst_out: dict
             output of vst()
    res_clip_range: string or list
                    options: 1)"seurat": Clips residuals to -sqrt(ncells/30), sqrt(ncells/30)
                             2)"default": Clips residuals to -sqrt(ncells), sqrt(ncells)
    var_features_n: int
                    Number of variable features to select (for calculating a subset of pearson residuals)

    Returns
    -------
    hvg_residuals: matrix
                   A gene x cell matrix of hvg residuals

    """

    gene_attr = vst_out["gene_attr"]
    total_cells = vst_out["total_cells"]
    gene_attr = gene_attr.sort_values(by=["residual_variance"], ascending=False)
    highly_variable = gene_attr.index[:var_features_n].tolist()
    if res_clip_range == "seurat":
        clip_range = [-npy.sqrt(total_cells / 30), npy.sqrt(total_cells / 30)]
    elif res_clip_range == "default":
        clip_range = [-npy.sqrt(total_cells), npy.sqrt(total_cells)]
    else:
        if not isinstance(res_clip_range, list):
            raise RuntimeError("res_clip_range should be a list or string")
        clip_range = res_clip_range
    hvg_residuals = vst_out["residuals"].T[highly_variable]
    hvg_residuals = npy.clip(hvg_residuals, clip_range[0], clip_range[1])
    return hvg_residuals


def SCTransform(
    data,
    vst_flavor=None,
    method="theta_ml",
    n_cells=5000,
    n_genes=2000,
    res_clip_range="seurat",
    var_features_n=3000,
    **kwargs
):
    """Wrapper around vst

    Parameters
    ----------
    data: StereoExpData object

    vst_flavor: string
                if set to 'v2' fixes slope and excludes non-poisson genes
                Requires rpy2 and glmGamPoi to be installed. This will
                automatically set method='fix-slope'

    method: method

    n_cells: int
             Number of cells to use for estimating parameters in Step1: default is 5000
    n_genes: int
             Number of genes to use for estimating parameters in Step1; default is 2000
    res_clip_range: string or list
                    options: 1)"seurat": Clips residuals to -sqrt(ncells/30), sqrt(ncells/30)
                             2)"default": Clips residuals to -sqrt(ncells), sqrt(ncells)
    var_features_n: int
                    Number of variable features to select (for calculating a subset of pearson residuals)


    """
    # adata = adata.copy()
    exclude_poisson = False
    # method = "theta_ml"
    if vst_flavor == "v2":
        method = "fix-slope"
        exclude_poisson = True
        n_cells = 2000
    vst_out = vst(
        data.exp_matrix.T,
        gene_names=data.gene_names.tolist(),
        cell_names=data.cell_names.tolist(),
        method=method,
        n_cells=n_cells,
        n_genes=n_genes,
        exclude_poisson=exclude_poisson,
    )
    residuals = get_hvg_residuals(vst_out, var_features_n, res_clip_range)
    return residuals
