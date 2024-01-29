import numba
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from joblib import (
    Parallel,
    delayed,
    cpu_count
)
from patsy.highlevel import dmatrix
from scipy import interpolate
from scipy.special import digamma
from scipy.special import polygamma

from .bw import bwSJ


def multi_pearson_residual(i, model_pars_final, regressor_data_final, umi, residual_type, min_variance, genes, bin_ind):
    min_var = min_variance
    genes_bin = genes[bin_ind == i]
    mu = np.exp(np.dot(model_pars_final.loc[genes_bin, ['Intercept', 'log_umi']], regressor_data_final.T))
    genes_bin_bool_list = np.isin(genes, genes_bin)
    y = umi[genes_bin_bool_list, :]
    if residual_type == "pearson":
        return pearson_residual(y, mu, model_pars_final.loc[genes_bin, "theta"], min_var=min_var)
    elif residual_type == "deviance":
        raise NotImplementedError
    else:
        raise Exception


@numba.jit(cache=True, forceobj=True, nogil=True)
def pearson_residual(y, mu, theta, min_var=-np.inf):
    variance = mu + np.divide(mu ** 2, theta.to_numpy().reshape(-1, 1))
    variance[variance < min_var] = min_var
    pearson_residuals = np.divide(y - mu, np.sqrt(variance))
    return pearson_residuals


def dds(genes_log10_gmean_step1, grid_points=2 ** 10):
    x, y = (
        FFTKDE(kernel="gaussian", bw="silverman").fit(np.asarray(genes_log10_gmean_step1)).evaluate(
            grid_points=grid_points)
    )
    density = interpolate.interp1d(x=x, y=y, assume_sorted=False)
    sampling_prob = 1 / (density(genes_log10_gmean_step1) + np.finfo(float).eps)
    return sampling_prob / sampling_prob.sum()


def row_gmean_sparse(umi, gmean_eps=1):
    return np.squeeze(np.asarray(np.array([row_gmean(x.todense(), gmean_eps)[0] for x in umi])))


def row_gmean(umi, gmean_eps=1):
    return np.exp(np.log(umi + gmean_eps).mean(1)) - gmean_eps


def make_cell_attr(umi, cells, latent_var, batch_var, latent_var_nonreg) -> pd.DataFrame:
    # TODO: only `log_umi` will be made, will complete in the future
    tmp_dict_cell_attr = {"umi": umi.sum(0).tolist()[0]}
    tmp_dict_cell_attr["log_umi"] = np.log10(tmp_dict_cell_attr['umi'])
    return pd.DataFrame(tmp_dict_cell_attr, index=cells)


def fit_poisson(umi, model_str, data, theta_estimation_fun="theta.ml") -> pd.DataFrame:
    # TODO: ignore `theta_estimation_fun`
    regressor_data = dmatrix("~log_umi", data, return_type='dataframe')
    results = Parallel(n_jobs=cpu_count(), backend="threading")(
        delayed(one_row_fit_poission)(regressor_data, y.toarray()[0], theta_estimation_fun)
        for y in umi
    )
    return pd.DataFrame(results, columns=["theta", "Intercept", "log_umi"])


@numba.jit(cache=True, forceobj=True, nogil=True)
def one_row_fit_poission(regressor_data, y, theta_estimation_fun='theta.ml'):
    fit = qpois_reg(regressor_data.to_numpy(), y, 1e-9, 100, 1.0001, True)
    if theta_estimation_fun == "theta.ml":
        theta = theta_ml(y=y, mu=fit['fitted'])
    elif theta_estimation_fun == "theta.mm":
        # TODO: `theta.mm` not yet finished
        raise NotImplementedError
    else:
        raise Exception
    return theta, fit['coefficients'][0], fit['coefficients'][1]


@numba.jit(cache=True, forceobj=True, nogil=True)
def qpois_reg(X, Y, tol, maxiters, minphi, returnfit):
    n, pcols = X.shape
    d = pcols

    b_old = np.zeros(shape=(d, 1), dtype=np.double)
    b_new = np.zeros(shape=(d, 1), dtype=np.double)

    y = Y.reshape((Y.shape[0], 1))
    m = np.zeros(shape=(n, 1), dtype=np.double)
    phi = np.zeros(shape=(n, 1), dtype=np.double)

    x = X

    i = 0
    while i < pcols:
        unique_vals = np.unique(x[0:, i])
        if unique_vals.shape[0] == 1:
            b_old[i] = np.log(np.mean(y))
            break
        if unique_vals.shape[0] == 2 and (unique_vals[0] == 0 or unique_vals[1] == 0):
            b_old[i] = np.matmul(y.T, x[0:, i]).item()
            b_old[i] = b_old[i] / np.sum(x[0:, i])
            b_old[i] = np.log(max(1e-9, b_old[i]))
        i += 1

    x_tr = x.T
    ij = 2
    dif = 1.0

    while dif > tol:
        yhat = np.matmul(x, b_old)
        # TODO temporarily value
        yhat = np.clip(yhat, -708, 709)

        m = np.exp(yhat)
        phi = y - m
        L1 = np.matmul(x_tr, phi)
        L2 = []
        for i in range(x.shape[1]):
            L2.append(np.multiply(np.asmatrix(x[0:, i]).T, m))
        L2 = np.array(L2).T
        L2 = np.matmul(x_tr, L2)

        b_new = b_old + np.matmul(np.linalg.inv(L2), L1)[0]
        dif = np.sum(np.abs(b_new - b_old))
        b_old = b_new
        ij += 1
        if ij == maxiters:
            break

    # TODO temporarily value
    phi[(-1.64487933e-154 < phi) & (phi < 1.64487933e-154)] = 0

    p = np.sum(np.square(phi) / m) / (n - pcols)
    return {
        "coefficients": b_new.T[0],
        "phi": p,
        "theta.guesstimate": (np.mean(m) / (max(p, minphi) - 1)),
        "fitted": m.T[0] if returnfit else None
    }


# @numba.jit(cache=True, forceobj=True, nogil=True)
def theta_ml(y, mu, limit=10, eps=0.0001220703):
    weights = np.ones(len(y))
    n = np.sum(weights)
    t0 = n / np.sum(np.power(weights * (y / mu - 1), 2))
    it = 1
    _del = 1
    while it < limit and np.abs(_del) > eps:
        t0 = np.abs(t0)
        i = info(n, t0, mu, y, weights)
        _del = score(n, t0, mu, y, weights) / i
        t0 = t0 + _del
        it += 1
    if t0 < 0:
        t0 = 0
    return t0


@numba.jit(cache=True, forceobj=True, nogil=True)
def score(n, th, mu, y, w):
    a = th + y
    b = th + mu
    return np.sum(w * (digamma(a) - digamma(th) + np.log(th) + 1 - np.log(b) - a / b))


@numba.jit(cache=True, forceobj=True, nogil=True)
def trigamma(x):
    return polygamma(1, x)


@numba.jit(cache=True, forceobj=True, nogil=True)
def info(n, th, mu, y, w):
    a = th + y
    b = th + mu
    return np.sum(w * (-trigamma(a) + trigamma(th) - 1 / th + 2 / b - a / np.power(b, 2)))


def is_outlier(y, x, th=10, eps=2.220446e-16 * 10):
    x_max, x_min = x.max(), x.min()
    bin_width = (x_max - x_min) * bwSJ(x) / 2
    breaks1 = seq(x_min - eps, x_max + bin_width, bin_width)
    breaks2 = seq(x_min - eps - bin_width / 2, x_max + bin_width, bin_width)
    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)
    return (np.where(abs(score1) < abs(score2), abs(score1), abs(score2)) > th).T[0]


def robust_scale_binned(
        y: pd.DataFrame,
        x,
        breaks
):
    bins = pd.cut(x, bins=breaks, ordered=True)
    y = y.astype(object)
    # TODO: https://pandas.pydata.org/pandas-docs/version/1.5.3/whatsnew/v1.5.1.html
    # changes in `pandas-1.5.1` shows return df whose length more than the length of `bins`
    tmp = y.groupby(bins, observed=True).agg(robust_scale).dropna().explode().to_list()
    o = np.array(sorted(range(0, len(bins)), key=lambda x1: bins[x1]))
    sorted_score = pd.DataFrame(np.zeros(shape=(len(x), 1))).iloc[o,]
    sorted_score[0] = tmp
    return sorted_score.sort_index().values


def mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826


def robust_scale(x):
    a = x.values.reshape((x.shape[0],))
    b = np.median(x)
    c = mad(x)
    return (a - b) / (c + 2.220446e-16)


def seq(start, stop, step=1):
    return np.arange(start, stop, step)
