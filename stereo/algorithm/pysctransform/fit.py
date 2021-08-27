import numbers
import sys

import numpy as npy
import statsmodels
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import digamma
from scipy.special import gammaln
from scipy.special import polygamma
from statsmodels.api import GLM


def trigamma(x):
    """Trigamma function.

    Parameters
    ----------
    x: float
    """
    return polygamma(1, x)


def _process_y(y):
    y = npy.asarray(y, dtype=int)
    y = npy.squeeze(y)
    return y


def lookup_table(y):
    y = npy.squeeze(y)
    y_bc = npy.bincount(y)
    y_i = npy.nonzero(y_bc)[0]
    y_lookup = npy.vstack((y_i, y_bc[y_i])).T
    return y_lookup


def theta_nb_score(y, mu, theta, fast=True):

    y_lookup = None
    N = len(y)
    if fast:
        # create a lookup table for y
        # Inspired from glmGamPoi, Ahlmann-Eltze and Huber (2020)
        # y_lookup = npy.asarray(lookup_table(y))
        y_lookup = lookup_table(y)
        y_sum = npy.dot(y_lookup[:, 0], y_lookup[:, 1])
        digamma_sum = npy.dot(digamma(y_lookup[:, 0] + theta), y_lookup[:, 1])
        digamma_theta = digamma(theta) * N
        mu_term = (npy.log(theta) - npy.log(mu + theta) + 1) * N
        y_term = (
            1 / (mu + theta) * (y_sum + N * theta)
        )  #  #sum((y + theta) / (mu + theta))

        lld = digamma_sum - digamma_theta - y_term + mu_term
        return lld
    else:
        digamma_sum = digamma(y + theta)
        digamma_theta = digamma(theta)
        mu_term = npy.log(theta) - npy.log(mu + theta) + 1
        y_term = (y + theta) / (mu + theta)

        lld = digamma_sum - digamma_theta - y_term + mu_term
        return npy.sum(lld)


def theta_nb_hessian(y, mu, theta, fast=True):
    y_lookup = None
    N = len(y)
    if fast:
        # create a lookup table for y
        # Inspired from glmGamPoi, Ahlmann-Eltze and Huber (2020)
        y_lookup = lookup_table(y)
        # y_lookup = npy.asarray(lookup_table(y))
        y_sum = npy.dot(y_lookup[:, 0], y_lookup[:, 1])
        trigamma_sum = npy.dot(trigamma(y_lookup[:, 0] + theta), y_lookup[:, 1])
        trigamma_theta = trigamma(theta) * N
        mu_term = (1 / theta - 2 / (mu + theta)) * N
        y_term = (y_sum + N * theta) / (mu + theta) ** 2
        # y_term = ((y + theta) / (mu + theta) ** 2).sum()
        lldd = trigamma_sum - trigamma_theta + y_term + mu_term
        return lldd
    else:
        trigamma_sum = trigamma(y + theta)
        trigamma_theta = trigamma(theta)

        mu_term = 1 / theta - 2 / (mu + theta)
        y_term = (y + theta) / (mu + theta) ** 2
        lldd = trigamma_sum - trigamma_theta + y_term + mu_term
        return npy.sum(lldd)


def estimate_mu_glm(y, model_matrix):
    y = _process_y(y)
    y = npy.asarray(y)
    model = sm.GLM(y, model_matrix, family=sm.families.Poisson())
    fit = model.fit()
    mu = fit.predict()
    return {"coef": fit.params, "mu": mu[0]}


def estimate_mu_poisson(y, model_matrix):
    y = _process_y(y)
    y = npy.asarray(y)
    model = statsmodels.discrete.discrete_model.Poisson(y, model_matrix)
    fit = model.fit(disp=False)
    mu = fit.predict()
    return {"coef": fit.params, "mu": mu[0]}


def theta_ml(y, mu, max_iters=20, tol=1e-4):
    y = _process_y(y)
    mu = npy.squeeze(mu)

    N = len(y)
    theta = N / sum((y / mu - 1) ** 2)
    for i in range(max_iters):
        theta = abs(theta)

        score_diff = theta_nb_score(y, mu, theta)
        # if first diff is negative, there is no maximum
        if score_diff < 0:
            return npy.inf
        delta_theta = score_diff / theta_nb_hessian(y, mu, theta)
        theta = theta - delta_theta

        if npy.abs(delta_theta) <= tol:
            return theta

    if theta < 0:
        theta = npy.inf

    return theta


def alpha_lbfgs(y, mu, maxoverdispersion=1e5):
    y = _process_y(y)
    mu = npy.squeeze(mu)
    N = len(y)
    ysum = npy.sum(y)

    def nll(alpha):
        return (
            -npy.sum(gammaln(1 / alpha + y))
            + N * gammaln(1 / alpha)
            - ysum * npy.log(mu / (1 / alpha + mu))
            - N * 1 / alpha * npy.log(1 / alpha / (1 / alpha + mu))
        )

    init_alpha = (npy.var(y) - mu) / (mu ** 2)
    if init_alpha <= 0:
        return npy.inf
    alpha = minimize(
        nll, init_alpha, bounds=[(0, maxoverdispersion)], method="L-BFGS-B"
    )
    return 1 / alpha.x[0]


def theta_lbfgs(y, mu, maxoverdispersion=1e5):
    y = _process_y(y)
    mu = npy.squeeze(mu)
    N = len(y)
    ysum = npy.sum(y)

    def nll(theta):
        return (
            -npy.sum(gammaln(theta + y))
            + N * gammaln(theta)
            - ysum * npy.log(mu / (theta + mu))
            - N * theta * npy.log(theta / (theta + mu))
        )

    init_theta = (mu ** 2) / (npy.var(y) - mu)
    if init_theta <= 0:
        return npy.inf
    theta = minimize(
        nll, init_theta, bounds=[(1 / maxoverdispersion, None)], method="L-BFGS-B"
    )
    return theta.x[0]
