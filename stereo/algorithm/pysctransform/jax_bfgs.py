import jax
import jax.numpy as jnp
import numpy as onp
from jax import grad
from jax import jit
from jax import vmap
from jax.config import config

config.update("jax_enable_x64", True)


import pandas as pd
from jax.scipy.optimize import minimize as jax_minimize
from jax.scipy.special import digamma as jax_digamma
from jax.scipy.special import gammaln as jax_gammaln
from jax.scipy.special import polygamma as jax_polygamma


@jax.jit
def fit_nbinom_bfgs_jit(y, mu):
    # theta is the overdispersion parameter. scuh that variance = mu + mu**2/theta
    N = len(y)
    ysum = jnp.sum(y)
    ybar = ysum / N
    thetamin = 1e-4

    def nll(theta):
        theta = theta[0]
        return (
            -jnp.sum(jax_gammaln(theta + y))
            + N * jax_gammaln(theta)
            - ysum * jnp.log(mu / (theta + mu))
            - N * theta * jnp.log(theta / (theta + mu))
        )

    init_theta = jnp.array([(ybar ** 2) / (jnp.var(y) - ybar)])
    theta = jax_minimize(
        nll, init_theta, method="BFGS", tol=1e-4, options={"maxiter": 100}
    )
    return theta.x[0]


@jax.jit
def fit_nbinom_bfgs_alpha_jit(y, mu):
    # theta is the overdispersion parameter. scuh that variance = mu + mu**2/theta
    N = len(y)
    ysum = jnp.sum(y)
    ybar = ysum / N
    thetamin = 1e-4

    def nll(alpha):
        alpha = alpha[0]
        return (
            -jnp.sum(jax_gammaln(1 / alpha + y))
            + N * jax_gammaln(1 / alpha)
            - ysum * jnp.log(mu / (1 / alpha + mu))
            - N * 1 / alpha * jnp.log(1 / (1 + alpha * mu))
        )

    init_alpha = jnp.array([(jnp.var(y) - ybar) / (ybar ** 2)])
    theta = jax_minimize(
        nll, init_alpha, method="BFGS", tol=1e-4, options={"maxiter": 100}
    )
    return 1 / theta.x[0]


# -*- coding: utf-8 -*-
