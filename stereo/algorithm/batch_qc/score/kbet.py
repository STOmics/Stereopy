#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:20
# @Author  : zhangchao
# @File    : kbet.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psutil
from anndata import AnnData
from scipy.stats import chi2

from .get_neighbors import get_neighbors


def one_sample_kbet(neighbor_indices, attr_values, ideal_distribution, n_neighbors):
    dof = ideal_distribution.size - 1
    # Frequency of observation in local region
    observed_counts = pd.Series(attr_values[neighbor_indices]).value_counts(sort=False).values
    expected_counts = ideal_distribution * n_neighbors
    # calculate chi-2 test
    stat = np.sum(np.divide(np.square(np.subtract(observed_counts, expected_counts)), expected_counts))
    p_value = 1 - chi2.cdf(stat, dof)
    return [stat, p_value]


def get_kbet(
        data: AnnData,
        key: str = "batch",
        use_rep: str = "X_umap",
        alpha: float = 0.05,
        n_neighbors: int = 30):
    """ Calculate the K-nearest neighbors Batch Effects Test (K-BET) metric of the data regarding a specific sample attribute and embedding. # noqa
    The K-BET metric measures if cells from different samples mix well in their local neighborhood.

    Parameters
    _______________________
    data: ``AnnData``
        Data matrix with rows for cells and columns for genes.
    key: ``str``
        The sample attribute to be consider. Must exist in ``data.obs``.
    use_rep: ``str``
         The embedding representation to be used. The key must be exist in ``data.obsm``. By default, use UMAP coordinates. # noqa
    n_neighbors: ``int``
        Number of nearest neighbors.
    alpha: ``float``
        Threshold. A cell is accepted is its K-BET p-value is greater than or equal to ``alpha``

    Returns
    -----------------------
    stat_mean: ``float``
        Mean K-BET chi-square statistic over all cells.
    pvalue_mean: ``float``
        Mean K-BET  p-value over all cells.
    accept_rate: ``float``
        K-BET acceptance rate of the sample.
    """
    assert key in data.obs_keys()
    if data.obs[key].dtype.name == "category":
        data.obs[key] = data.obs[key].astype("category")

    ideal_distribution = data.obs[key].value_counts(normalize=True, sort=False)
    n_sample = data.shape[0]
    get_neighbors(data, n_neighbors=n_neighbors, use_rep=use_rep)

    # add itself into the knn connectivity graph
    assert f"{use_rep}_knn_connectivity" in data.obsm_keys(), \
        f"Error, can not found '{use_rep}_knn_connectivity' in " \
        f".obsm_keys(). Please calculate nearest neighbors graph first."
    indices = np.concatenate((np.arange(n_sample).reshape(-1, 1), data.obsm[f"{use_rep}_knn_connectivity"][:, :-1]),
                             axis=1)
    partial_kbet = partial(
        one_sample_kbet,
        attr_values=data.obs[key].values.copy(),
        ideal_distribution=ideal_distribution,
        n_neighbors=n_neighbors)

    n_jobs = psutil.cpu_count(logical=False)
    if n_jobs is None:
        n_jobs = psutil.cpu_count(logical=True)

    with Pool(n_jobs) as p:
        results = p.map(partial_kbet, indices)

    results = np.array(results)
    stat_mean = results[:, 0].mean()
    pvalue_mean = results[:, 1].mean()
    accept_rate = (results[:, 1] >= alpha).sum() / n_sample

    reject_score = np.mean(results[:, 1] < alpha)

    return reject_score, stat_mean, pvalue_mean, accept_rate
