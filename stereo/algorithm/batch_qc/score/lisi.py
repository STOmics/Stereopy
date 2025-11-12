#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:22
# @Author  : zhangchao
# @File    : lisi.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psutil
from anndata import AnnData

from .get_neighbors import get_neighbors


def one_sample_lisi(neighbors_indices, attr_values):
    probability = pd.Series(attr_values[neighbors_indices]).value_counts(normalize=True, sort=False).values
    score = 1 / (probability ** 2).sum()
    return score


def get_lisi(
        data: AnnData,
        key: str = "batch",
        use_rep: str = "X_umap",
        n_neighbors: int = 30):
    """"Calculate the Local inverse Simpson's Index (LISI) metric of the data regarding a specific sample attribute and embedding. # noqa
    The LISI metric measures if cells from different samples mix well in their local neighborhood.

    Parameters
    --------------------------------
    data: ``AnnData``
        Data matrix with rows for cells and columns for genes.
    key: ``str``
        The sample attribute to be consider. Must exist in ``data.obs``.
    use_rep: ``str``
         The embedding representation to be used. The key must be exist in ``data.obsm``. By default, use UMAP coordinates. # noqa
    n_neighbors: ``int``
        Number of nearest neighbors.

    Returns
    --------------------------------
    lisi_mean: ``float``
        Mean of calculated score.
    lower: ``float``
        Lower bound of 95% confidence interval.
    upper: ``float``
        Upper bound of 95% confidence interval.
    """
    assert key in data.obs_keys()

    n_sample = data.shape[0]
    get_neighbors(data, n_neighbors=n_neighbors, use_rep=use_rep)

    # add itself into the knn connectivity graph
    assert f"{use_rep}_knn_connectivity" in data.obsm_keys(), \
        f"Error, can not found '{use_rep}_knn_connectivity' in .obsm_keys()." \
        f" Please calculate nearest neighbors graph first."
    indices = np.concatenate((np.arange(n_sample).reshape(-1, 1), data.obsm[f"{use_rep}_knn_connectivity"][:, :-1]),
                             axis=1)

    partial_lisi = partial(
        one_sample_lisi,
        attr_values=data.obs[key].values.copy())

    n_jobs = psutil.cpu_count(logical=False)
    if n_jobs is None:
        n_jobs = psutil.cpu_count(logical=True)

    with Pool(n_jobs) as p:
        results = p.map(partial_lisi, indices)
    results = np.array(results)
    lisi_mean = results.mean()

    std = results.std()
    lower = lisi_mean - 1.96 * std / np.sqrt(n_sample)
    upper = lisi_mean + 1.96 * std / np.sqrt(n_sample)
    return lisi_mean, lower, upper
