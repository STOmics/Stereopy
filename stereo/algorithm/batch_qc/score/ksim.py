#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:24
# @Author  : zhangchao
# @File    : ksim.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import numpy as np
from anndata import AnnData

from .get_neighbors import get_neighbors


def get_ksim(
        data: AnnData,
        key: str = "cell type",
        use_rep: str = "X_umap",
        beta: float = 0.9,
        n_neighbors: int = 30):
    """"Calculate the kSIM acceptance rate metric of the data regarding a specific sample attribute and embedding.
    The kSIM acceptance rate requires ground truth cell type information and measures whether the neighbors of a cell have the same cell type as it does. # noqa
    If a method overcorrects the batch effects, it will have a low kSIM acceptance rate.

    Parameters
    -------------------------
    data: ``AnnData``
        Data matrix with rows for cells and columns for genes.
    key: ``str``
        The sample attribute to be consider. Must exist in ``data.obs``.
    use_rep: ``str``
         The embedding representation to be used. The key must be exist in ``data.obsm``. By default, use UMAP coordinates. # noqa
    beta: ``float``
        Acceptance rate threshold. A cell  is accepted is its kMIS rate is larger than or equal to `bata`.
    n_neighbors: ``int``
        Number of nearest neighbors.

    Returns
    -------------------------
    ksim_mean: ``float``
        Mean of calculated ksim score
    ksim_accept_rate: ``float``
        the ksim acceptance rate
    """
    assert key in data.obs_keys()
    n_sample = data.shape[0]
    get_neighbors(data, n_neighbors=n_neighbors, use_rep=use_rep)

    # add itself into the knn connectivity graph
    assert f"{use_rep}_knn_connectivity" in data.obsm_keys(), \
        f"Error, can not found '{use_rep}_knn_connectivity' " \
        f"in .obsm_keys(). Please calculate nearest neighbors graph first."
    indices = np.concatenate((np.arange(n_sample).reshape(-1, 1), data.obsm[f"{use_rep}_knn_connectivity"][:, :-1]),
                             axis=1)
    labels = data.obs[key].values[indices.flatten()].reshape(-1, 1)
    same_label = labels == labels[:, 0].reshape(-1, 1)
    correct_rates = same_label.sum(axis=1) / n_neighbors

    ksim_mean = correct_rates.mean()
    ksim_accept_rate = (correct_rates >= beta).sum() / n_sample

    return ksim_mean, ksim_accept_rate
