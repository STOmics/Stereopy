#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:54
# @Author  : zhangchao
# @File    : pca_regression.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import numpy as np
from anndata import AnnData
from sklearn.linear_model import LinearRegression

from ..utils import pca_lowrank


def pca_regression(merge_data: AnnData, n_pcs=50, batch_key="batch", embed_key=None):
    """Principal component regression

    Compute the overall variance contribution given a covariate according to the following formula:

    .. math::
        Var(C|B) = \\sum^G_{i=1} Var(C|PC_i) \\cdot R^2(PC_i|B)

    for :math:`G` principal components (:math:`PC_i`), where :math:`Var(C|PC_i)` is the variance of the data matrix
    :math:`C` explained by the i-th principal component, and :math:`R^2(PC_i|B)` is the :math:`R^2` of the i-th
    principal component regressed against a covariate :math:`B`.
    """
    if embed_key is None and "pca" not in merge_data.uns_keys():
        pca_lowrank(merge_data, use_rep=None, n_component=n_pcs)
    elif embed_key is not None:
        pca_lowrank(merge_data, use_rep=embed_key, n_component=n_pcs)
    assert "pca" in merge_data.uns_keys()
    pc_var = merge_data.uns["pca"]["variance"]

    covariate = merge_data.obs[batch_key].cat.codes.values.reshape(-1, 1)
    x_pca = merge_data.obsm["X_pca"]
    r2 = []
    for i in range(n_pcs):
        pc = x_pca[:, [i]]
        lm = LinearRegression()
        lm.fit(covariate, pc)
        r2.append(np.maximum(0, lm.score(covariate, pc)))
    var = pc_var / sum(pc_var) * 100
    r2var = sum(r2 * var) / 100
    return r2var
