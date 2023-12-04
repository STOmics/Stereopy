#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:45
# @Author  : zhangchao
# @File    : pca_lowrank.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import scipy.sparse as sp
import torch
from anndata import AnnData


def pca_lowrank(merge_data: AnnData, use_rep: str = None, n_component: int = 50):
    """Calculate Fast PCA
    Parameters
    --------------------
    merge_data: ``AnnData``
    use_rep: ``str``
    n_component: ``int``
        The number of principal components retained.

    Returns
    --------------------
    """
    merge_data.uns["pca"] = {}
    if use_rep is None:
        x_tensor = torch.tensor(merge_data.X.toarray()) if sp.issparse(merge_data.X) else torch.tensor(merge_data.X)
    else:
        assert use_rep in merge_data.obsm_keys()
        x_tensor = torch.tensor(merge_data.obsm[use_rep])
    u, s, v = torch.pca_lowrank(x_tensor, q=n_component)

    explained_variance_ = s.pow(2) / (merge_data.shape[0] - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var
    merge_data.obsm["X_pca"] = torch.matmul(x_tensor, v).numpy()
    merge_data.uns["pca"]["variance"] = explained_variance_.numpy()
    merge_data.uns["pca"]['variance_ratio'] = explained_variance_ratio_.numpy()
