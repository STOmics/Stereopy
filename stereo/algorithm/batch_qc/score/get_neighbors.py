#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:21
# @Author  : zhangchao
# @File    : get_neighbors.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import psutil
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def get_neighbors(
        data: AnnData,
        n_neighbors: int = 30,
        use_rep: str = "X_umap"):
    """"Find K nearest neighbors for each data and return the indices and distance arrays.

    Parameters
    --------------------------
    data: ``AnnData``
        An AnnData object.
    n_neighbors: ``int``
        Number of neighbors, NOT including the data itself. By default, 30
    use_rep: ``str``
        Representation to used calculated kNN, which must exist in ``data.obsm``. If `None` use data.X, by default, 'X_umap' # noqa
    """
    if use_rep is None:
        embed_x = data.X if not sp.issparse(data.X) else data.X.toarray()
    else:
        assert use_rep in data.obsm_keys()
        embed_x = data.obsm[use_rep]

    indices_key = f"{use_rep}_knn_connectivity"
    distances_key = f"{use_rep}_knn_distances"

    n_jobs = psutil.cpu_count(logical=False)
    if n_jobs is None:
        n_jobs = psutil.cpu_count(logical=True)

    knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    knn.fit(embed_x)
    distances, indices = knn.kneighbors()
    data.obsm[indices_key] = indices
    data.obsm[distances_key] = distances
