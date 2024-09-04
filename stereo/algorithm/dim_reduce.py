#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
"""

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import svd_flip

from stereo.log_manager import logger
from .scale import _get_mean_var


def low_variance(x, threshold=0.01):
    """
    filter the features which have low variance between the samples.

    :param x: 2D array, shape (M, N)
    :param threshold: the min threshold of variance.
    :return: a new array which filtered the feature with low variance.
    """
    x_var = np.var(x, axis=0)
    var_index = np.where(x_var > threshold)[0]
    x = x[:, var_index]
    return x


def factor_analysis(x, n_pcs):
    """
    the dim reduce function of factor analysis

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :return:  ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
    """
    fa = FactorAnalysis(n_components=n_pcs)
    fa.fit(x)
    tran_x = fa.transform(x)
    return tran_x


def pca(x, n_pcs, svd_solver='auto', random_state=0, dtype='float32'):
    """
    Principal component analysis.

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :param svd_solver: {'auto', 'full', 'arpack', 'randomized'}, default to 'auto'
                If auto :
                    The solver is selected by a default policy based on `X.shape` and
                    `n_pcs`: if the input data is larger than 500x500 and the
                    number of components to extract is lower than 80% of the smallest
                    dimension of the data, then the more efficient 'randomized'
                    method is enabled. Otherwise the exact full SVD is computed and
                    optionally truncated afterwards.
                If full :
                    run exact full SVD calling the standard LAPACK solver via
                    `scipy.linalg.svd` and select the components by postprocessing
                If arpack :
                    run SVD truncated to n_pcs calling ARPACK solver via
                    `scipy.sparse.linalg.svds`. It requires strictly
                    0 < n_pcs < min(x.shape)
                If randomized :
                    run randomized SVD by the method of Halko et al.
    :param random_state : int, RandomState instance
    :return:  ndarray of shape (n_samples, n_pcs) Embedding of the training data in low-dimensional space.
    """
    if issparse(x):
        if svd_solver != 'arpack':
            logger.warning(
                f'svd_solver: {svd_solver} can not be used with sparse input.\n'
                'Use "arpack" (the default) instead.'
            )
            svd_solver = 'arpack'
        if x.dtype.char not in "fFdD":
            x = x.astype(dtype)
            logger.info(f'exp_matrix dType is not float, it is changed to {dtype}')
        output = _pca_with_sparse(x, n_pcs, solver=svd_solver, random_state=random_state)
        result = dict(
            [('x_pca', output['X_pca']), ('variance', output['variance']), ('variance_ratio', output['variance_ratio']),
             ('pcs', output['components'].T)])
    else:
        pca_obj = PCA(n_components=n_pcs, svd_solver=svd_solver, random_state=random_state)
        x_pca = pca_obj.fit_transform(x)
        variance = pca_obj.explained_variance_
        variance_ratio = pca_obj.explained_variance_ratio_
        pcs = pca_obj.components_.T
        result = dict([('x_pca', x_pca), ('variance', variance), ('variance_ratio', variance_ratio), ('pcs', pcs)])
    
    if result['x_pca'].dtype.descr != np.dtype(dtype).descr:
        logger.info(f'x_pca dType is changed from {result["x_pca"].dtype} to {dtype}')
        result['x_pca'] = result['x_pca'].astype(dtype)

    return result


def _pca_with_sparse(X, n_pcs, solver='arpack', mu=None, random_state=None):
    random_state = check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))
    X = check_array(X, accept_sparse=['csr', 'csc'])

    if mu is None:
        mu = X.mean(0).A.flatten()[None, :]
    mdot = mu.dot
    mmat = mdot
    mhdot = mu.T.dot
    mhmat = mu.T.dot
    Xdot = X.dot
    Xmat = Xdot
    XHdot = X.T.conj().dot
    XHmat = XHdot
    ones = np.ones(X.shape[0])[None, :].dot

    def matvec(x):
        return Xdot(x) - mdot(x)

    def matmat(x):
        return Xmat(x) - mmat(x)

    def rmatvec(x):
        return XHdot(x) - mhdot(ones(x))

    def rmatmat(x):
        return XHmat(x) - mhmat(ones(x))

    XL = LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    u, s, v = svds(XL, solver=solver, k=n_pcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    v = v[idx, :]

    X_pca = (u * s)[:, idx]
    ev = s[idx] ** 2 / (X.shape[0] - 1)

    total_var = _get_mean_var(X)[1].sum()
    ev_ratio = ev / total_var

    output = {
        'X_pca': X_pca,
        'variance': ev,
        'variance_ratio': ev_ratio,
        'components': v,
    }
    return output


def t_sne(x, n_pcs, n_iter=200):
    """
    the dim reduce function of TSEN

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :param n_iter: the number of iterators.
    :return:  ndarray of shape (n_samples, n_components) Embedding of the training data in low-dimensional space.
    """
    tsen = TSNE(n_components=n_pcs, n_iter=n_iter)
    tsne_x = tsen.fit_transform(x)
    return tsne_x


def u_map(x, n_pcs, n_neighbors=5, min_dist=0.3):
    """
    the dim reduce function of UMAP

    :param x: 2D array, shape (M, N)
    :param n_pcs: the number of features for a return array after reducing.
    :param n_neighbors: the number of neighbors
    :param min_dist: the min value of distance.
    :return: ndarray of shape (n_samples, n_components) Embedding of the training data in low-dimensional space.
    """
    import umap
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_pcs, min_dist=min_dist)
    umap_x = umap_obj.fit_transform(x)
    return umap_x
