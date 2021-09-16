#!/usr/bin/env python3
# coding: utf-8
"""
@file: umap.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/09/07  create file.
"""
from typing import Optional
import warnings

import numpy as np
from packaging import version
from sklearn.utils import check_random_state


def umap(
        x: np.ndarray,
        neighbors_connectivities,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_components: int = 2,
        maxiter: Optional[int] = None,
        alpha: float = 1.0,
        gamma: float = 1.0,
        negative_sample_rate: int = 5,
        init_pos: str = 'spectral',
        random_state: int = 0,
        a: Optional[float] = None,
        b: Optional[float] = None,
):
    """\
    Embed the neighborhood graph using UMAP [McInnes18]_.

    Parameters
    ----------
    x
        Annotated data matrix.
    neighbors_connectivities
        connectivities of neighbors.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out. The default of in the `umap-learn` package is
        0.1.
    spread
        The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called `n_epochs`
        in the original UMAP.
    alpha
        The initial learning rate for the embedding optimization.
    gamma
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Called `init` in the
        original UMAP. Options are:
        * 'spectral': use a spectral embedding of the graph.
        * 'random': assign initial embedding positions at random.
    random_state
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` or `Generator`, `random_state` is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    a
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    b
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

    **X_umap** : `adata.obsm` field
        UMAP coordinates of data.
    """

    # Compat for umap 0.4 -> 0.5
    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        import umap

    if version.parse(umap.__version__) >= version.parse("0.5.0"):

        def simplicial_set_embedding(*args, **kwargs):
            from umap.umap_ import simplicial_set_embedding

            X_umap, _ = simplicial_set_embedding(
                *args,
                densmap=False,
                densmap_kwds={},
                output_dens=False,
                **kwargs,
            )
            return X_umap

    else:
        from umap.umap_ import simplicial_set_embedding
    from umap.umap_ import find_ab_params

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
    init_coords = init_pos  # Let umap handle it
    random_state = check_random_state(random_state)

    # the data matrix X is really only used for determining the number of connected components
    # for the init condition in the UMAP embedding
    n_epochs = 0 if maxiter is None else maxiter
    x_umap = simplicial_set_embedding(
        x,
        neighbors_connectivities.tocoo(),
        n_components,
        alpha,
        a,
        b,
        gamma,
        negative_sample_rate,
        n_epochs,
        init_coords,
        random_state,
        'euclidean',
        {},
        verbose=True,
    )
    return x_umap
