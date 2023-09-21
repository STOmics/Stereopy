#!/usr/bin/env python3
# coding: utf-8
"""
@author: Yiran Wu  wuyiran@genomics.cn
@last modified by: Yiran Wu
@file:neighbors.py
@time:2021/09/07
"""
from types import MappingProxyType
from typing import (
    Optional,
    Type,
    Mapping,
    Any,
    Union
)

import numpy as np
import pandas as pd
from natsort import natsorted
from numpy import random
from packaging import version
from scipy import sparse
from typing_extensions import Literal

from ..log_manager import logger

AnyRandom = Union[None, int, random.RandomState]

try:
    from _louvain.VertexPartition import MutableVertexPartition
except ImportError:

    class MutableVertexPartition:
        pass


    MutableVertexPartition.__module__ = 'louvain.VertexPartition'  # noqa


def louvain(
        neighbor,
        adjacency: sparse.spmatrix,
        resolution: float = None,
        random_state: AnyRandom = 0,
        flavor: Literal['vtraag', 'igraph', 'rapids'] = 'vtraag',
        directed: bool = True,
        use_weights: bool = False,
        partition_type: Optional[Type[MutableVertexPartition]] = None,
        partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
):
    """
    :param neighbor:
        Neighbors object.
    :param adjacency:
        Sparse adjacency matrix of the graph.
    :param resolution:
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn’t accept a `resolution_parameter`.
    :param random_state:
        Change the initialization of the optimization.
    :param flavor:
        Choose between to packages for computing the clustering.
        Including: ``'vtraag'``, ``'igraph'``, ``'taynaud'``.
        ``'vtraag'`` is much more powerful, and the default.
    :param directed:
        If True, treat the graph as directed. If False, undirected.
    :param use_weights:
        Use weights from knn graph.
    :param partition_type:
        Type of partition to use.
        Only a valid argument if ``flavor`` is ``'vtraag'``.
    :param partition_kwargs:
        Key word arguments to pass to partitioning,
        if ``vtraag`` method is being used.
    :return: cluster: pandas.Categorical
    """

    partition_kwargs = dict(partition_kwargs)

    if (flavor != 'vtraag') and (partition_type is not None):
        raise ValueError(
            '`partition_type` is only a valid argument ' 'when `flavour` is "vtraag"'
        )

    if flavor in {'vtraag', 'igraph'}:
        if directed and flavor == 'igraph':
            directed = False
        from .neighbors import Neighbors
        g = Neighbors.get_igraph_from_adjacency(adjacency, directed=directed)
        if use_weights:
            weights = np.array(g.es["weight"]).astype(np.float64)
        else:
            weights = None
        if flavor == 'vtraag':
            import louvain
            if partition_type is None:
                partition_type = louvain.RBConfigurationVertexPartition
            if resolution is not None:
                partition_kwargs["resolution_parameter"] = resolution
            if use_weights:
                partition_kwargs["weights"] = weights
            if version.parse(louvain.__version__) < version.parse("0.7.0"):
                louvain.set_rng_seed(random_state)
            else:
                partition_kwargs["seed"] = random_state
            logger.info('    using the "louvain" package of Traag (2017)')
            part = louvain.find_partition(
                g,
                partition_type,
                **partition_kwargs,
            )
        else:
            part = g.community_multilevel(weights=weights)
        groups = np.array(part.membership)
    elif flavor == 'rapids':
        # nv louvain only works with undirected graphs,
        # and `adjacency` must have a directed edge in both directions
        try:
            import cudf
            import cugraph
        except ImportError:
            raise ImportError(
                "Your env don't have GPU related RAPIDS packages, if you want to run this option, follow the "
                "guide at https://stereopy.readthedocs.io/en/latest/Tutorials/clustering_by_gpu.html")

        offsets = cudf.Series(adjacency.indptr)
        indices = cudf.Series(adjacency.indices)
        params = {"source": "src", "destination": "dst", "renumber": False}
        if use_weights:
            sources, targets = adjacency.nonzero()
            weights = adjacency[sources, targets]
            if isinstance(weights, np.matrix):
                weights = weights.A1
            weights = cudf.Series(weights)
            params["edge_attr"] = "weights"
        else:
            weights = None
        g = cugraph.Graph()

        if hasattr(g, 'add_adj_list'):
            g.add_adj_list(offsets, indices, weights)
        else:
            g.from_cudf_adjlist(offsets, indices, weights)

        original_edge_list = g.view_edge_list()

        # FIXME: according to the bug issue: https://github.com/rapidsai/cugraph/issues/3016 with the `cugraph`
        #  version 22.10.1+0.g9cf07eb6.dirty
        g = cugraph.Graph()
        g.from_cudf_edgelist(original_edge_list, **params)

        # logg.info('    using the "louvain" package of rapids')
        louvain_parts, _ = cugraph.louvain(g)
        groups = (
            louvain_parts.to_pandas()
            .sort_values('vertex')[['partition']]
            .to_numpy()
            .ravel()
        )
    elif flavor == 'taynaud':
        # this is deprecated
        import networkx as nx
        import community

        g = nx.Graph(adjacency)
        partition = community.best_partition(g)
        groups = np.zeros(len(partition), dtype=int)
        for k, v in partition.items():
            groups[k] = v
    else:
        raise ValueError('`flavor` needs to be "vtraag" or "igraph" or "taynaud".')

    groups = groups + 1
    cluster = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(map(str, np.unique(groups))),
    )

    return cluster
