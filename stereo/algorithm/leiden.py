#!/usr/bin/env python3
# coding: utf-8
"""
@file: leiden.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/09/07  create file.
"""

import numpy as np
from scipy import sparse
import leidenalg
import pandas as pd
from typing import Optional, Type, Union
from natsort import natsorted
from numpy import random
AnyRandom = Union[None, int, random.RandomState]

try:
    from leidenalg.VertexPartition import MutableVertexPartition
except ImportError:
    class MutableVertexPartition:
        pass

    MutableVertexPartition.__module__ = 'leidenalg.VertexPartition'


def leiden(
    neighbor,
    adjacency: sparse.spmatrix,
    directed: bool = True,
    resolution: float = 1,
    use_weights: bool = True,
    random_state: AnyRandom = 0,
    n_iterations: int = -1,
    partition_type: Optional[Type[MutableVertexPartition]] = None,
    **partition_kwargs,
):
    """

    :param neighbor:
        Neighbors object.
    :param adjacency:
        Sparse adjacency matrix of the graph.
    :param directed:
        If True, treat the graph as directed. If False, undirected.
    :param resolution:
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn’t accept a `resolution_parameter`.
    :param use_weights:
        If `True`, edge weights from the graph are used in the computation(placing more emphasis on stronger edges).
    :param random_state:
        Change the initialization of the optimization.
    :param n_iterations:
        How many iterations of the Leiden clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
    :param partition_type:
        Type of partition to use.
        Defaults to :class:`~leidenalg.RBConfigurationVertexPartition`.
        For the available options, consult the documentation for
        :func:`~leidenalg.find_partition`.
    :param partition_kwargs:
        Any further arguments to pass to `~leidenalg.find_partition`
        (which in turn passes arguments to the `partition_type`).
    :return: cluster: pandas.Categorical
    """
    partition_kwargs = dict(partition_kwargs)
    # convert it to igraph
    g = neighbor.get_igraph_from_adjacency(adjacency, directed=directed)
    # filp to the default partition type if not overriden by the user
    if partition_type is None:
        partition_type = leidenalg.RBConfigurationVertexPartition
    if use_weights:
        partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
        partition_kwargs['n_iterations'] = n_iterations
        partition_kwargs['seed'] = random_state
    if resolution is not None:
        partition_kwargs['resolution_parameter'] = resolution
    # clustering proper
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    # store output
    groups = np.array(part.membership)
    cluster = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(map(str, np.unique(groups))),
    )
    return cluster
