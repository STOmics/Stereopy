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
import leidenalg
import pandas as pd
from typing import Optional, Type
from natsort import natsorted


try:
    from leidenalg.VertexPartition import MutableVertexPartition
except ImportError:
    class MutableVertexPartition:
        pass

    MutableVertexPartition.__module__ = 'leidenalg.VertexPartition'


def leiden(
    neighbor,
    adjacency=None,
    directed: bool = True,
    resolution: float = 1,
    use_weights: bool = True,
    random_state:  int = 0,
    n_iterations: int = -1,
    partition_type: Optional[Type[MutableVertexPartition]] = None,
    **partition_kwargs,
):
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
    #cluster = [str(i) for i in cluster]
    #info = {'bins': data.cell_names, 'cluster': cluster}
    #df = pd.DataFrame(info)
    return cluster
