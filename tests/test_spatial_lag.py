#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:test_spatial_lag.py
@time:2021/04/20
"""
import sys

sys.path.append('/data/workspace/st/stereopy-release')


from stereo.tools.spatial_lag import SpatialLag
import pandas as pd
from anndata import AnnData
import numpy as np
from stereo.core.stereo_result import ClusterResult

np.random.seed(9)


def init(genes=50, cells=20, dtype='dataframe'):
    gname = [f'g{i}' for i in range(genes)]
    cname = [f'c{i}' for i in range(cells)]
    x = np.random.randint(0, 100, (cells, genes))
    if dtype == 'anndata':
        var = pd.DataFrame(index=gname)
        obs = pd.DataFrame(index=cname)
        groups = np.random.choice(['1', '2', '3'], cells)
        obs['cluster'] = groups
        andata = AnnData(x, obs=obs, var=var)
        andata.obsm['spatial'] = np.random.randint(0, 50, (cells, 2))
        andata.uns['clustering'] = ClusterResult('clustering', cluster_info=obs)
        return andata
    else:
        return pd.DataFrame(x, index=cname, columns=gname)


def test():
    andata = init(30, 100, 'anndata')
    tmp = SpatialLag(data=andata, cluster='clustering')
    res = tmp.fit()
    print(res)


test()
