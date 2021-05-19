#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:test_find_markers.py
@time:2021/03/16
"""
import sys

sys.path.append('/data/workspace/st/stereopy-release')


from stereo.tools.spatial_pattern_score import *
import pandas as pd
from anndata import AnnData
import numpy as np

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
        return andata
    else:
        return pd.DataFrame(x, index=cname, columns=gname)


def test():
    andata = init(30, 100, 'anndata')
    tmp = SpatialPatternScore(data=andata)
    tmp.fit()
    print(andata.var)


test()
