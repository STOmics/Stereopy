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


from stereo.tools.dim_reduce import DimReduce
import pandas as pd
from anndata import AnnData
import numpy as np
import scanpy as sc

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


def test_reduce():
    andata = init(30, 100, 'anndata')
    DimReduce(andata=andata, method='pca', name='test_pca', n_pcs=3).fit()
    print(andata.uns.keys())
    print(andata.uns['test_pca'].x_reduce)
    DimReduce(andata=andata, method='tsen', name='test_tsen', n_pcs=3).fit()
    print(andata.uns['test_tsen'].x_reduce)
    DimReduce(andata=andata, method='umap', name='test_umap', n_pcs=3).fit()
    print(andata.uns['test_umap'].x_reduce)
    DimReduce(andata=andata, method='factor_analysis', name='test_factor_analysis', n_pcs=3).fit()
    print(andata.uns['test_factor_analysis'].x_reduce)
    DimReduce(andata=andata, method='low_variance', name='test_low_variance', n_pcs=3).fit()
    print(andata.uns['test_low_variance'].x_reduce)
test_reduce()
