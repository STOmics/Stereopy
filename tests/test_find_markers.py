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


from stereo.tools.find_markers import *
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


def test_t_test():
    test_group = init(50, 20)
    control_group = init(50, 30)
    result1 = t_test(test_group, control_group)
    print(result1)


def test_wilcoxon():
    test_group = init(50, 20)
    control_group = init(50, 30)
    result1 = wilcoxon_test(test_group, control_group)


def test_find_marker_gene():
    andata = init(30, 100, 'anndata')
    marker = FindMarker(data=andata, cluster='cluster', corr_method='bonferroni', method='wilcoxon', name='marker_test')
    marker.fit()
    for i in andata.uns['marker_test']:
        print(i, str(andata.uns['marker_test'][i]))
        print(andata.uns['marker_test'][i].degs_data)

    print('###########scanpy##########')
    sc.tl.rank_genes_groups(andata, 'cluster')


test_find_marker_gene()
