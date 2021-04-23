#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:test_cell_type_anno.py
@time:2021/03/12
"""
import sys
sys.path.append('/data/workspace/st/stereopy-release')
from stereo.tools.clustering import Clustering
import scanpy as sc


andata = sc.read_h5ad('/data/workspace/st/data/E4/raw_andata.bin100.h5ad')
method='louvain'
outdir=None
dim_reduce_key='dim_reduce'
n_neighbors=30
normalize_key='cluster_normalize'
normalize_method='quantile'
nor_target_sum=10000
name='test_clustering'
cluster = Clustering(data=andata, method=method, outdir=outdir,dim_reduce_key=dim_reduce_key,normalize_method=normalize_method,name=name)
cluster.fit()
print(cluster.data.uns['test_clustering'])
