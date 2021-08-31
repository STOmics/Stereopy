#!/usr/bin/env python3
# coding: utf-8
"""
@file: st_pipeline.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/07/20  create file.
"""
from ..preprocess.qc import cal_qc
from ..preprocess.filter import filter_cells, filter_genes, filter_coordinates
from ..algorithm.normalization import normalize_total, quantile_norm, zscore_disksmooth
import numpy as np
from scipy.sparse import issparse
from ..algorithm.dim_reduce import pca, u_map
from typing import Optional, Union
import copy
from ..algorithm.neighbors import Neighbors
import leidenalg as la
import phenograph
import pandas as pd
from ..tools.find_markers import FindMarker
from ..tools.spatial_pattern_score import SpatialPatternScore
from ..tools.spatial_lag import SpatialLag


class StPipeline(object):
    def __init__(self, data):
        self.data = data
        self.result = dict()
        self._raw = None

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, value):
        self._raw = value

    def cal_qc(self):
        cal_qc(self.data)

    def filter_cells(self, min_gene=None, max_gene=None, n_genes_by_counts=None, pct_counts_mt=None, cell_list=None,
                     inplace=True):
        return filter_cells(self.data, min_gene, max_gene, n_genes_by_counts, pct_counts_mt, cell_list, inplace)

    def filter_genes(self, min_cell=None, max_cell=None, gene_list=None, inplace=True):
        return filter_genes(self.data, min_cell, max_cell, gene_list, inplace)

    def filter_coordinates(self, min_x=None, max_x=None, min_y=None, max_y=None, inplace=True):
        return filter_coordinates(self.data, min_x, max_x, min_y, max_y, inplace)

    def log1p(self, inplace=True, res_key='log1p'):
        if inplace:
            self.data.exp_matrix = np.log1p(self.data.exp_matrix)
        else:
            self.result[res_key] = np.log1p(self.data.exp_matrix)

    def normalize_total(self, target_sum=10000, inplace=True, res_key='normalize_total'):
        if inplace:
            self.data.exp_matrix = normalize_total(self.data.exp_matrix, target_sum=target_sum)
        else:
            self.result[res_key] = normalize_total(self.data.exp_matrix, target_sum=target_sum)

    def quantile(self, inplace=True, res_key='quantile'):
        if issparse(self.data.exp_matrix):
            self.data.exp_matrix = self.data.exp_matrix.toarray()
        if inplace:
            self.data.exp_matrix = quantile_norm(self.data.exp_matrix)
        else:
            self.result[res_key] = quantile_norm(self.data.exp_matrix)

    def disksmooth_zscore(self, r, inplace=True, res_key='disksmooth_zscore'):
        if issparse(self.data.exp_matrix):
            self.data.exp_matrix = self.data.exp_matrix.toarray()
        if inplace:
            self.data.exp_matrix = zscore_disksmooth(self.data.exp_matrix, self.data.partitions, r)
        else:
            self.result[res_key] = zscore_disksmooth(self.data.exp_matrix, self.data.partitions, r)

    def sctransform(self):
        pass

    def highly_var_genes(self,
                         groups=None,
                         method: Optional[str] = 'seurat',
                         n_top_genes: Optional[int] = 2000,
                         min_disp: Optional[float] = 0.5,
                         max_disp: Optional[float] = np.inf,
                         min_mean: Optional[float] = 0.0125,
                         max_mean: Optional[float] = 3,
                         span: Optional[float] = 0.3,
                         n_bins: int = 20, res_key='highly_var_genes'):
        from ..tools.highly_variable_genes import HighlyVariableGenes
        hvg = HighlyVariableGenes(self.data, groups=groups, method=method, n_top_genes=n_top_genes, min_disp=min_disp,
                                  max_disp=max_disp, min_mean=min_mean, max_mean=max_mean, span=span, n_bins=n_bins)
        hvg.fit()
        self.result[res_key] = hvg.result

    def subset_by_hvg(self, hvg_res_key, inplace=True):
        data = self.data if inplace else copy.deepcopy(self.data)
        if hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the normalization func.')
        df = self.result[hvg_res_key]
        genes_index = df['highly_variable'].values
        data.sub_by_index(gene_index=genes_index)
        return data

    def pca(self, use_highly_genes, hvg_res_key, n_pcs, res_key='pca'):
        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        data = self.subset_by_hvg(hvg_res_key, inplace=False) if use_highly_genes else self.data
        x = data.exp_matrix.toarray() if issparse(data.exp_matrix) else data.exp_matrix
        res = pca(x, n_pcs)
        self.result[res_key] = res

    def umap(self, use_highly_genes, hvg_res_key, n_pcs, n_neighbors=5, min_dist=0.3, res_key='pca'):
        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        data = self.subset_by_hvg(hvg_res_key, inplace=False) if use_highly_genes else self.data
        x = data.exp_matrix.toarray() if issparse(data.exp_matrix) else data.exp_matrix
        res = u_map(x, n_pcs, n_neighbors, min_dist)
        self.result[res_key] = res

    def neighbors(self, pca_res_key, n_neighbors, res_key='neighbors'):
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        neighbor = Neighbors(self.result[pca_res_key], n_neighbors)
        nn_idx, nn_dist = neighbor.find_n_neighbors()
        res = {'neighbor': neighbor, 'nn_idx': nn_idx, 'nn_dist': nn_dist}
        self.result[res_key] = res

    def get_neighbors_res(self, neighbors_res_key):
        if neighbors_res_key not in self.result:
            raise Exception(f'{neighbors_res_key} is not in the result, please check and run the neighbors func.')
        neighbors_res = self.result[neighbors_res_key]
        neighbor = neighbors_res['neighbor']
        nn_idx = neighbors_res['nn_idx']
        nn_dist = neighbors_res['nn_dist']
        return neighbor, nn_idx, nn_dist

    def leiden(self, neighbors_res_key, res_key='leiden', diff=1):
        neighbor, nn_idx, nn_dist = self.get_neighbors_res(neighbors_res_key)
        g = neighbor.get_igraph_from_knn(nn_idx, nn_dist)
        optimiser = la.Optimiser()
        leiden_partition = la.ModularityVertexPartition(g, weights=g.es['weight'])
        while diff > 0:
            diff = optimiser.optimise_partition(leiden_partition, n_iterations=10)
        clusters = np.arange(len(self.data.cell_names))
        for i in range(len(leiden_partition)):
            clusters[leiden_partition[i]] = str(i)
        df = pd.DataFrame({'bins': self.data.cell_names, 'cluster': clusters})
        self.result[res_key] = df

    def louvain(self, neighbors_res_key, res_key='louvain'):
        neighbor, nn_idx, nn_dist = self.get_neighbors_res(neighbors_res_key)
        g = neighbor.get_igraph_from_knn(nn_idx, nn_dist)
        louvain_partition = g.community_multilevel(weights=g.es['weight'], return_levels=False)
        clusters = np.arange(len(self.data.cell_names))
        for i in range(len(louvain_partition)):
            clusters[louvain_partition[i]] = str(i)
        df = pd.DataFrame({'bins': self.data.cell_names, 'cluster': clusters})
        self.result[res_key] = df

    def phenograph_cluster(self, phenograph_k, pca_res_key, res_key='phenograph'):
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        communities, _, _ = phenograph.cluster(self.result[phenograph_k], k=phenograph_k)
        clusters = communities.astype(str)
        df = pd.DataFrame({'bins': self.data.cell_names, 'cluster': clusters})
        self.result[res_key] = df

    def find_marker_genes(self,
                          cluster_res_key,
                          method: str = 't-test',
                          case_groups: Union[str, np.ndarray] = 'all',
                          control_groups: Union[str, np.ndarray] = 'rest',
                          corr_method: str = 'bonferroni',
                          use_raw: bool = True,
                          use_highly_genes: bool = True,
                          hvg_res_key: Optional[str] = None,
                          res_key: str = 'marker_genes'
                          ):
        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        if use_raw and not self.raw:
            raise Exception(f'self.raw must be set if use_raw is True.')
        if cluster_res_key not in self.result:
            raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
        data = self.raw if use_raw else self.data
        data = self.subset_by_hvg(hvg_res_key, inplace=False) if use_highly_genes else data
        tool = FindMarker(data=data, groups=self.result[cluster_res_key], method=method, case_groups=case_groups,
                          control_groups=control_groups, corr_method=corr_method)
        tool.fit()
        self.result[res_key] = tool.result

    def spatial_lag(self,
                    cluster_res_key,
                    genes=None,
                    random_drop=True,
                    drop_dummy=None,
                    n_neighbors=8,
                    res_key='spatial_lag'):
        if cluster_res_key not in self.result:
            raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
        tool = SpatialLag(data=self.data, groups=self.result[cluster_res_key], genes=genes, random_drop=random_drop,
                          drop_dummy=drop_dummy, n_neighbors=n_neighbors)
        tool.fit()
        self.result[res_key] = tool.result

    def spatial_pattern_score(self, res_key='spatial_pattern'):
        tool = SpatialPatternScore(data=self.data)
        tool.fit()
        self.result[res_key] = tool.result
