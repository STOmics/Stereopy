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
from ..algorithm.neighbors import find_neighbors
import phenograph
import pandas as pd
from ..algorithm.leiden import leiden
from ..algorithm._louvain import louvain
from typing_extensions import Literal


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
        self._raw = copy.deepcopy(value)

    def data2raw(self):
        self.data = self.raw

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

    def sctransform(self,
                    method="theta_ml",
                    n_cells=5000,
                    n_genes=2000,
                    filter_hvgs=False,
                    res_clip_range="seurat",
                    var_features_n=3000,
                    inplace=True,
                    res_key='sctransform'):
        from ..preprocess.sc_transform import sc_transform
        if inplace:
            sc_transform(self.data, method, n_cells, n_genes, filter_hvgs, res_clip_range, var_features_n)
        else:
            import copy
            data = copy.deepcopy(self.data)
            self.result[res_key] = sc_transform(data, method, n_cells, n_genes, filter_hvgs,
                                                res_clip_range, var_features_n)

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

    def pca(self, use_highly_genes, n_pcs, hvg_res_key=None, res_key='dim_reduce'):
        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        data = self.subset_by_hvg(hvg_res_key, inplace=False) if use_highly_genes else self.data
        x = data.exp_matrix.toarray() if issparse(data.exp_matrix) else data.exp_matrix
        res = pca(x, n_pcs)
        self.result[res_key] = pd.DataFrame(res['x_pca'])

    def umap(self, pca_res_key, n_pcs=None, n_neighbors=5, min_dist=0.3, res_key='dim_reduce'):
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        x = self.result[pca_res_key][:, n_pcs] if n_pcs is not None else self.result[pca_res_key]
        res = u_map(x, 2, n_neighbors, min_dist)
        self.result[res_key] = pd.DataFrame(res)

    def u_map(self,
              pca_res_key,
              neighbors_res_key,
              res_key='dim_reduce',
              min_dist: float = 0.5,
              spread: float = 1.0,
              n_components: int = 2,
              maxiter: Optional[int] = None,
              alpha: float = 1.0,
              gamma: float = 1.0,
              negative_sample_rate: int = 5,
              init_pos: str = 'spectral', ):
        from ..algorithm.umap import umap
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        if neighbors_res_key not in self.result:
            raise Exception(f'{neighbors_res_key} is not in the result, please check and run the neighbors func.')
        _, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        x_umap = umap(x=self.result[pca_res_key], neighbors_connectivities=connectivities,
                      min_dist=min_dist, spread=spread, n_components=n_components, maxiter=maxiter, alpha=alpha,
                      gamma=gamma, negative_sample_rate=negative_sample_rate, init_pos=init_pos)
        self.result[res_key] = pd.DataFrame(x_umap)

    def neighbors(self, pca_res_key, method='umap', metric='euclidean', n_pcs=40, n_neighbors=10, knn=True,
                  res_key='neighbors'):

        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        neighbor, dists, connectivities = find_neighbors(self.result[pca_res_key].values, method, n_pcs, n_neighbors,
                                                         metric, knn)
        res = {'neighbor': neighbor, 'connectivities': connectivities, 'nn_dist': dists}
        self.result[res_key] = res

    def get_neighbors_res(self, neighbors_res_key):
        if neighbors_res_key not in self.result:
            raise Exception(f'{neighbors_res_key} is not in the result, please check and run the neighbors func.')
        neighbors_res = self.result[neighbors_res_key]
        neighbor = neighbors_res['neighbor']
        connectivities = neighbors_res['connectivities']
        nn_dist = neighbors_res['nn_dist']
        return neighbor, connectivities, nn_dist

    def run_leiden(self,
                   neighbors_res_key,
                   res_key='cluster',
                   directed: bool = True,
                   resolution: float = 1,
                   use_weights: bool = True,
                   random_state: int = 0,
                   n_iterations: int = -1
                   ):
        neighbor, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        clusters = leiden(neighbor=neighbor, adjacency=connectivities, directed=directed, resolution=resolution,
                          use_weights=use_weights, random_state=random_state, n_iterations=n_iterations)
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
        self.result[res_key] = df

    def run_louvain(self,
                    neighbors_res_key,
                    res_key='cluster',
                    resolution: float = None,
                    random_state: int = 0,
                    flavor: Literal['vtraag', 'igraph', 'rapids'] = 'vtraag',
                    directed: bool = True,
                    use_weights: bool = False
                    ):
        neighbor, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        clusters = louvain(neighbor=neighbor, resolution=resolution, random_state=random_state,
                           adjacency=connectivities, flavor=flavor, directed=directed, use_weights=use_weights)
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
        self.result[res_key] = df

    def run_phenograph(self, phenograph_k, pca_res_key, res_key='cluster'):
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        communities, _, _ = phenograph.cluster(self.result[pca_res_key], k=phenograph_k)
        clusters = communities.astype(str)
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
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
        from ..tools.find_markers import FindMarker

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
        self.result[res_key] = tool.result

    def spatial_lag(self,
                    cluster_res_key,
                    genes=None,
                    random_drop=True,
                    drop_dummy=None,
                    n_neighbors=8,
                    res_key='spatial_lag'):
        from ..tools.spatial_lag import SpatialLag
        if cluster_res_key not in self.result:
            raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
        tool = SpatialLag(data=self.data, groups=self.result[cluster_res_key], genes=genes, random_drop=random_drop,
                          drop_dummy=drop_dummy, n_neighbors=n_neighbors)
        tool.fit()
        self.result[res_key] = tool.result

    def spatial_pattern_score(self, use_raw=True, res_key='spatial_pattern'):
        from ..algorithm.spatial_pattern_score import spatial_pattern_score

        if use_raw and not self.raw:
            raise Exception(f'self.raw must be set if use_raw is True.')
        data = self.raw if use_raw else self.data
        x = data.exp_matrix.toarray() if issparse(data.exp_matrix) else data.exp_matrix
        df = pd.DataFrame(x, columns=data.gene_names, index=data.cell_names)
        res = spatial_pattern_score(df)
        self.result[res_key] = res
