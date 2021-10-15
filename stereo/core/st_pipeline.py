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
import phenograph as phe
import pandas as pd
from ..algorithm.leiden import leiden as le
from ..algorithm._louvain import louvain as lo
from typing_extensions import Literal


class StPipeline(object):
    def __init__(self, data):
        """
        A analysis tool sets for StereoExpData. include preprocess, filter, cluster, plot and so on.

        :param data: StereoExpData object.
        """
        self.data = data
        self.result = dict()
        self._raw = None

    @property
    def raw(self):
        """
        get the StereoExpData whose exp_matrix is raw count.

        :return:
        """
        return self._raw

    @raw.setter
    def raw(self, value):
        """
        set the raw data.

        :param value: StereoExpData.
        :return:
        """
        self._raw = copy.deepcopy(value)

    def reset_raw_data(self):
        """
        reset the self.data to the raw data.

        :return:
        """
        self.data = self.raw

    def raw_checkpoint(self):
        self.raw = self.data

    def cal_qc(self):
        """
        calculate three qc index including the number of genes expressed in the count matrix, the total counts per cell
        and the percentage of counts in mitochondrial genes.

        :return:
        """
        cal_qc(self.data)

    def filter_cells(self, min_gene=None, max_gene=None, min_n_genes_by_counts=None, max_n_genes_by_counts=None,
                     pct_counts_mt=None, cell_list=None, inplace=True):
        """
        filter cells based on numbers of genes expressed.

        :param min_gene: Minimum number of genes expressed for a cell pass filtering.
        :param max_gene: Maximum number of genes expressed for a cell pass filtering.
        :param min_n_genes_by_counts: Minimum number of  n_genes_by_counts for a cell pass filtering.
        :param max_n_genes_by_counts: Maximum number of  n_genes_by_counts for a cell pass filtering.
        :param pct_counts_mt: Maximum number of  pct_counts_mt for a cell pass filtering.
        :param cell_list: the list of cells which will be filtered.
        :param inplace: whether inplace the original data or return a new data.
        :return:
        """
        return filter_cells(self.data, min_gene, max_gene, min_n_genes_by_counts, max_n_genes_by_counts, pct_counts_mt,
                            cell_list, inplace)

    def filter_genes(self, min_cell=None, max_cell=None, gene_list=None, inplace=True):
        """
        filter genes based on the numbers of cells.

        :param min_cell: Minimum number of cells for a gene pass filtering.
        :param max_cell: Maximun number of cells for a gene pass filtering.
        :param gene_list: the list of genes which will be filtered.
        :param inplace: whether inplace the original data or return a new data.
        :return:
        """
        return filter_genes(self.data, min_cell, max_cell, gene_list, inplace)

    def filter_coordinates(self, min_x=None, max_x=None, min_y=None, max_y=None, inplace=True):
        """
        filter cells based on the coordinates of cells.

        :param min_x: Minimum of x for a cell pass filtering.
        :param max_x: Maximum of x for a cell pass filtering.
        :param min_y: Minimum of y for a cell pass filtering.
        :param max_y: Maximum of y for a cell pass filtering.
        :param inplace: whether inplace the original data or return a new data.
        :return:
        """
        return filter_coordinates(self.data, min_x, max_x, min_y, max_y, inplace)

    def log1p(self, inplace=True, res_key='log1p'):
        """
        log1p for express matrix.

        :param inplace: whether inplace the original data or get a new express matrix after log1p.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        if inplace:
            self.data.exp_matrix = np.log1p(self.data.exp_matrix)
        else:
            self.result[res_key] = np.log1p(self.data.exp_matrix)

    def normalize_total(self, target_sum=10000, inplace=True, res_key='normalize_total'):
        """
        total count normalize the data to `target_sum` reads per cell, so that counts become comparable among cells.

        :param target_sum: the number of reads per cell after normalization.
        :param inplace: whether inplace the original data or get a new express matrix after normalize_total.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        if inplace:
            self.data.exp_matrix = normalize_total(self.data.exp_matrix, target_sum=target_sum)
        else:
            self.result[res_key] = normalize_total(self.data.exp_matrix, target_sum=target_sum)

    def quantile(self, inplace=True, res_key='quantile'):
        """
        Normalize the columns of X to each have the same distribution. Given an expression matrix  of M genes by N
        samples, quantile normalization ensures all samples have the same spread of data (by construction).

        :param inplace: whether inplace the original data or get a new express matrix after quantile.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        if issparse(self.data.exp_matrix):
            self.data.exp_matrix = self.data.exp_matrix.toarray()
        if inplace:
            self.data.exp_matrix = quantile_norm(self.data.exp_matrix)
        else:
            self.result[res_key] = quantile_norm(self.data.exp_matrix)

    def disksmooth_zscore(self, r=20, inplace=True, res_key='disksmooth_zscore'):
        """
        for each position, given a radius, calculate the z-score within this circle as final normalized value.

        :param r: radius for normalization.
        :param inplace: whether inplace the original data or get a new express matrix after disksmooth_zscore.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        if issparse(self.data.exp_matrix):
            self.data.exp_matrix = self.data.exp_matrix.toarray()
        if inplace:
            self.data.exp_matrix = zscore_disksmooth(self.data.exp_matrix, self.data.position, r)
        else:
            self.result[res_key] = zscore_disksmooth(self.data.exp_matrix, self.data.position, r)

    def sctransform(self,
                    method="theta_ml",
                    n_cells=5000,
                    n_genes=2000,
                    filter_hvgs=False,
                    res_clip_range="seurat",
                    var_features_n=3000,
                    inplace=True,
                    res_key='sctransform'):
        """
        scTransform reference Seruat.

        :param method: offset, theta_ml, theta_lbfgs, alpha_lbfgs.
        :param n_cells: Number of cells to use for estimating parameters in Step1: default is 5000.
        :param n_genes: Number of genes to use for estimating parameters in Step1; default is 2000.
        :param filter_hvgs: bool.
        :param res_clip_range: string or list
                    options: 1)"seurat": Clips residuals to -sqrt(ncells/30), sqrt(ncells/30)
                             2)"default": Clips residuals to -sqrt(ncells), sqrt(ncells)
                    only used when filter_hvgs is true.
        :param var_features_n: Number of variable features to select (for calculating a subset of pearson residuals).
        :param inplace: whether inplace the original data or get a new express matrix after sctransform.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        from ..preprocess.sc_transform import sc_transform
        if inplace:
            sc_transform(self.data, method, n_cells, n_genes, filter_hvgs, res_clip_range, var_features_n)
        else:
            import copy
            data = copy.deepcopy(self.data)
            self.result[res_key] = sc_transform(data, method, n_cells, n_genes, filter_hvgs,
                                                res_clip_range, var_features_n)

    def highly_variable_genes(self,
                         groups=None,
                         method: Optional[str] = 'seurat',
                         n_top_genes: Optional[int] = 2000,
                         min_disp: Optional[float] = 0.5,
                         max_disp: Optional[float] = np.inf,
                         min_mean: Optional[float] = 0.0125,
                         max_mean: Optional[float] = 3,
                         span: Optional[float] = 0.3,
                         n_bins: int = 20, res_key='highly_variable_genes'):
        """
        Annotate highly variable genes. reference scanpy.

        :param groups:  If specified, highly-variable genes are selected within each batch separately and merged.
                        This simple process avoids the selection of batch-specific genes and acts as a
                        lightweight batch correction method. For all flavors, genes are first sorted
                        by how many batches they are a HVG. For dispersion-based flavors ties are broken
                        by normalized dispersion. If `flavor = 'seurat_v3'`, ties are broken by the median
                        (across batches) rank based on within-batch normalized variance.
        :param method:  Choose the flavor for identifying highly variable genes. For the dispersion
                        based methods in their default workflows, Seurat passes the cutoffs whereas
                        Cell Ranger passes `n_top_genes`.
        :param n_top_genes: Number of highly-variable genes to keep. Mandatory if `flavor='seurat_v3'`.
        :param min_disp: If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
                         normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        :param max_disp: If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
                         normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        :param min_mean: If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
                         normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        :param max_mean: If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
                         normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        :param span: The fraction of the data (cells) used when estimating the variance in the loess
                         model fit if `flavor='seurat_v3'`.
        :param n_bins: Number of bins for binning the mean gene expression. Normalization is
                       done with respect to each bin. If just a single gene falls into a bin,
                       the normalized dispersion is artificially set to 1. You'll be informed
                       about this if you set `settings.verbosity = 4`.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        from ..tools.highly_variable_genes import HighlyVariableGenes
        hvg = HighlyVariableGenes(self.data, groups=groups, method=method, n_top_genes=n_top_genes, min_disp=min_disp,
                                  max_disp=max_disp, min_mean=min_mean, max_mean=max_mean, span=span, n_bins=n_bins)
        hvg.fit()
        self.result[res_key] = hvg.result

    def subset_by_hvg(self, hvg_res_key, inplace=True):
        """
        get the subset by the result of highly variable genes.

        :param hvg_res_key: the key of highly varialbe genes to getting the result.
        :param inplace: whether inplace the data or get a new data after highly variable genes, which only save the
                        data info of highly variable genes.
        :return: a StereoExpData object.
        """
        data = self.data if inplace else copy.deepcopy(self.data)
        if hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the normalization func.')
        df = self.result[hvg_res_key]
        genes_index = df['highly_variable'].values
        data.sub_by_index(gene_index=genes_index)
        return data

    def pca(self, use_highly_genes, n_pcs, hvg_res_key='highly_variable_genes', res_key='pca'):
        """
        Principal component analysis.

        :param use_highly_genes: Whether to use only the expression of hypervariable genes as input.
        :param n_pcs: the number of features for a return array after reducing.
        :param hvg_res_key: the key of highly varialbe genes to getting the result.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        data = self.subset_by_hvg(hvg_res_key, inplace=False) if use_highly_genes else self.data
        x = data.exp_matrix.toarray() if issparse(data.exp_matrix) else data.exp_matrix
        res = pca(x, n_pcs)
        self.result[res_key] = pd.DataFrame(res['x_pca'])

    # def umap(self, pca_res_key, n_pcs=None, n_neighbors=5, min_dist=0.3, res_key='dim_reduce'):
    #     if pca_res_key not in self.result:
    #         raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
    #     x = self.result[pca_res_key][:, n_pcs] if n_pcs is not None else self.result[pca_res_key]
    #     res = u_map(x, 2, n_neighbors, min_dist)
    #     self.result[res_key] = pd.DataFrame(res)

    def umap(self,
             pca_res_key,
             neighbors_res_key,
             res_key='umap',
             min_dist: float = 0.5,
             spread: float = 1.0,
             n_components: int = 2,
             maxiter: Optional[int] = None,
             alpha: float = 1.0,
             gamma: float = 1.0,
             negative_sample_rate: int = 5,
             init_pos: str = 'spectral', ):
        """
        Embed the neighborhood graph using UMAP [McInnes18]_.

        :param pca_res_key: the key of pca to getting the result. Usually, in spatial omics analysis, the results
                            after using pca are used for umap.
        :param neighbors_res_key: the key of neighbors to getting the connectivities of neighbors result for umap.
        :param res_key: the key for getting the result from the self.result.
        :param min_dist: The effective minimum distance between embedded points. Smaller values
                         will result in a more clustered/clumped embedding where nearby points on
                         the manifold are drawn closer together, while larger values will result
                         on a more even dispersal of points. The value should be set relative to
                         the ``spread`` value, which determines the scale at which embedded
                         points will be spread out. The default of in the `umap-learn` package is
                         0.1.
        :param spread: The effective scale of embedded points. In combination with `min_dist`
                       this determines how clustered/clumped the embedded points are.
        :param n_components: The number of dimensions of the embedding.
        :param maxiter: The number of iterations (epochs) of the optimization. Called `n_epochs`
                        in the original UMAP.
        :param alpha: The initial learning rate for the embedding optimization.
        :param gamma: Weighting applied to negative samples in low dimensional embedding
                      optimization. Values higher than one will result in greater weight
                      being given to negative samples.
        :param negative_sample_rate: The number of negative edge/1-simplex samples to use per positive
                      edge/1-simplex sample in optimizing the low dimensional embedding.
        :param init_pos: How to initialize the low dimensional embedding.Called `init` in the original UMAP.Options are:
                        * 'spectral': use a spectral embedding of the graph.
                        * 'random': assign initial embedding positions at random.
        :return:
        """
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

    def neighbors(self, pca_res_key, method='umap', metric='euclidean', n_pcs=None, n_neighbors=10, knn=True,
                  res_key='neighbors'):
        """
        run the neighbors.

        :param pca_res_key: the key of pca to getting the result.
        :param method: Use 'umap' or 'gauss'. for computing connectivities.
        :param metric: A known metric’s name or a callable that returns a distance.
                        include:
                            * euclidean
                            * manhattan
                            * chebyshev
                            * minkowski
                            * canberra
                            * braycurtis
                            * mahalanobis
                            * wminkowski
                            * seuclidean
                            * cosine
                            * correlation
                            * haversine
                            * hamming
                            * jaccard
                            * dice
                            * russelrao
                            * kulsinski
                            * rogerstanimoto
                            * sokalmichener
                            * sokalsneath
                            * yule
        :param n_pcs: the number of pcs used to runing neighbor.
        :param n_neighbors: Use this number of nearest neighbors.
        :param knn: If `True`, use a hard threshold to restrict the number of neighbors to
                    `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
                    Kernel to assign low weights to neighbors more distant than the
                    `n_neighbors` nearest neighbor.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        neighbor, dists, connectivities = find_neighbors(x=self.result[pca_res_key].values, method=method, n_pcs=n_pcs,
                                                         n_neighbors=n_neighbors, metric=metric, knn=knn)
        res = {'neighbor': neighbor, 'connectivities': connectivities, 'nn_dist': dists}
        self.result[res_key] = res

    def get_neighbors_res(self, neighbors_res_key):
        """
        get the neighbor result by the key.

        :param neighbors_res_key: the key of neighbors to getting the result.
        :return: neighbor, connectivities, nn_dist.
        """
        if neighbors_res_key not in self.result:
            raise Exception(f'{neighbors_res_key} is not in the result, please check and run the neighbors func.')
        neighbors_res = self.result[neighbors_res_key]
        neighbor = neighbors_res['neighbor']
        connectivities = neighbors_res['connectivities']
        nn_dist = neighbors_res['nn_dist']
        return neighbor, connectivities, nn_dist

    def leiden(self,
               neighbors_res_key,
               res_key='cluster',
               directed: bool = True,
               resolution: float = 1,
               use_weights: bool = True,
               random_state: int = 0,
               n_iterations: int = -1
               ):
        """
        leiden of cluster.

        :param neighbors_res_key: the key of neighbors to getting the result.
        :param res_key: the key for getting the result from the self.result.
        :param directed: If True, treat the graph as directed. If False, undirected.
        :param resolution: A parameter value controlling the coarseness of the clustering.
                            Higher values lead to more clusters.
                            Set to `None` if overriding `partition_type`
                            to one that doesn’t accept a `resolution_parameter`.
        :param use_weights: If `True`, edge weights from the graph are used in the computation(placing more emphasis
                            on stronger edges).
        :param random_state: Change the initialization of the optimization.
        :param n_iterations: How many iterations of the Leiden clustering algorithm to perform.
                             Positive values above 2 define the total number of iterations to perform,
                             -1 has the algorithm run until it reaches its optimal clustering.
        :return:
        """
        neighbor, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        clusters = le(neighbor=neighbor, adjacency=connectivities, directed=directed, resolution=resolution,
                      use_weights=use_weights, random_state=random_state, n_iterations=n_iterations)
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
        self.result[res_key] = df

    def louvain(self,
                neighbors_res_key,
                res_key='cluster',
                resolution: float = None,
                random_state: int = 0,
                flavor: Literal['vtraag', 'igraph', 'rapids'] = 'vtraag',
                directed: bool = True,
                use_weights: bool = False
                ):
        """
        louvain of cluster.

        :param neighbors_res_key: the key of neighbors to getting the result.
        :param res_key: the key for getting the result from the self.result.
        :param resolution: A parameter value controlling the coarseness of the clustering.
                            Higher values lead to more clusters.
                            Set to `None` if overriding `partition_type`
                            to one that doesn’t accept a `resolution_parameter`.
        :param random_state: Change the initialization of the optimization.
        :param flavor: Choose between to packages for computing the clustering.
                        Including: ``'vtraag'``, ``'igraph'``, ``'taynaud'``.
                        ``'vtraag'`` is much more powerful, and the default.
        :param directed: If True, treat the graph as directed. If False, undirected.
        :param use_weights: Use weights from knn graph.
        :return:
        """
        neighbor, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        clusters = lo(neighbor=neighbor, resolution=resolution, random_state=random_state,
                      adjacency=connectivities, flavor=flavor, directed=directed, use_weights=use_weights)
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
        self.result[res_key] = df

    def phenograph(self, phenograph_k, pca_res_key, res_key='cluster'):
        """
        phenograph of cluster.

        :param phenograph_k: the k value of phenograph.
        :param pca_res_key: the key of pca to getting the result for running the phenograph.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        communities, _, _ = phe.cluster(self.result[pca_res_key], k=phenograph_k)
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
        """
        a tool of finding maker gene. for each group, find statistical test different genes between one group and
        the rest groups using t_test or wilcoxon_test.

        :param cluster_res_key: the key of cluster to getting the result for group info.
        :param method: t_test or wilcoxon_test.
        :param case_groups: case group info, default all clusters.
        :param control_groups: control group info, default the rest of groups.
        :param corr_method: correlation method.
        :param use_raw: whether use the raw count express matrix for the analysis, default True.
        :param use_highly_genes: Whether to use only the expression of hypervariable genes as input, default True.
        :param hvg_res_key: the key of highly varialbe genes to getting the result.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
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
        """
        spatial lag model, calculate cell-bin's lag coefficient, lag z-stat and p-value.

        :param cluster_res_key: the key of cluster to getting the result for group info.
        :param genes: specify genes, default using all genes.
        :param random_drop: randomly drop bin-cells if True.
        :param drop_dummy: drop specify clusters.
        :param n_neighbors: number of neighbors.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        from ..tools.spatial_lag import SpatialLag
        if cluster_res_key not in self.result:
            raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
        tool = SpatialLag(data=self.data, groups=self.result[cluster_res_key], genes=genes, random_drop=random_drop,
                          drop_dummy=drop_dummy, n_neighbors=n_neighbors)
        tool.fit()
        self.result[res_key] = tool.result

    def spatial_pattern_score(self, use_raw=True, res_key='spatial_pattern'):
        """
        calculate the spatial pattern score.

        :param use_raw: whether use the raw count express matrix for the analysis, default True.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        from ..algorithm.spatial_pattern_score import spatial_pattern_score

        if use_raw and not self.raw:
            raise Exception(f'self.raw must be set if use_raw is True.')
        data = self.raw if use_raw else self.data
        x = data.exp_matrix.toarray() if issparse(data.exp_matrix) else data.exp_matrix
        df = pd.DataFrame(x, columns=data.gene_names, index=data.cell_names)
        res = spatial_pattern_score(df)
        self.result[res_key] = res
