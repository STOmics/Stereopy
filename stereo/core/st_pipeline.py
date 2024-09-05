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

import copy
from functools import wraps
from multiprocessing import cpu_count
from typing import (
    Optional,
    Union,
    List
)

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from typing_extensions import Literal

from .result import Result, AnnBasedResult
from .stereo_exp_data import AnnBasedStereoExpData
from .stereo_exp_data import StereoExpData
from ..log_manager import logger
from ..utils.time_consume import TimeConsume

tc = TimeConsume()


def logit(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        logger.info('start to run {}...'.format(func.__name__))
        tk = tc.start()
        res = func(*args, **kwargs)
        logger.info('{} end, consume time {:.4f}s.'.format(func.__name__, tc.get_time_consumed(key=tk, restart=False)))
        return res

    return wrapped


class StPipeline(object):

    def __init__(self, data: Union[StereoExpData, AnnBasedStereoExpData]):
        """
        A analysis tool sets for StereoExpData. include preprocess, filter, cluster, plot and so on.

        :param data: StereoExpData object.
        """
        self.data: Union[StereoExpData, AnnBasedStereoExpData] = data
        self.result = Result(data)
        self._raw: Union[StereoExpData, AnnBasedStereoExpData] = None
        self._key_record = {'hvg': [], 'pca': [], 'neighbors': [], 'umap': [], 'cluster': [], 'marker_genes': []}
        # self.reset_key_record = self._reset_key_record

    def __getattr__(self, item):
        dict_attr = self.__dict__.get(item, None)
        if dict_attr:
            return dict_attr

        # start with __ may not be our algorithm function, and will cause import problem
        if item.startswith('__'):
            raise AttributeError

        from ..algorithm.algorithm_base import AlgorithmBase
        new_attr = AlgorithmBase.get_attribute_helper(item, self.data, self.result)
        if new_attr:
            self.__setattr__(item, new_attr)
            logger.info(f'register algorithm {item} to {self}')
            return new_attr

        raise AttributeError(
            f'{item} not existed, please check the function name you called!'
        )

    @property
    def key_record(self):
        return self._key_record

    @key_record.setter
    def key_record(self, key_record):
        self._key_record = key_record

    @property
    def raw(self) -> Union[StereoExpData, AnnBasedStereoExpData]:
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
        Reset `self.data` to the raw data saved in `self.raw` when you want data
        get raw expression matrix.

        :return:
        """
        # self.data = self.raw
        self.data.exp_matrix = copy.deepcopy(self.raw.exp_matrix)
        self.data.cells = copy.deepcopy(self.raw.cells)
        self.data.genes = copy.deepcopy(self.raw.genes)
        self.data.position = copy.deepcopy(self.raw.position)
        self.data.position_z = copy.deepcopy(self.raw.position_z)
        from stereo.preprocess.qc import cal_qc
        cal_qc(self.data)

    def raw_checkpoint(self):
        """
        Save current data to `self.raw`. Running this function will be a convinent choice,
        when your data have gone through several steps of basic preprocessing.

        Parameters
        -----------------------------

        Returns
        -----------------------------
        None
        """
        self.raw = self.data

    def reset_key_record(self, key, res_key):
        """
        reset key and coordinated res_key in key_record.
        :param key:
        :param res_key:
        :return:
        """
        if key in self.key_record.keys():
            if res_key in self.key_record[key]:
                self.key_record[key].remove(res_key)
            self.key_record[key].append(res_key)
        else:
            self.key_record[key] = [res_key]

    @logit
    def cal_qc(self):
        """
        Calculate the key indicators of quality control.

        Observation level metrics include:
            * total_counts: total number of counts for a cell.
            * n_genes_by_count: number of genes expressed of counts for a cell.
            * pct_counts_mt: percentage of total counts in a cell which are mitochondrial.

        Parameters
        ---------------------

        Returns
        ---------------------
        A StereoExpData object storing quality control indicators, including two levels of obs (cell) and var (gene).

        """
        from ..preprocess.qc import cal_qc
        cal_qc(self.data)

    @logit
    def filter_cells(
        self,
        min_counts: Optional[int] = None,
        max_counts: Optional[int] = None,
        min_genes: Optional[int] = None,
        max_genes: Optional[int] = None,
        pct_counts_mt: Optional[float] = None,
        cell_list: Optional[list] = None,
        filter_raw: Optional[bool] = True,
        excluded: Optional[bool] = False,
        inplace: bool = True,
        **kwargs
    ):
        """
        Filter cells based on counts or the numbers of genes expressed.

        Parameters
        ----------------------
        min_counts
            minimum number of counts required for a cell to pass fitlering.
        max_counts
            maximum number of counts required for a cell to pass fitlering.
        min_genes
            minimum number of genes expressed required for a cell to pass filtering.
        max_genes
            maximum number of genes expressed required for a cell to pass filtering.
        pct_counts_mt
            maximum number of `pct_counts_mt` required for a cell to pass filtering.
        cell_list
            the list of cells to be retained.
        filter_raw
            whether to filter raw data meanwhile.
        excluded
            set it to True to exclude the cells which are specified by parameter `cell_list` while False to include.
        inplace
            whether to replace the previous data or return a new data.

        Returns
        ------------------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those filtered.
        """
        from ..preprocess.filter import filter_cells
        min_counts = kwargs.get('min_gene', None) if min_counts is None else min_counts
        max_counts = kwargs.get('max_gene', None) if max_counts is None else max_counts
        min_genes = kwargs.get('min_n_genes_by_counts', None) if min_genes is None else min_genes
        max_genes = kwargs.get('max_n_genes_by_counts', None) if max_genes is None else max_genes
        data = filter_cells(self.data, min_counts, max_counts, min_genes, max_genes, pct_counts_mt,
                            cell_list, excluded, inplace)
        if data.raw is not None and filter_raw:
            # filter_cells(data.raw, min_gene, max_gene, min_n_genes_by_counts, max_n_genes_by_counts, pct_counts_mt,
            #              cell_list, True)
            filter_cells(data.raw, cell_list=data.cell_names, inplace=True)
            if isinstance(data, AnnBasedStereoExpData):
                data.adata.raw = data.raw.adata
        return data

    @logit
    def filter_genes(
        self,
        min_cells: Optional[int] = None,
        max_cells: Optional[int] = None,
        min_counts: Optional[int] = None,
        max_counts: Optional[int] = None,
        gene_list: Optional[Union[list, np.ndarray]] = None,
        mean_umi_gt: Optional[float] = None,
        filter_raw: Optional[bool] = True,
        excluded: Optional[bool] = False,
        filter_mt_genes: Optional[bool] = False,
        inplace: bool = True,
        **kwargs
    ):
        """
        Filter genes based on the numbers of cells or counts.

        Parameters
        ---------------------
        min_cells
            minimum number of cells expressed required for a gene to pass filering.
        max_cells
            maximum number of cells expressed required for a gene to pass filering.
        min_counts
            minimum number of counts expressed required for a gene to pass filtering.
        max_counts
            maximum number of counts expressed required for a gene to pass filtering.
        gene_list
            the list of genes to be retained.
        mean_umi_gt
            mean counts greater than this value for a gene to pass filtering.
        filter_raw
            whether to filter raw data meanwhile.
        excluded
            set it to True to exclude the genes which are specified by parameter `gene_list` while False to include.
        filter_mt_genes
            whether to filter out mitochondrial genes.
        inplace
            whether to replace the previous data or return a new data.

        Returns
        --------------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those filtered.
        """
        from ..preprocess.filter import filter_genes
        min_cells = kwargs.get('min_cell', None) if min_cells is None else min_cells
        max_cells = kwargs.get('max_cell', None) if max_cells is None else max_cells
        min_counts = kwargs.get('min_count', None) if min_counts is None else min_counts
        max_counts = kwargs.get('max_count', None) if max_counts is None else max_counts
        data = filter_genes(self.data, min_cells, max_cells, min_counts, max_counts, gene_list, mean_umi_gt, excluded, filter_mt_genes, inplace)
        if data.raw is not None and filter_raw:
            filter_genes(data.raw, gene_list=data.genes.gene_name, inplace=True)
            if isinstance(data, AnnBasedStereoExpData):
                data.adata.raw = data.raw.adata
        return data

    @logit
    def filter_by_hvgs(self,
                       hvg_res_key: str = 'highly_variable_genes',
                       filter_raw: bool = True,
                       inplace: bool = False):
        """
        Filter genes based on the result of highly_variable_genes function.

        Parameters
        ---------------------
        hvg_res_key
            the key of highly variable genes to get corresponding result.
        filter_raw
            whether to filter raw data meanwhile.
        inplace
            whether to replace the previous data or return a new data.

        Returns
        --------------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those filtered.
        """
        if hvg_res_key not in self.result:
            raise KeyError(f'Can not find result of highly_variable_genes function by key {hvg_res_key}.')

        from ..preprocess import filter_genes
        hvgs_flag = self.result[hvg_res_key]['highly_variable'].to_numpy()
        hvgs = self.data.gene_names[hvgs_flag]
        hvg_result_filtered = self.result[hvg_res_key][hvgs_flag]
        data = filter_genes(self.data, gene_list=hvgs, inplace=inplace)
        if data.raw is not None and filter_raw:
            filter_genes(data.raw, gene_list=hvgs, inplace=True)
            if isinstance(data, AnnBasedStereoExpData):
                data.adata.raw = data.raw.adata
        data.tl.result[hvg_res_key] = hvg_result_filtered
        return data

    @logit
    def filter_coordinates(self,
                           min_x: int = None,
                           max_x: int = None,
                           min_y: int = None,
                           max_y: int = None,
                           filter_raw: bool = True,
                           inplace: bool = True):
        """
        Filter cells based on coordinate information.

        Parameters
        -----------------
        min_x
            minimum of coordinate x for a cell to pass filtering.
        max_x
            maximum of coordinate x for a cell to pass filtering.
        min_y
            minimum of coordinate y for a cell to pass filtering.
        max_y
            maximum of coordinate y for a cell to pass filtering.
        filter_raw
            whether to filter raw data meanwhile.
        inplace
            whether to replace the previous data or return a new data.

        Returns
        --------------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those filtered.
        """
        from ..preprocess.filter import filter_coordinates
        data = filter_coordinates(self.data, min_x, max_x, min_y, max_y, inplace)
        if data.raw is not None and filter_raw:
            filter_coordinates(data.raw, min_x, max_x, min_y, max_y, True)
            if isinstance(data, AnnBasedStereoExpData):
                data.adata.raw = data.raw.adata
        return data

    @logit
    def filter_by_clusters(
            self,
            cluster_res_key: str = 'cluster',
            groups: Union[str, np.ndarray, List[str]] = None,
            excluded: bool = False,
            filter_raw: Optional[bool] = True,
            inplace: bool = False
    ):
        """
        Filter cells based on clustering result.

        Parameters
        -----------------
        cluster_res_key
            the key of clustering to get corresponding result from `self.result`.
        groups
            the groups in clustering result which will be retained or filtered based on the value of `excluded`.
        excluded:
            set it to True to exclude the groups which are specified by parameter `groups` while False to include.
        filter_raw
            whether to filter raw data meanwhile.
        inplace
            whether to replace the previous data or return a new data.

        Returns
        --------------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those filtered.
        """
        from ..preprocess.filter import filter_by_clusters, filter_cells

        if cluster_res_key not in self.result:
            raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')

        data, cluster_res = filter_by_clusters(self.data, self.result[cluster_res_key], groups, excluded, inplace)
        data.tl.result[cluster_res_key] = cluster_res
        gene_exp_cluster_key = f'gene_exp_{cluster_res_key}'
        if gene_exp_cluster_key in data.tl.result:
            if isinstance(groups, str):
                groups = [groups]
            data.tl.result[gene_exp_cluster_key] = data.tl.result[gene_exp_cluster_key][groups]
        if data.raw is not None and filter_raw:
            filter_cells(data.raw, cell_list=data.cell_names, inplace=True)
            if isinstance(data, AnnBasedStereoExpData):
                data.adata.raw = data.raw.adata
        return data

    @logit
    def log1p(self,
              inplace: bool = True,
              res_key: str = 'log1p'):
        """
        Transform the express matrix logarithmically.

        Parameters
        -----------------
        inplace
            whether to replace previous data or get a new express matrix after normalization of log1p.
        res_key
            the key to get targeted result from `self.result`.

        Returns
        ----------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those normalized.
        """
        if inplace:
            self.data.exp_matrix = np.log1p(self.data.exp_matrix)
        else:
            self.result[res_key] = np.log1p(self.data.exp_matrix)

    @logit
    def normalize_total(self,
                        target_sum: int = 10000,
                        inplace: bool = True,
                        res_key: str = 'normalize_total'):
        """
        Normalize total counts over all genes per cell such that each cell has the same
        total count after normalization.

        Parameters
        -----------------------
        target_sum
            the number of total counts per cell after normalization, if `None`, each cell has a
            total count equal to the median of total counts for all cells before normalization.
        inplace
            whether to replace previous data or get a new express matrix after normalize_total.
        res_key
            the key to get targeted result from `self.result`.

        Returns
        ----------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those normalized
        """
        from ..algorithm.normalization import normalize_total
        if inplace:
            self.data.exp_matrix = normalize_total(self.data.exp_matrix, target_sum=target_sum)
        else:
            self.result[res_key] = normalize_total(self.data.exp_matrix, target_sum=target_sum)

    @logit
    def scale(self,
              zero_center: bool = True,
              max_value: Optional[float] = None,
              inplace: bool = True,
              res_key: str = 'scale'):
        """
        Scale express matrix to unit variance and zero mean.

        Parameters
        --------------------
        zero_center
            if `False`, ignore zero variables, which allows to deal with sparse input efficently.
        max_value
            truncate to this value after scaling, if `None`, do not truncate.
        inplace
            whether to replace the previous data or get a new express matrix after scaling.
        res_key
            the key to get targeted result from `self.result`.

        Returns
        -----------------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those scaled.
        """
        from ..algorithm.scale import scale
        if inplace:
            self.data.exp_matrix = scale(self.data.exp_matrix, zero_center, max_value)
        else:
            self.result[res_key] = scale(self.data.exp_matrix, zero_center, max_value)

    @logit
    def quantile(self, inplace=True, res_key='quantile'):
        """
        Normalize the columns of X to each have the same distribution. Given an expression matrix  of M genes by N
        samples, quantile normalization ensures all samples have the same spread of data (by construction).

        :param inplace: whether replace the original data or get a new express matrix after quantile.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        from ..algorithm.normalization import quantile_norm
        if issparse(self.data.exp_matrix):
            self.data.exp_matrix = self.data.exp_matrix.toarray()
        if inplace:
            self.data.exp_matrix = quantile_norm(self.data.exp_matrix)
        else:
            self.result[res_key] = quantile_norm(self.data.exp_matrix)

    @logit
    def disksmooth_zscore(self, r=20, inplace=True, res_key='disksmooth_zscore'):
        """
        for each position, given a radius, calculate the z-score within this circle as final normalized value.

        :param r: radius for normalization.
        :param inplace: whether replace the original data or get a new express matrix after disksmooth_zscore.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        from ..algorithm.normalization import zscore_disksmooth
        if issparse(self.data.exp_matrix):
            self.data.exp_matrix = self.data.exp_matrix.toarray()
        if inplace:
            self.data.exp_matrix = zscore_disksmooth(self.data.exp_matrix, self.data.position, r)
        else:
            self.result[res_key] = zscore_disksmooth(self.data.exp_matrix, self.data.position, r)

    @logit
    def sctransform(
            self,
            n_cells: int = 5000,
            n_genes: int = 2000,
            filter_hvgs: bool = True,
            var_features_n: int = 3000,
            inplace: bool = True,
            res_key: str = 'sctransform',
            exp_matrix_key: str = "scale.data",
            seed_use: int = 1448145,
            filter_raw: Optional[bool] = True,
            **kwargs
    ):
        """
        Normalization of scTransform, refering to Seurat [Hafemeister19]_.

        Parameters
        ----------------------
        n_cells
            number of cells to use for estimating parameters.
        n_genes
            number of genes to use for estimating parameters. means all genes.
        filter_hvgs
            True to retain data associated with highly variable genes only while False to entire data.
        var_features_n
            the number of variable features to select, for calculating a subset of pearson residuals.
        inplace
            whether to replace the previous expression data.
        res_key
            the key to get targeted result from `self.result`.
        exp_matrix_key
            which expression matrix to use for analysis.
        seed_use
            random seed.
        filter_raw
            because this function will filter data, whether to filter raw data meanwhile by setting `filter_raw`.

        Returns
        -----------
        An object of StereoExpData.
        Depending on `inplace`, if `True`, the data will be replaced by those normalized.
        """
        from ..preprocess.sc_transform import sc_transform
        from ..preprocess.filter import filter_genes
        data = self.data if inplace else copy.deepcopy(self.data)
        self.result[res_key] = sc_transform(data, n_cells, n_genes, filter_hvgs, var_features_n,
                                            exp_matrix_key=exp_matrix_key, seed_use=seed_use, **kwargs)
        key = 'sct'
        self.reset_key_record(key, res_key)

        if data.raw is not None and filter_raw and data.shape != data.raw.shape:
            filter_genes(data.raw, gene_list=data.gene_names, inplace=True)
            if isinstance(data, AnnBasedStereoExpData):
                data.adata.raw = data.raw.adata

    @logit
    def highly_variable_genes(
            self,
            groups: Optional[str] = None,
            method: Literal['seurat', 'cell_ranger', 'seurat_v3'] = 'seurat',
            n_top_genes: Optional[int] = 2000,
            min_disp: Optional[float] = 0.5,
            max_disp: Optional[float] = np.inf,
            min_mean: Optional[float] = 0.0125,
            max_mean: Optional[float] = 3,
            span: Optional[float] = 0.3,
            n_bins: int = 20,
            res_key='highly_variable_genes'
    ):
        """
        Annotate highly variable genes, refering to Scanpy.
        Which method to implement depends on `flavor`,including Seurat [Satija15]_ ,
        Cell Ranger [Zheng17]_ and Seurat v3 [Stuart19]_.

        Parameters
        ----------------------
        groups
            if specified, highly variable genes are selected within each batch separately and merged,
            which simply avoids the selection of batch-specific genes and acts as a lightweight batch
            correction method. For all flavors, genes are first sorted by how many batches they are a HVG.
            For dispersion-based flavors ties are broken by normalized dispersion. If `flavor`
            is `'seurat_v3'`, ties are broken by the median (across batches) rank based on within-
            batch normalized variance.
        method
            Choose the flavor to identify highly variable genes. For the dispersion-based methods in
            their default workflows, Seurat passes the cutoffs whereas Cell Ranger passes `n_top_genes`.
        n_top_genes
            number of highly variable genes to keep. Mandatory if `flavor='seurat_v3'`.
        min_disp
            if `n_top_genes` is not None, this and all other cutoffs for the means and the normalized
            dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        max_disp
            if `n_top_genes` is not None, this and all other cutoffs for the means and the normalized
            dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        min_mean
            if `n_top_genes` is not None, this and all other cutoffs for the means and the normalized
            dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        max_mean
            if `n_top_genes` is not None, this and all other cutoffs for the means and the normalized
            dispersions are ignored. Ignored if `flavor='seurat_v3'`.
        span
            the fraction of data (cells) used when estimating the variance in the Loess model fit
            if `flavor='seurat_v3'`.
        n_bins
            number of bins for binning the mean gene expression. Normalization is done with respect to
            each bin. If just a single gene falls into a bin, the normalized dispersion is artificially set to 1.
        res_key
            the key for getting the result from `self.result`.

        Returns
        -----------------
        An object of StereoExpData with the result of highly variable genes.

        """
        from ..tools.highly_variable_genes import HighlyVariableGenes
        hvg = HighlyVariableGenes(self.data, groups=groups, method=method, n_top_genes=n_top_genes, min_disp=min_disp,
                                  max_disp=max_disp, min_mean=min_mean, max_mean=max_mean, span=span, n_bins=n_bins)
        hvg.fit()
        self.result[res_key] = hvg.result
        key = 'hvg'
        self.reset_key_record(key, res_key)

    def subset_by_hvg(self, hvg_res_key, use_raw=False, inplace=True):
        """
        get the subset by the result of highly variable genes.

        :param hvg_res_key: the key of highly varialbe genes to getting the result.
        :param inplace: whether replace the data or get a new data after highly variable genes, which only save the
                        data info of highly variable genes.
        :return: a StereoExpData object.
        """
        if not use_raw:
            data = self.data if inplace else copy.deepcopy(self.data)
        else:
            data = self.raw if inplace else copy.deepcopy(self.raw)
        if hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the normalization func.')
        df = self.result[hvg_res_key]
        genes_index = df['highly_variable'].values
        data.sub_by_index(gene_index=genes_index)
        return data

    @logit
    def pca(self,
            use_highly_genes: bool = False,
            n_pcs: int = None,
            svd_solver: Literal['auto', 'full', 'arpack', 'randomized'] = 'auto',
            hvg_res_key: Optional[str] = 'highly_variable_genes',
            random_state: Optional[Union[None, int, np.random.RandomState]] = 0,
            dtype: str = 'float32',
            res_key: str = 'pca'):
        """
        Principal component analysis.

        :param use_highly_genes: whether to use the expression of hypervariable genes only.
        :param n_pcs: the number of principle components to compute.
        :param svd_solver: default to `'auto'`.

                    - If `'auto'` :
                        The solver is selected by a default policy based on `X.shape` and
                        `n_pcs`: if the input data is larger than 500x500 and the
                        number of components to extract is lower than 80% of the smallest
                        dimension of the data, then the more efficient 'randomized'
                        method is enabled. Otherwise the exact full SVD is computed and
                        optionally truncated afterwards.
                    - If `'full'` :
                        run exact full SVD calling the standard LAPACK solver via
                        `scipy.linalg.svd` and select the components by postprocessing
                    - If `'arpack'` :
                        run SVD truncated to n_pcs calling ARPACK solver via
                        `scipy.sparse.linalg.svds`. It requires strictly
                        0 < n_pcs < min(x.shape)
                    - If `'randomized'` :
                        run randomized SVD.

        :param hvg_res_key: the key of highly variable genes to get targeted result,`use_highly_genes=True` is a necessary prerequisite.
        :param random_state: change to use different initial states for the optimization, fixed value to fixed result.
        :param dtype: numpy data type string to which to convert the result.
        :param res_key: the key for storage of PCA result.

        :return: Computation result of principal component analysis is stored in `self.result` where the result key is `'pca'`.
        """  # noqa
        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        # data = self.subset_by_hvg(hvg_res_key, inplace=False) if use_highly_genes else self.data
        from ..algorithm.dim_reduce import pca

        if use_highly_genes:
            hvgs = self.result[hvg_res_key]['highly_variable']
            exp_matrix = self.data.exp_matrix[:, hvgs]
        else:
            exp_matrix = self.data.exp_matrix
        if n_pcs is None:
            n_pcs = min(exp_matrix.shape) - 1
            if n_pcs > 50:
                n_pcs = 50
        res = pca(exp_matrix, n_pcs, svd_solver=svd_solver, random_state=random_state, dtype=dtype)

        self.result[res_key] = pd.DataFrame(res['x_pca'])
        self.result[f'{res_key}_variance_ratio'] = res['variance_ratio']
        if use_highly_genes:
            pcs = np.zeros((self.data.shape[1], n_pcs), dtype=res['pcs'].dtype)
            pcs[hvgs] = res['pcs']
        else:
            pcs = res['pcs']
        self.result['PCs'] = pcs
        key = 'pca'
        self.reset_key_record(key, res_key)

    # def umap(self, pca_res_key, n_pcs=None, n_neighbors=5, min_dist=0.3, res_key='dim_reduce'):
    #     if pca_res_key not in self.result:
    #         raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
    #     x = self.result[pca_res_key][:, n_pcs] if n_pcs is not None else self.result[pca_res_key]
    #     res = u_map(x, 2, n_neighbors, min_dist)
    #     self.result[res_key] = pd.DataFrame(res)

    @logit
    def umap(
            self,
            pca_res_key: str = 'pca',
            neighbors_res_key: str = 'neighbors',
            res_key: str = 'umap',
            min_dist: float = 0.5,
            spread: float = 1.0,
            n_components: int = 2,
            maxiter: Optional[int] = None,
            alpha: float = 1.0,
            gamma: float = 1.0,
            negative_sample_rate: int = 5,
            init_pos: str = 'spectral',
            method: str = 'umap',
            random_state: int = 0,
            parallel: bool = False
    ):
        """
        Embed the neighborhood graph using UMAP [McInnes18]_.

        :param pca_res_key: the key of PCA analysis to get corresponding result from `self.result`.
        :param neighbors_res_key: the key of neighbors to get corresponding result from `self.result`.
        :param res_key: the key for storing result of UMAP.
        :param min_dist: the effective minimum distance between embedded points. Smaller values
                         will result in a more clustered/clumped embedding where nearby points on
                         the manifold are drawn closer together, while larger values will result
                         on a more even dispersal of points. The value should be set relative to
                         the ``spread`` value, which determines the scale at which embedded
                         points will be spread out. The default of in the `umap-learn` package is
                         0.1.
        :param spread: the effective scale of embedded points. In combination with `min_dist`
                       this determines how clustered/clumped the embedded points are.
        :param n_components: the number of dimensions of the embedding.
        :param maxiter: the number of iterations (epochs) of the optimization. Called `n_epochs` in the original UMAP.
        :param alpha: the initial learning rate for the embedding optimization.
        :param gamma: weighting applied to negative samples in low dimensional embedding
                      optimization. Values higher than one will result in greater weight
                      being given to negative samples.
        :param negative_sample_rate: the number of negative edge/1-simplex samples to use per positive
                      edge/1-simplex sample in optimizing the low dimensional embedding.
        :param init_pos: how to initialize the low dimensional embedding. Called init in the original UMAP.
                        Options are:
                            `'spectral'`: use a spectral embedding of the graph.
                            `'random'`: assign initial embedding positions at random.
        :param method: Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
        :return: UMAP result is stored in `self.result` where the result key is `'umap'`.
        """
        from ..algorithm.umap import umap
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        if neighbors_res_key not in self.result:
            raise Exception(f'{neighbors_res_key} is not in the result, please check and run the neighbors func.')
        _, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        x_umap = umap(
            x=self.result[pca_res_key],
            neighbors_connectivities=connectivities,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            maxiter=maxiter,
            alpha=alpha,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            init_pos=init_pos,
            method=method,
            random_state=random_state,
            parallel=parallel
        )
        self.result[res_key] = pd.DataFrame(x_umap)
        key = 'umap'
        self.reset_key_record(key, res_key)

    @logit
    def neighbors(self,
                  pca_res_key: str = 'pca',
                  method: Literal['umap', 'gauss'] = 'umap',
                  metric: str = 'euclidean',
                  n_pcs: int = None,
                  n_neighbors: int = 10,
                  knn: bool = True,
                  n_jobs: int = 10,
                  res_key: str = 'neighbors'):
        """
        Compute a spatial neighborhood graph over all cells.

        :param pca_res_key: the key of PCA analysis to get corresponding result from `self.result`.
        :param method: use `umap` or `gauss` to compute connectivities.
        :param metric: a known metric's name or a callable that returns a distance,
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
        :param n_pcs: the number of principle components to run neighbors, default is None such that `self.X` is used.
        :param n_neighbors: the size of nearest neighbors.
        :param knn: if `True`, use a hard threshold to restrict the number of neighbors to `n_neighbors`,
                    namely consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights
                    to neighbors more distant than the `n_neighbors` nearest neighbors.
        :param n_jobs: the number of parallel running jobs for neighbors, if set to `-1`, all CPUs will
                    be used. Notice that extremely high value of `n_jobs` may cause segment fault.
        :param res_key: the key for storing result of neighbors, default is `neighbors`.

        :return: Neighbors result is stored in `self.result` where the result key is `'neighbors'`.
        """
        if pca_res_key != 'spatial' and pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        if n_jobs > cpu_count():
            n_jobs = -1
        if pca_res_key == 'spatial':
            pca_res = self.data.position
        else:
            pca_res = self.result[pca_res_key].to_numpy()
        if n_pcs is None:
            n_pcs = pca_res.shape[1]
        from ..algorithm.neighbors import find_neighbors
        neighbor, dists, connectivities = find_neighbors(x=pca_res, method=method, n_pcs=n_pcs,
                                                         n_neighbors=n_neighbors, metric=metric, knn=knn, n_jobs=n_jobs)
        res = {
            'neighbor': neighbor,
            'connectivities': connectivities, 'nn_dist': dists,
            'n_neighbors': n_neighbors, 'method': method, 'metric': metric}
        self.result[res_key] = res
        key = 'neighbors'
        self.reset_key_record(key, res_key)

    def get_neighbors_res(self, neighbors_res_key, ):
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

    @logit
    def spatial_neighbors(self,
                          neighbors_res_key: str = 'neighbors',
                          n_neighbors: int = 6,
                          res_key: str = 'spatial_neighbors'):
        """
        Create a graph from spatial coordinates using Squidpy.

        :param neighbors_res_key: the key of neighbors to getting the result.
        :param n_neighbors: 6 or 4, the number of neighboring tiles.
        :param res_key: the key for getting the result from the `self.result`.
        :return: Spatial neighbors result is stored in `self.result` where the result key is `'spatial_neighbors'`.
        """
        from ..io.reader import stereo_to_anndata
        import squidpy as sq
        neighbor, connectivities, dists = copy.deepcopy(self.get_neighbors_res(neighbors_res_key))
        if isinstance(self.data, AnnBasedStereoExpData):
            adata = self.data.adata
        else:
            adata = stereo_to_anndata(self.data, split_batches=False)
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors)
        connectivities.data[connectivities.data > 0] = 1
        adj = connectivities + adata.obsp['spatial_connectivities']
        adj.data[adj.data > 0] = 1
        res = {'neighbor': neighbor, 'connectivities': adj, 'nn_dist': dists, 'n_neighbors': n_neighbors}
        self.result[res_key] = res
        key = 'neighbors'
        self.reset_key_record(key, res_key)

    @logit
    def leiden(self,
               neighbors_res_key: str = 'neighbors',
               res_key: str = 'leiden',
               directed: bool = True,
               resolution: float = 1,
               use_weights: bool = True,
               random_state: int = 0,
               n_iterations: int = -1,
               method='normal'
               ):
        """
        Cluster cells into subgroups by Leiden algorithm [Traag18]_.

        :param neighbors_res_key: the key of neighbors to get corresponding result from `self.result`.
        :param res_key: the key for storing result of Leiden clustering.
        :param directed: if `True`, treat the graph as directed. If `False`, undirected.
        :param resolution: a parameter value controlling the coarseness of the clustering.
                            Higher values lead to more clusters.
                            Set to `None` if overriding `partition_type`
                            to one that doesn’t accept a `resolution_parameter`.
        :param use_weights: if `True`, edge weights from the graph are used in computation, more emphasis should be placed on stronger edges. # noqa
        :param random_state: change the initialization of the optimization.
        :param n_iterations: how many iterations of the Leiden clustering algorithm to perform.
                             Positive values above 2 define the total number of iterations to perform,
                             `-1` has the algorithm run until it reaches its optimal clustering.
        :param method: Use the original 'normal' implementation, or 'rapids' (experimental, GPU only)
        :return: Clustering result of Leiden is stored in `self.result` where the key is `'leiden'`.
        """
        neighbor, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        if method == 'rapids':
            from ..algorithm.leiden import leiden_rapids
            clusters = leiden_rapids(adjacency=connectivities, resolution=resolution)
        else:
            from ..algorithm.leiden import leiden as le
            clusters = le(neighbor=neighbor, adjacency=connectivities, directed=directed, resolution=resolution,
                          use_weights=use_weights, random_state=random_state, n_iterations=n_iterations)
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
        self.result[res_key] = df
        key = 'cluster'
        self.reset_key_record(key, res_key)
        gene_cluster_res_key = f'gene_exp_{res_key}'
        from ..utils.pipeline_utils import cell_cluster_to_gene_exp_cluster
        gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(self.data, res_key)
        if gene_exp_cluster_res is not False:
            self.result[gene_cluster_res_key] = gene_exp_cluster_res
            self.reset_key_record('gene_exp_cluster', gene_cluster_res_key)

    @logit
    def louvain(self,
                neighbors_res_key: str = 'neighbors',
                res_key: str = 'louvain',
                resolution: float = None,
                random_state: int = 0,
                flavor: Literal['vtraag', 'igraph', 'rapids'] = 'vtraag',
                directed: bool = True,
                use_weights: bool = False
                ):
        """
        Cluster cells into subgroups by Louvain algorithm [Blondel08]_.

        :param neighbors_res_key: the key of neighbors to get corresponding result from `self.result`.
        :param res_key: the key for storing result of Louvain clustering.
        :param resolution: a parameter value to control the coarseness of clustering where higher value
                            leads to more clusters.
        :param random_state: change the initialization of the optimization.
        :param flavor: choose among of packages for computing the clustering.
                        `'vtraag'` is much more powerful, and the default.
                        Set to `None` if overriding `partition_type` to one that
                        doesn't accept a `resolution_parameter`.
        :param directed: if `True`, treat the graph as directed. If `False`, undirected.
        :param use_weights: use weights from knn graph.

        :return: Clustering result of Louvain is stored in `self.result` where the key is `'louvain'`.
        """
        neighbor, connectivities, _ = self.get_neighbors_res(neighbors_res_key)
        from ..algorithm._louvain import louvain as lo
        from ..utils.pipeline_utils import cell_cluster_to_gene_exp_cluster
        clusters = lo(neighbor=neighbor, resolution=resolution, random_state=random_state,
                      adjacency=connectivities, flavor=flavor, directed=directed, use_weights=use_weights)
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
        self.result[res_key] = df
        key = 'cluster'
        self.reset_key_record(key, res_key)
        gene_cluster_res_key = f'gene_exp_{res_key}'
        gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(self.data, res_key)
        if gene_exp_cluster_res is not False:
            self.result[gene_cluster_res_key] = gene_exp_cluster_res
            self.reset_key_record('gene_exp_cluster', gene_cluster_res_key)

    @logit
    def phenograph(self,
                   phenograph_k: int = 30,
                   pca_res_key: str = 'pca',
                   n_jobs: int = 10,
                   res_key: str = 'phenograph',
                   seed: int = 0):
        """
        Cluster cells into subgroups by Phenograph.

        :param phenograph_k: the k value of Phenograph.
        :param pca_res_key: the key of PCA analysis to get corresponding result from `self.result`.
        :param n_jobs: the number of parallel jobs to run for neighbors search. If set to `-1`, all CPUs will be used.
                    Too high value may cause segment fault.
        :param res_key: the key for storing result of Phenograph clustering.
        :param seed: leiden initialization of the optimization.
        :return: Clustering result of Phenograph is stored in `self.result` where the key is `'phenograph'`.
        """
        if pca_res_key not in self.result:
            raise Exception(f'{pca_res_key} is not in the result, please check and run the pca func.')
        import phenograph as phe
        from natsort import natsorted
        from ..utils.pipeline_utils import cell_cluster_to_gene_exp_cluster
        communities, _, _ = phe.cluster(self.result[pca_res_key], k=phenograph_k, clustering_algo='leiden',
                                        n_jobs=n_jobs, seed=seed)
        communities = communities + 1
        clusters = pd.Categorical(
            values=communities.astype('U'),
            categories=natsorted(map(str, np.unique(communities))),
        )
        df = pd.DataFrame({'bins': self.data.cell_names, 'group': clusters})
        self.result[res_key] = df
        key = 'cluster'
        self.reset_key_record(key, res_key)
        gene_cluster_res_key = f'gene_exp_{res_key}'
        gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(self.data, res_key)
        if gene_exp_cluster_res is not False:
            self.result[gene_cluster_res_key] = gene_exp_cluster_res
            self.reset_key_record('gene_exp_cluster', gene_cluster_res_key)

    @logit
    def find_marker_genes(self,
                          cluster_res_key,
                          method: Literal['t_test', 'wilcoxon_test', 'logreg'] = 't_test',
                          case_groups: Union[str, np.ndarray, list] = 'all',
                          control_groups: Union[str, np.ndarray, list] = 'rest',
                          corr_method: Literal['bonferroni', 'benjamini-hochberg'] = 'benjamini-hochberg',
                          use_raw: bool = True,
                          use_highly_genes: bool = True,
                          hvg_res_key: Optional[str] = 'highly_variable_genes',
                          res_key: str = 'marker_genes',
                          output: Optional[str] = None,
                          sort_by='scores',
                          n_genes: Union[str, int] = 'all',
                          ascending: bool = False,
                          n_jobs: int = 4
                          ):
        """
        A tool to find maker genes. For each group, find statistical test different genes
        between one group and the rest groups using `t_test` or `wilcoxon_test`.

        :param cluster_res_key: the key of clustering to get corresponding result from `self.result`.
        :param method: choose method for statistics.
        :param case_groups: case group, default all clusters.
        :param control_groups: control group, default the rest of groups.
        :param corr_method: p-value correction method, only available for `t_test` and `wilcoxon_test`.
        :param use_raw: whether to use raw express matrix for analysis, default True.
        :param use_highly_genes: whether to use only the expression of hypervariable genes as input, default True.
        :param hvg_res_key: the key of highly variable genes to get corresponding result.
        :param res_key: the key for storing result of marker genes.
        :param output: the path to output file `.csv`. If None, do not generate output file.
        :param sort_by: default to 'scores', the result will sort by the key, other options 'log2fc'.
        :param n_genes: default to 0, means will auto calculate n_genes by N = 10000/K². K is cluster number, and N is
                larger or equal to 1, less or equal to 50.
        :param ascending: default to False.
        :param n_jobs: the number of parallel jobs to run. default to 4.
        :return: The result of marker genes is stored in `self.result` where the key is `'marker_genes'`.
        """
        from ..tools.find_markers import FindMarker

        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        if use_raw and not self.raw:
            raise Exception('self.raw must be set if use_raw is True.')
        if cluster_res_key not in self.result:
            raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
        if self.result[cluster_res_key]['group'].unique().size <= 1:
            raise Exception('this function must be based on a cluster result which includes at least two groups.')
        data = self.raw if use_raw else self.data
        data = self.subset_by_hvg(hvg_res_key, use_raw=use_raw, inplace=False) if use_highly_genes else data

        if n_jobs <= 0:
            n_jobs = cpu_count()

        from stereo.utils.pipeline_utils import calc_pct_and_pct_rest, cell_cluster_to_gene_exp_cluster
        pct, pct_rest = calc_pct_and_pct_rest(self.data, cluster_res_key, filter_raw=False)
        mean_count_in_cluster = cell_cluster_to_gene_exp_cluster(self.data, cluster_res_key, kind='mean', filter_raw=False)

        tool = FindMarker(data=data, groups=self.result[cluster_res_key], method=method, case_groups=case_groups,
                          control_groups=control_groups, corr_method=corr_method, sort_by=sort_by,
                          n_genes=n_genes, ascending=ascending, n_jobs=n_jobs, pct=pct, pct_rest=pct_rest, mean_count=mean_count_in_cluster)
        result = tool.result
        result['parameters'] = {
            'cluster_res_key': cluster_res_key,
            'method': method,
            'control_groups': control_groups,
            'corr_method': corr_method,
            'use_raw': use_raw
        }
        self.result[res_key] = result
        if output is not None:
            import natsort
            result = self.result[res_key]
            show_cols = ['genes', 'scores', 'pvalues', 'pvalues_adj', 'log2fc', 'pct', 'pct_rest']
            if self.data.genes.real_gene_name is not None:
                show_cols.insert(1, 'gene_name')
            groups = natsort.natsorted([key for key in result.keys() if '.vs.' in key])
            dat = pd.concat(
                [
                    pd.DataFrame(
                        {group.split(".vs.")[0] + "_" + key: result[group][key].values}
                    ) for group in groups for key in show_cols
                ],
                axis=1
            )
            dat.to_csv(output)
        key = 'marker_genes'
        self.reset_key_record(key, res_key)

    # TODO old method can not use
    # @logit
    # def spatial_lag(self,
    #                 cluster_res_key,
    #                 genes=None,
    #                 random_drop=True,
    #                 drop_dummy=None,
    #                 n_neighbors=8,
    #                 res_key='spatial_lag'):
    #     """
    #     spatial lag model, calculate cell-bin's lag coefficient, lag z-stat and p-value.
    #
    #     :param cluster_res_key: the key of cluster to getting the result for group info.
    #     :param genes: specify genes, default using all genes.
    #     :param random_drop: randomly drop bin-cells if True.
    #     :param drop_dummy: drop specify clusters.
    #     :param n_neighbors: number of neighbors.
    #     :param res_key: the key for getting the result from the self.result.
    #     :return:
    #     """
    #     from ..tools.spatial_lag import SpatialLag
    #     if cluster_res_key not in self.result:
    #         raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
    #     tool = SpatialLag(data=self.data, groups=self.result[cluster_res_key], genes=genes, random_drop=random_drop,
    #                       drop_dummy=drop_dummy, n_neighbors=n_neighbors)
    #     tool.fit()
    #     self.result[res_key] = tool.result

    @logit
    def spatial_pattern_score(self, use_raw=True, res_key='spatial_pattern'):
        """
        calculate the spatial pattern score.

        :param use_raw: whether use the raw count express matrix for the analysis, default True.
        :param res_key: the key for getting the result from the self.result.
        :return:
        """
        from ..algorithm.spatial_pattern_score import spatial_pattern_score

        if use_raw and not self.raw:
            raise Exception('self.raw must be set if use_raw is True.')
        data = self.raw if use_raw else self.data
        x = data.exp_matrix.toarray() if issparse(data.exp_matrix) else data.exp_matrix
        df = pd.DataFrame(x, columns=data.gene_names, index=data.cell_names)
        res = spatial_pattern_score(df)
        self.result[res_key] = res

    @logit
    def spatial_hotspot(self,
                        use_highly_genes: bool = True,
                        hvg_res_key: Optional[str] = 'highly_variable_genes',
                        model: Literal['danb', 'bernoilli', 'normal', 'none'] = 'normal',
                        n_neighbors: int = 30,
                        n_jobs: int = 20,
                        fdr_threshold: float = 0.05,
                        min_gene_threshold: int = 10,
                        outdir: str = None,
                        res_key: str = 'spatial_hotspot',
                        use_raw: bool = True, ):
        """
        Identify informative genes or gene modules.

        :param use_highly_genes: whether to use only the expression of hypervariable genes as input, default True.
        :param hvg_res_key: the key of highly variable genes to get corresponding result.
        :param model: specify the null model on gene expression from below:
                        `'danb'`: Depth-Adjusted Negative Binomial
                        `'bernoulli'`: Models probability of detection
                        `'normal'`: Depth-Adjusted Normal
                        `'none'`: Assumes data has been pre-standardized
        :param n_neighbors: the neighborhood size.
        :param n_jobs: the number of parallel jobs to run.
        :param fdr_threshold: correlation threshold at which to stop assigning genes into modules.
        :param min_gene_threshold: threshold that controls how small the modules could be.
            Increase if there are too many modules being formed,
            and decrease if substructre is not being captured.
        :param outdir: the path to output file(`hotspot.pkl`), containing total hotspot object.
        :param res_key: the key for storing result of spatial hotspot.
        :param use_raw: whether to use raw express matrix for analysis.

        :return: The result of spatial hotspot is stored in `self.result` where the key is `'spatial_hotspot'`.
        """
        from ..algorithm.spatial_hotspot import spatial_hotspot
        if use_highly_genes and hvg_res_key not in self.result:
            raise Exception(f'{hvg_res_key} is not in the result, please check and run the highly_var_genes func.')
        if use_raw and not self.raw:
            raise Exception('self.raw must be set if use_raw is True.')
        data = copy.deepcopy(self.raw) if use_raw else copy.deepcopy(self.data)
        if use_highly_genes:
            df = self.result[hvg_res_key]
            highly_genes_name = df.index[df['highly_variable']]
            data = data.sub_by_name(gene_name=highly_genes_name)
        hs = spatial_hotspot(data, model=model, n_neighbors=n_neighbors, n_jobs=n_jobs, fdr_threshold=fdr_threshold,
                             min_gene_threshold=min_gene_threshold, outdir=outdir)
        self.result[res_key] = hs
        self.reset_key_record('spatial_hotspot', res_key)

    @logit
    def gaussian_smooth(self,
                        n_neighbors: int = 10,
                        smooth_threshold: int = 90,
                        pca_res_key: str = 'pca',
                        n_jobs: int = -1,
                        inplace: bool = True
                        ):
        """
        Smooth the express matrix by the algorithm of Gaussian smoothing [Shen22]_.

        :param n_neighbors: the number of the nearest points to search.
            Too high value may cause overfitting, and too low value may cause porr smoothing effect.
        :param smooth_threshold: the threshold that indicates Gaussian variance with a value between 20 and 100.
            Also too high value may cause overfitting, and low value may cause poor smoothing effect.
        :param pca_res_key: the key of PCA to get targeted result from `self.result`.
        :param n_jobs: the number of parallel jobs to run, if `-1`, all CPUs will be used.
        :param inplace: whether to replace the previous express matrix or get a new StereoExpData object with the new express matrix. # noqa

        :return: An object of StereoExpData with the express matrix processed by Gaussian smooting.
        """
        assert pca_res_key in self.result, f'{pca_res_key} is not in the result, please check and run the pca func.'
        assert self.raw is not None, 'no raw exp_matrix to be saved, please check and run the raw_checkpoint.'
        assert n_neighbors > 0, 'n_neighbors must be greater than 0'
        assert smooth_threshold >= 20 and smooth_threshold <= 100, 'smooth_threshold must be between 20 and 100'

        pca_exp_matrix = self.result[pca_res_key].to_numpy()
        raw_exp_matrix = self.raw.exp_matrix if self.raw.issparse() else self.raw.array2sparse()

        if pca_exp_matrix.shape[0] != raw_exp_matrix.shape[0]:
            raise Exception(
                """
                The first dimension of pca matrix not equals to raw express matrix's,
                it may be because of running data.tl.raw_checkpoint before filtering cells and/or filtering genes.
                """
            )

        from ..algorithm.gaussian_smooth import gaussian_smooth

        if n_jobs <= 0 or n_jobs > cpu_count():
            n_jobs = cpu_count()

        result = gaussian_smooth(
            pca_exp_matrix,
            raw_exp_matrix,
            self.data.position,
            n_neighbors=n_neighbors,
            smooth_threshold=smooth_threshold,
            n_jobs=n_jobs
        )
        data = self.data if inplace else copy.deepcopy(self.data)
        data.exp_matrix = result
        return data

    def lr_score(
            self,
            lr_pairs: Union[list, np.array],
            distance: Union[int, float] = 5,
            spot_comp: pd.DataFrame = None,
            verbose: bool = True,
            key_add: str = 'cci_score',
            min_exp: Union[int, float] = 0,
            use_raw: bool = False,
            min_spots: int = 20,
            n_pairs: int = 1000,
            adj_method: str = "fdr_bh",
            bin_scale: int = 1,
            n_jobs: int = 4,
            res_key: str = 'lr_score'
    ):
        """calculate cci score for each LR pair and do permutation test

        Parameters
        ----------
        lr_pairs : Union[list, np.array]
            LR pairs
        distance : Union[int, float], optional
            the distance between spots which are considered as neighbors , by default 5
        spot_comp : `pd.DataFrame`, optional
            spot component of different cells, by default None
        key_add : str, optional
            key added in `result`, by default 'cci_score'
        min_exp : Union[int, float], optional
            the min expression of ligand or receptor gene when caculate reaction strength, by default 0
        use_raw : bool, optional
            whether to use counts in `adata.raw.X`, by default False
        min_spots : int, optional
            the min number of spots that score > 0, by default 20
        n_pairs : int, optional
            number of pairs to random sample, by default 1000
        adj_method : str, optional
            adjust method of p value, by default "fdr_bh"
        bin_scale : int, optional
            to scale the distance `distance = bin_scale * distance`, by default 1
        n_jobs: int, optional
            the number of parallel jobs to run, by default 4
        res_key:  str, optional
            the key for getting the result after integrating from the self.result, defaults to 'lr_score'

        Raises
        ------
        ValueError
            _description_
        """
        from ..tools.LR_interaction import LrInteraction
        interaction = LrInteraction(self,
                                    verbose=verbose,
                                    bin_scale=bin_scale,
                                    distance=distance,
                                    spot_comp=spot_comp,
                                    n_jobs=n_jobs,
                                    min_exp=min_exp,
                                    min_spots=min_spots,
                                    n_pairs=n_pairs,
                                    )

        result = interaction.fit(lr_pairs=lr_pairs,
                                 adj_method=adj_method,
                                 use_raw=use_raw,
                                 key_add=key_add)

        self.result[res_key] = result

    @logit
    def batches_integrate(self, pca_res_key='pca', res_key='pca_integrated', **kwargs):
        """integrate different experiments base on the pca result

        :param pca_res_key: the key of original pca to get from self.result, defaults to 'pca'
        :param res_key: the key for getting the result after integrating from the self.result, defaults to 'pca_integrated' # noqa
        """
        import harmonypy as hm
        assert pca_res_key in self.result, f'{pca_res_key} is not in the result, please check and run the pca method.'
        assert self.data.cells.batch is not None, 'this is not a data were merged from different experiments'

        out = hm.run_harmony(self.result[pca_res_key], self.data.cells.to_df(), 'batch', **kwargs)
        self.result[res_key] = pd.DataFrame(out.Z_corr.T)
        key = 'pca'
        self.reset_key_record(key, res_key)

    @logit
    def annotation(
            self,
            annotation_information: Union[list, dict],
            cluster_res_key: str = 'cluster',
            default: str = None,
            res_key: str = 'annotation'
    ):
        """
        Set annotation to clusters.

        :param annotation_information: describe the annotation information to the clusters in a list or dictionary format. # noqa
        :param cluster_res_key: get the targeted cluster result to add annotation.
        :param default: the default value for the groups without being annotated, if None, remain the original value.
        :param res_key: the key for storing annotation result in `self.result`.

        """

        assert cluster_res_key in self.result, f'{cluster_res_key} is not in the result, please check and run the ' \
                                               f'cluster func.'

        # df = copy.deepcopy(self.result[cluster_res_key])
        # if isinstance(annotation_information, list):
        #     df.group.cat.categories = np.unique(annotation_information)
        # elif isinstance(annotation_information, dict):
        #     new_annotation_list = []
        #     for i in df.group.cat.categories:
        #         new_annotation_list.append(annotation_information[i])
        #     df.group.cat.categories = new_annotation_list

        cluster_res: pd.DataFrame = self.result[cluster_res_key]
        if cluster_res['group'].dtype.name != 'category':
            cluster_res['group'] = cluster_res['group'].astype('category')

        # if isinstance(annotation_information, (list, np.ndarray)) and \
        #         len(annotation_information) != cluster_res['group'].cat.categories.size:
        #     raise Exception(f"The length of annotation information is {len(annotation_information)}, \
        #                     not equal to the categories of cluster result whose"
        #                     f" length is {cluster_res['group'].cat.categories.size}.")

        if isinstance(annotation_information, (list, np.ndarray)):
            if len(annotation_information) < cluster_res['group'].cat.categories.size:
                new_categories_list = []
                for i in range(cluster_res['group'].cat.categories.size):
                    if i < len(annotation_information):
                        new_categories_list.append(annotation_information[i])
                    else:
                        new_categories_list.append(cluster_res['group'].cat.categories[i] if default is None else default)
            else:
                new_categories_list = np.array(annotation_information, dtype='U')
        elif isinstance(annotation_information, dict):
            new_categories_list = []
            for i in cluster_res['group'].cat.categories:
                if i in annotation_information:
                    new_categories_list.append(annotation_information[i])
                else:
                    new_categories_list.append(i if default is None else default)
            # new_categories = np.array(new_categories_list, dtype='U')
        else:
            raise TypeError("The type of 'annotation_information' only supports list, ndarray or dict.")
        
        new_categories = np.array(new_categories_list, dtype='U')

        new_categories_values = new_categories[cluster_res['group'].cat.codes]

        self.result[res_key] = pd.DataFrame(data={
            'bins': cluster_res['bins'],
            'group': pd.Series(new_categories_values, dtype='category')
        })

        key = 'cluster'
        self.reset_key_record(key, res_key)

        from ..utils.pipeline_utils import cell_cluster_to_gene_exp_cluster
        gene_cluster_res_key = f'gene_exp_{res_key}'
        gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(self.data, res_key)
        if gene_exp_cluster_res is not False:
            self.result[gene_cluster_res_key] = gene_exp_cluster_res
            self.reset_key_record('gene_exp_cluster', gene_cluster_res_key)

    @logit
    def filter_marker_genes(
            self,
            marker_genes_res_key='marker_genes',
            min_fold_change=1,
            min_in_group_fraction=0.25,
            max_out_group_fraction=0.5,
            compare_abs=False,
            res_key='marker_genes_filtered',
            output=None
    ):
        """Filters out genes based on log fold change and fraction of genes expressing the gene within and outside each group.

        :param marker_genes_res_key: The key of the result of find_marker_genes to get from self.result, defaults to 'marker_genes'
        :param min_fold_change: Minimum threshold of log fold change, defaults to None
        :param min_in_group_fraction:  Minimum fraction of cells expressing the genes for each group, defaults to None
        :param max_out_group_fraction: Maximum fraction of cells from the union of the rest of each group expressing the genes, defaults to None
        :param compare_abs: If `True`, compare absolute values of log fold change with `min_fold_change`, defaults to False
        :param res_key: the key of the result of this function to be set to self.result, defaults to 'marker_genes_filtered'
        :param output: path of output_file(.csv). If None, do not generate the output file.
        """  # noqa
        if marker_genes_res_key not in self.result:
            raise Exception(
                f'{marker_genes_res_key} is not in the result, please check and run the find_marker_genes func.')

        old_result = self.result[marker_genes_res_key]
        new_result = {}
        new_result['parameters'] = copy.deepcopy(old_result['parameters'])
        new_result['parameters']['marker_genes_res_key'] = marker_genes_res_key
        new_result['pct'] = pct = old_result['pct']
        new_result['pct_rest'] = pct_rest = old_result['pct_rest']
        new_result['mean_count'] = old_result['mean_count']
        for key, res in old_result.items():
            if '.vs.' not in key:
                continue
            new_res = res.copy()
            group_name = key.split('.vs.')[0]
            if not compare_abs:
                gene_set_1 = res[res['log2fc'] < min_fold_change]['genes'].to_numpy() if min_fold_change is not None else []  # noqa
            else:
                gene_set_1 = res[res['log2fc'].abs() < min_fold_change]['genes'].to_numpy() if min_fold_change is not None else []  # noqa
            gene_set_2 = pct[pct[group_name] < min_in_group_fraction]['genes'].to_numpy() if min_in_group_fraction is not None else []  # noqa
            gene_set_3 = pct_rest[pct_rest[group_name] > max_out_group_fraction]['genes'].to_numpy() if max_out_group_fraction is not None else []  # noqa
            flag = res['genes'].isin(np.union1d(gene_set_1, np.union1d(gene_set_2, gene_set_3))).to_numpy()
            columns = new_res.columns[~new_res.columns.isin(['genes'])].to_numpy()
            new_res.loc[flag, columns] = np.nan  # noqa
            new_result[key] = new_res
        self.result[res_key] = new_result
        if output is not None:
            import natsort
            result = self.result[res_key]
            show_cols = ['genes', 'scores', 'pvalues', 'pvalues_adj', 'log2fc', 'pct', 'pct_rest']
            if self.data.genes.real_gene_name is not None:
                show_cols.insert(1, 'gene_name')
            groups = natsort.natsorted([key for key in result.keys() if '.vs.' in key])
            dat = pd.concat(
                [
                    pd.DataFrame(
                        {group.split(".vs.")[0] + "_" + key: result[group][key].values}
                    ) for group in groups for key in show_cols
                ],
                axis=1
            )
            dat.to_csv(output)

        key = 'marker_genes'
        self.reset_key_record(key, res_key)


    @logit
    def adjusted_rand_score(
        self,
        cluster_res_key_a: str,
        cluster_res_key_b: str
    ):
        """
        Calculate Adjusted Rand index between two cluster results.

        The first cluster result can be seen as true labels while the second as predicted labels.

        :param cluster_res_key_a: the key to get the first cluster result, defaults to None
        :param cluster_res_key_b: the key to get the second cluster result, defaults to None

        """
        from sklearn.metrics import adjusted_rand_score

        if cluster_res_key_a is None or cluster_res_key_a not in self.data.cells:
            raise ValueError(f"Cann't found cluster result by key {cluster_res_key_a}")
        
        if cluster_res_key_b is None or cluster_res_key_b not in self.data.cells:
            raise ValueError(f"Cann't found cluster result by key {cluster_res_key_b}")
        
        res_key = f'adjusted_rand_score_{cluster_res_key_a}_{cluster_res_key_b}'
        self.result[res_key] = adjusted_rand_score(
            self.data.cells[cluster_res_key_a],
            self.data.cells[cluster_res_key_b]
        )


    @logit
    def silhouette_score(
        self,
        cluster_res_key: str,
        used_pca_cluster_res_key: str = 'pca',
        metric: str = 'euclidean',
        sample_size: Optional[int] = None,
        random_number: int = 10086,
        use_raw: bool = True
    ):
        """
        Calculate the mean Silhouette Coefficient for a cluster result.

        :param cluster_res_key: the key to get cluster result from cells, defaults to None.
        :param used_pca_cluster_res_key: the key to get pca result used for clustering, defaults to 'pca',
                                            if it is None, use the express matrix.
        :param metric: The metric to use when calculating distance between cells/bins based on exp_matrix, defaults to 'euclidean'.
                       It must be one of the options allowed by <sklearn.metrics.pairwise.pairwise_distances>.
        :param sample_size: The size of the sample to use when computing the Silhouette Coefficient
                            on a random subset of the data, if it is None, no sampling is used.
        :param random_number: random number for selecting a subset of samples,
                              used when sample_size is not None, defaults to 10086,
                              give fixed value in multiple calls for reproducible results.
        :param use_raw: whether to use the raw express matrix when `used_pca_cluster_res_key` is None, default to True.

        """
        from sklearn.metrics import silhouette_score
        if not self.data.issparse():
            self.data.array2sparse()
        
        if cluster_res_key is None or cluster_res_key not in self.data.cells:
            raise ValueError(f"Cann't found cluster result by key {cluster_res_key}")
        
        res_key = f'silhouette_score_{cluster_res_key}'
        if used_pca_cluster_res_key is not None and used_pca_cluster_res_key in self.result:
            X = self.result[used_pca_cluster_res_key].to_numpy()
        else:
            if use_raw and self.raw is not None:
                X = self.raw.exp_matrix
            else:
                X = self.data.exp_matrix
        self.result[res_key] = silhouette_score(
            X,
            self.data.cells[cluster_res_key],
            metric=metric,
            sample_size=sample_size,
            random_state=random_number
        )


    # def scenic(self, tfs, motif, database_dir, res_key='scenic', use_raw=True, outdir=None,):
    #     """
    #
    #     :param tfs: tfs file in txt format
    #     :param motif: motif file in tbl format
    #     :param database_dir: directory containing reference database(*.feather files) from cisTarget.
    #     :param res_key: the key for getting the result from the self.result.
    #     :param use_raw: whether use the raw count express matrix for the analysis, default True.
    #     :param outdir: directory containing output files(including modules.pkl, regulons.csv, adjacencies.tsv,
    #         motifs.csv). If None, results will not be output to files.
    #
    #     :return:
    #     """
    #     from ..algorithm.scenic import scenic as cal_sce
    #     if use_raw and not self.raw:
    #         raise Exception(f'self.raw must be set if use_raw is True.')
    #     data = self.raw if use_raw else self.data
    #     modules, regulons, adjacencies, motifs, auc_mtx, regulons_df = cal_sce(data, tfs, motif, database_dir, outdir)
    #     res = {"modules": modules, "regulons": regulons, "adjacencies": adjacencies, "motifs": motifs,
    #            "auc_mtx":auc_mtx, "regulons_df": regulons_df}
    #     self.result[res_key] = res


class AnnBasedStPipeline(StPipeline):

    def __init__(self, based_ann_data: AnnData, data: AnnBasedStereoExpData):
        super().__init__(data)
        self.__based_ann_data = based_ann_data
        self.result = AnnBasedResult(based_ann_data)

    # def subset_by_hvg(self, hvg_res_key, use_raw=False, inplace=True):
    #     data: AnnBasedStereoExpData = self.data if inplace else copy.deepcopy(self.data)
    #     if hvg_res_key not in self.result:
    #         raise Exception(f'{hvg_res_key} is not in the result, please check and run the normalization func.')
    #     df = self.result[hvg_res_key]
    #     data._ann_data._inplace_subset_var(df['highly_variable'].values)
    #     return data

    # def raw_checkpoint(self):
    #     from .stereo_exp_data import AnnBasedStereoExpData
    #     if self.__based_ann_data.raw:
    #         data = AnnBasedStereoExpData("", based_ann_data=self.__based_ann_data.raw.to_adata())
    #     else:
    #         data = AnnBasedStereoExpData("", based_ann_data=copy.deepcopy(self.__based_ann_data))
    #     self.raw = data

    def raw_checkpoint(self):
        super().raw_checkpoint()
        self.data._ann_data.raw = self.data._ann_data
    
    @property
    def key_record(self):
        if 'key_record' not in self.data.adata.uns:
            self.data.adata.uns['key_record'] = self._key_record
        return self.data.adata.uns['key_record']

    @key_record.setter
    def key_record(self, key_record):
        self._key_record = key_record
        self.data.adata.uns['key_record'] = key_record
