#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua
@time:2021/08/12
"""
from ..core.tool_base import ToolBase
from typing import Optional
import numpy as np
import pandas as pd
from ..algorithm.highly_variable_genes import highly_variable_genes_seurat_v3, highly_variable_genes_single_batch
from ..utils.hvg_utils import filter_genes


class HighlyVariableGenes(ToolBase):
    """
    Annotate highly variable genes. reference by scanpy

    :param data: stereo exp data object
    :param groups
    If specified, highly-variable genes are selected within each batch separately and merged.
    This simple process avoids the selection of batch-specific genes and acts as a
    lightweight batch correction method. For all flavors, genes are first sorted
    by how many batches they are a HVG. For dispersion-based flavors ties are broken
    by normalized dispersion. If `flavor = 'seurat_v3'`, ties are broken by the median
    (across batches) rank based on within-batch normalized variance.
    :param n_top_genes
        Number of highly-variable genes to keep. Mandatory if `flavor='seurat_v3'`.
    :param min_mean
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    :param max_mean
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    :param min_disp
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    :param max_disp
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    s:param pan
        The fraction of the data (cells) used when estimating the variance in the loess
        model fit if `flavor='seurat_v3'`.
    :param n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    :param method
        Choose the flavor for identifying highly variable genes. For the dispersion
        based methods in their default workflows, Seurat passes the cutoffs whereas
        Cell Ranger passes `n_top_genes`.

    :return:
    """
    def __init__(
            self,
            data,
            groups=None,
            method: Optional[str] = 'seurat',
            n_top_genes: Optional[int] = 2000,
            min_disp: Optional[float] = 0.5,
            max_disp: Optional[float] = np.inf,
            min_mean: Optional[float] = 0.0125,
            max_mean: Optional[float] = 3,
            span: Optional[float] = 0.3,
            n_bins: int = 20,
    ):
        self.n_top_genes = n_top_genes
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.span = span
        self.n_bins = n_bins
        super(HighlyVariableGenes, self).__init__(data=data,  groups=groups,  method=method)

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['seurat', 'cell_ranger', 'seurat_v3']
        self._method_check(method, m_range)

    def fit(self):
        # group_info = None if self.groups is None else np.array(self.groups['group'])
        group_info = None if self.groups is None else self.groups['group'].astype('category')
        if self.method == 'seurat_v3':
            df = highly_variable_genes_seurat_v3(
                self.data.exp_matrix,
                n_top_genes=self.n_top_genes,
                span=self.span,
                batch_info=group_info
            )
            df.index = self.data.gene_names
        else:
            if self.groups is None:
                # print('groups none')
                df = highly_variable_genes_single_batch(
                    self.data.exp_matrix,
                    min_disp=self.min_disp,
                    max_disp=self.max_disp,
                    min_mean=self.min_mean,
                    max_mean=self.max_mean,
                    n_top_genes=self.n_top_genes,
                    n_bins=self.n_bins,
                    method=self.method
                )
                df.index = self.data.gene_names
            else:
                # print('groups not none')
                batches = set(group_info)
                df = []
                gene_list = self.data.gene_names
                for batch in batches:
                    data_subset = self.data.exp_matrix[group_info == batch]
                    # Filter to genes that are in the dataset
                    # with settings.verbosity.override(Verbosity.error):
                    #
                    filt = filter_genes(data_subset, min_cells=1)[0]
                    data_subset = data_subset[:, filt]

                    hvg = highly_variable_genes_single_batch(
                        data_subset,
                        min_disp=self.min_disp,
                        max_disp=self.max_disp,
                        min_mean=self.min_mean,
                        max_mean=self.max_mean,
                        n_top_genes=self.n_top_genes,
                        n_bins=self.n_bins,
                        method=self.method,
                    )
                    hvg.index = gene_list[filt]
                    # hvg.index = gene_list
                    # Add 0 values for genes that were filtered out
                    missing_hvg = pd.DataFrame(
                        np.zeros((np.sum(~filt), len(hvg.columns))),
                        columns=hvg.columns,
                    )
                    missing_hvg['highly_variable'] = missing_hvg['highly_variable'].astype(bool)
                    missing_hvg['gene'] = gene_list[~filt]
                    hvg['gene'] = gene_list[filt]
                    hvg = hvg.append(missing_hvg, ignore_index=True)

                    # Order as before filtering
                    idxs = np.concatenate((np.where(filt)[0], np.where(~filt)[0]))
                    hvg = hvg.loc[np.argsort(idxs)]

                    df.append(hvg)

                df = pd.concat(df, axis=0)
                df['highly_variable'] = df['highly_variable'].astype(int)
                df = df.groupby('gene').agg(
                    dict(
                        means=np.nanmean,
                        dispersions=np.nanmean,
                        dispersions_norm=np.nanmean,
                        highly_variable=np.nansum,
                    )
                )
                df.rename(
                    columns=dict(highly_variable='highly_variable_nbatches'), inplace=True
                )
                df['highly_variable_intersection'] = df['highly_variable_nbatches'] == len(
                    batches
                )

                if self.n_top_genes is not None:
                    # sort genes by how often they selected as hvg within each batch and
                    # break ties with normalized dispersion across batches
                    df.sort_values(
                        ['highly_variable_nbatches', 'dispersions_norm'],
                        ascending=False,
                        na_position='last',
                        inplace=True,
                    )
                    df['highly_variable'] = False
                    df.highly_variable.iloc[:self.n_top_genes] = True
                    df = df.loc[self.data.gene_names]
                else:
                    df = df.loc[self.data.gene_names]
                    dispersion_norm = df.dispersions_norm.values
                    dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
                    gene_subset = np.logical_and.reduce(
                        (
                            df.means > self.min_mean,
                            df.means < self.max_mean,
                            df.dispersions_norm > self.min_disp,
                            df.dispersions_norm < self.max_disp,
                        )
                    )
                    df['highly_variable'] = gene_subset
        self.result = df
        self.data.genes.hvgs = np.array(df['highly_variable'])
