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

    :param data:
    :param groups:
    :param method:
    :param n_top_genes:
    :param min_disp:
    :param max_disp:
    :param min_mean:
    :param max_mean:
    :param span:
    :param n_bins:
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
                print('groups none')
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
                print('groups not none')
                batches = set(group_info)
                df = []
                gene_list = self.data.gene_names
                for batch in batches:
                    data_subset = self.data.exp_matrix[group_info == batch]
                    # Filter to genes that are in the dataset
                    # with settings.verbosity.override(Verbosity.error):
                    #
                    filt = filter_genes(data_subset, min_cells=1)[0]
                    # data_subset = data_subset[:, filt]

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
                    # hvg.index = gene_list[filt]
                    hvg.index = gene_list
                    # Add 0 values for genes that were filtered out
                    missing_hvg = pd.DataFrame(
                        np.zeros((np.sum(~filt), len(hvg.columns))),
                        columns=hvg.columns,
                    )
                    missing_hvg['highly_variable'] = missing_hvg['highly_variable'].astype(bool)
                    missing_hvg['gene'] = gene_list[~filt]
                    hvg['gene'] = gene_list
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
