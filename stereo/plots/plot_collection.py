#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/04/14
"""
from typing import Optional
from .qc import plot_genes_count, plot_spatial_distribution, plot_violin_distribution
from .interactive_scatter import InteractiveScatter
from .marker_genes import plot_marker_genes_heatmap, plot_marker_genes_text
from .scatter import plot_scatter, plot_multi_scatter, colors, plt
import colorcet as cc
import numpy as np


class PlotCollection:
    def __init__(
            self,
            data
    ):
        self.data = data
        # self.result = self.data.tl.result
        self.result = dict()

    def plot_genes_count(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        plot_genes_count(data=self.data, **kwargs)

    def plot_spatial_distribution(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        plot_spatial_distribution(self.data, **kwargs)

    def plot_violin_distribution(self):
        """

        :return:
        """
        plot_violin_distribution(self.data)

    def interact_spatial_distribution(self, inline=True):
        ins = InteractiveScatter(self.data)
        fig = ins.interact_scatter()
        if not inline:
            fig.show()
        return ins

    def plot_dim_reduce(self, gene_name: Optional[list], file_path=None, **kwargs):
        """
        plot scatter after dimension reduce
        :param gene_name list of gene names
        :param file_path:
        :return:
        """
        # from scipy.sparse import issparse
        # if issparse(self.data.exp_matrix):
        #     self.data.exp_matrix = self.data.exp_matrix.toarray()
        if 'dim_reduce' not in self.result:
            raise ValueError(f'can not found dimension reduce result, please run stereo.tool.DimReduce before plot')
        self.data.sparse2array()
        if len(gene_name) > 1:
            plot_multi_scatter(self.result['dim_reduce'].values[:, 0], self.result['dim_reduce'].values[:, 1],
                               color_values=np.array(self.data.sub_by_name(gene_name=gene_name).exp_matrix).T,
                               color_list=colors, color_bar=True, **kwargs
                               )
        else:
            plot_scatter(self.result['dim_reduce'].values[:, 0], self.result['dim_reduce'].values[:, 1],
                         color_values=np.array(self.data.sub_by_name(gene_name=gene_name).exp_matrix[:, 0]),
                         color_list=colors, color_bar=True, **kwargs)
        if file_path:
            plt.savefig(file_path)

    def plot_cluster_scatter(self, plot_dim_reduce=False, file_path=None, **kwargs):
        """
        plot scatter after
        :param plot_dim_reduce: plot cluster after dimension reduce if true
        :param file_path:

        :return:
        """
        if 'cluster' not in self.result:
            raise ValueError(f'can not found cluster result, please run stereo.tool.Cluster before plot')
        if plot_dim_reduce:
            if 'dim_reduce' not in self.result:
                raise ValueError(f'can not found dimension reduce result, please run stereo.tool.DimReduce before plot')
            plot_scatter(self.result['dim_reduce'].values[:, 0], self.result['dim_reduce'].values[:, 1],
                         color_values=np.array(self.result['cluster']['cluster']), color_list=cc.glasbey, **kwargs)
        else:
            plot_scatter(self.data.position[:, 0], self.data.position[:, 1],
                         color_values=np.array(self.result['cluster']['cluster']),
                         color_list=cc.glasbey, **kwargs)
        if file_path:
            plt.savefig(file_path)

    def plot_marker_genes_text(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        if 'gene_maker' not in self.result:
            raise ValueError(f'can not found gene maker result, please run stereo.tool.GeneMaker before plot')
        plot_marker_genes_text(self.result['gene_maker'], **kwargs)

    def plot_marker_genes_heatmap(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        if 'gene_maker' not in self.result:
            raise ValueError(f'can not found gene maker result, please run stereo.tool.GeneMaker before plot')
        plot_marker_genes_heatmap(
            self.data,
            self.result['cluster'],
            self.result['gene_maker'],
            **kwargs
        )
