#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/31
"""
from typing import Optional
import colorcet as cc
import numpy as np
from .scatter import plot_scatter, plot_multi_scatter, colors


class PlotCollection:

    def __init__(
            self,
            data
    ):
        self.data = data
        self.result = self.data.tl.result

    def plot_genes_count(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        from .qc import plot_genes_count

        plot_genes_count(data=self.data, **kwargs)

    def plot_spatial_distribution(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        from .qc import plot_spatial_distribution

        plot_spatial_distribution(self.data, **kwargs)

    def plot_violin_distribution(self):
        """

        :return:
        """
        from .qc import plot_violin_distribution
        plot_violin_distribution(self.data)

    def interact_spatial_distribution(self, inline=True):
        from .interactive_scatter import InteractiveScatter

        ins = InteractiveScatter(self.data)
        fig = ins.interact_scatter()
        if not inline:
            fig.show()
        return ins

    def plot_dim_reduce(self,
                        gene_name: Optional[list] = None,
                        res_key='dim_reduce',
                        cluster_key=None,
                        **kwargs):
        """
        plot scatter after dimension reduce
        :param gene_name list of gene names
        :param cluster_key: dot color set by cluster if given
        :param res_key:
        :return:
        """
        res = self.check_res_key(res_key)
        self.data.sparse2array()
        if cluster_key:
            cluster_res = self.check_res_key(cluster_key)
            return plot_scatter(
                res.values[:, 0],
                res.values[:, 1],
                color_values=np.array(cluster_res['group']),
                color_list=cc.glasbey,
                **kwargs)
        else:
            if len(gene_name) > 1:
                return plot_multi_scatter(
                    res.values[:, 0],
                    res.values[:, 1],
                    color_values=np.array(self.data.sub_by_name(gene_name=gene_name).exp_matrix).T,
                    color_list=colors,
                    color_bar=True,
                    **kwargs
                )
            else:
                return plot_scatter(
                    res.values[:, 0],
                    res.values[:, 1],
                    color_values=np.array(self.data.sub_by_name(gene_name=gene_name).exp_matrix[:, 0]),
                    color_list=colors,
                    color_bar=True,
                    **kwargs
                )

    def plot_cluster_scatter(
            self,
            res_key='cluster',
            # file_path=None,
            **kwargs):
        """
        plot scatter after
        :param res_key: plot cluster after dimension reduce if true

        :return:
        """
        res = self.check_res_key(res_key)
        ax = plot_scatter(
            self.data.position[:, 0],
            self.data.position[:, 1],
            color_values=np.array(res['group']),
            color_list=cc.glasbey,
            **kwargs
        )
        return ax
        # if file_path:
        #     plt.savefig(file_path)

    def plot_marker_genes_text(self, res_key='marker_genes', **kwargs):
        """
        :param res_key
        :param kwargs:
        :return:
        """
        from .marker_genes import plot_marker_genes_text
        res = self.check_res_key(res_key)
        plot_marker_genes_text(res, **kwargs)

    def plot_marker_genes_heatmap(self, res_key='marker_genes', cluster_res_key='cluster', **kwargs):
        """
        :param res_key
        :param cluster_res_key
        :param kwargs:
        :return:
        """
        from .marker_genes import plot_marker_genes_heatmap
        maker_res = self.check_res_key(res_key)
        cluster_res = self.check_res_key(cluster_res_key)
        cluster_res = cluster_res.set_index(['bins'])
        plot_marker_genes_heatmap(
            self.data,
            cluster_res,
            maker_res,
            **kwargs
        )

    def check_res_key(self, res_key):
        """
        check if result exist
        :param res_key:
        :return:
        """
        if res_key in self.result:
            res = self.result[res_key]
            return res
        else:
            raise ValueError(f'{res_key} result not found, please run tool before plot')
