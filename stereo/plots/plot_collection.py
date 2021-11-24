#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/31
"""
import os.path
from typing import Optional, Union, Sequence
# import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from .scatter import base_scatter, multi_scatter, marker_gene_volcano, highly_variable_genes
from stereo.config import StereoConfig

conf = StereoConfig()


class PlotCollection:
    """
    stereo plot collection

    :param data: StereoExpData object

    """

    def __init__(
            self,
            data
    ):
        self.data = data
        self.result = self.data.tl.result

    def interact_cluster(
            self,
            res_key='cluster', inline=True,
            width=700, height=500
    ):
        """
        interactive spatial scatter after clustering

        :param res_key: cluster result key
        :param inline: show in notebook
        :param width: figure width
        :param height: figure height

        """
        res = self.check_res_key(res_key)
        from .interact_plot.spatial_cluster import interact_spatial_cluster
        import pandas as pd
        df = pd.DataFrame({
            'x': self.data.position[:, 0],
            'y': self.data.position[:, 1],
            'group': np.array(res['group'])
        })
        fig = interact_spatial_cluster(df, width=width, height=height)
        if not inline:
            fig.show()
        return fig

    def highly_variable_genes(self, res_key='highly_var_genes'):
        """
        scatter of highly variable genes

        :param res_key: result key

        """
        res = self.check_res_key(res_key)
        highly_variable_genes(res)

    def marker_gene_volcano(
            self,
            group_name,
            res_key='marker_genes',
            hue_order=('down', 'normal', 'up'),
            colors=("#377EB8", "grey", "#E41A1C"),
            alpha=1, dot_size=15,
            text_genes: Optional[list] = None,
            x_label='log2(fold change)', y_label='-log10(pvalue)',
            vlines=True,
            cut_off_pvalue=0.01,
            cut_off_logFC=1
    ):
        """
        volcano of maker genes

        :param group_name: group name
        :param res_key: result key
        :param hue_order: order of gene type
        :param colors: color tuple
        :param alpha: alpha
        :param dot_size: dot size
        :param text_genes: show these genes name
        :param x_label: x label
        :param y_label: y label
        :param vlines: plot cutoff line or not
        :param cut_off_pvalue: cut off of pvalue to define gene type, pvalues < cut_off and log2fc > cut_off_logFC
        define as up genes, pvalues < cut_off and log2fc < -cut_off_logFC define as down genes
        :param cut_off_logFC: cut off of log2fc to define gene type

        :return: axes a axes object
        """
        res = self.check_res_key(res_key)[group_name]
        marker_gene_volcano(
            res,
            text_genes=text_genes,
            cut_off_pvalue=cut_off_pvalue,
            cut_off_logFC=cut_off_logFC,
            hue_order=hue_order,
            palette=colors,
            alpha=alpha, s=dot_size,
            x_label=x_label, y_label=y_label,
            vlines=vlines
        )
        # return df

    def genes_count(
            self,
            x=["total_counts", "total_counts"],
            y=["pct_counts_mt", "n_genes_by_counts"],
            ncols=2,
            dot_size=None,
            **kwargs
    ):
        """
        quality control index distribution visualization

        :param x: list of x label
        :param y: list of y label
        :param ncols: number of cols
        :param dot_size

        """
        import math
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        x = [x] if isinstance(x, str) else x
        y = [y] if isinstance(y, str) else y

        width = 20
        height = 10
        nrows = math.ceil(len(x) / ncols)
        fig = plt.figure(figsize=(width, height))
        axs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
        )
        for i, (xi, yi) in enumerate(zip(x, y)):
            draw_data = np.c_[self.data.cells.get_property(xi), self.data.cells.get_property(yi)]
            ax = fig.add_subplot(axs[i])
            base_scatter(
                draw_data[:, 0],
                draw_data[:, 1],
                hue=[0 for i in range(len(draw_data[:, 1]))],
                ax=ax,
                palette=['#808080'],
                x_label=' '.join(xi.split('_')),
                y_label=' '.join(yi.split('_')),
                dot_size=dot_size,
                color_bar=False,
                show_legend=False,
                invert_y=False,
                show_ticks=True,
                **kwargs
            )
        # return fig

    def spatial_scatter(
            self,
            cells_key: list = ["total_counts", "n_genes_by_counts"],
            ncols=2,
            dot_size=None,
            palette='stereo',
            # invert_y=True,
            **kwargs
    ):
        """
        spatial distribution of total_counts and n_genes_by_counts

        :param cells_key: specified obs key list, for example: ["total_counts", "n_genes_by_counts"]
        :param ncols: numbr of plot columns.
        :param dot_size: marker size.
        :param palette: Color theme.
        # :param invert_y: whether to invert y-axis.

        """
        from .scatter import multi_scatter

        multi_scatter(
            x=self.data.position[:, 0],
            y=self.data.position[:, 1],
            hue=[self.data.cells.get_property(key) for key in cells_key],
            x_label=['spatial1', 'spatial1'],
            y_label=['spatial2', 'spatial2'],
            title=[' '.join(i.split('_')) for i in cells_key],
            ncols=ncols,
            dot_size=dot_size,
            palette=palette,
            color_bar=True,
            **kwargs
        )

    def violin(self):
        """
        violin plot showing quality control index distribution

        :return:
        """
        from .violin import violin_distribution
        violin_distribution(self.data)

    def interact_spatial_scatter(
            self, inline=True,
            width: Optional[int] = 600, height: Optional[int] = 600,
            bgcolor='#2F2F4F',
            poly_select=False
    ):
        """
        interactive spatial distribution

        :param inline: notebook out if true else open in a new window
        :param width: width
        :param height: height
        :param bgcolor: background color

        """
        from .interact_plot.interactive_scatter import InteractiveScatter

        fig = InteractiveScatter(self.data, width=width, height=height, bgcolor=bgcolor)
        # fig = ins.interact_scatter()
        if poly_select:
            from stereo.plots.interact_plot.poly_selection import PolySelection
            fig = PolySelection(self.data, width=width, height=height, bgcolor=bgcolor)
        if not inline:
            fig.figure.show()
        return fig

    def umap(
            self,
            gene_names: Optional[list] = None,
            res_key='umap',
            cluster_key=None,
            title: Optional[Union[str, list]] = None,
            x_label: Optional[Union[str, list]] = 'umap1',
            y_label: Optional[Union[str, list]] = 'umap2',
            dot_size: int = None,
            colors: Optional[Union[str, list]] = 'stereo',
            **kwargs
    ):
        """
        plot scatter after dimension reduce

        :param gene_names: list of gene names
        :param cluster_key: dot color set by cluster if given
        :param res_key: result key
        :param title: title, it's list when plot multiple scatter
        :param x_label: x label, it's list when plot multiple scatter
        :param y_label: y label, it's list when plot multiple scatter
        :param dot_size: dot size
        :param colors: color list

        """
        res = self.check_res_key(res_key)
        self.data.sparse2array()
        if cluster_key:
            cluster_res = self.check_res_key(cluster_key)
            n = len(set(cluster_res['group']))
            return base_scatter(
                res.values[:, 0],
                res.values[:, 1],
                hue=np.array(cluster_res['group']),
                palette=conf.get_colors('stereo_30' if colors == 'stereo' else colors, n),
                title=cluster_key if title is None else title,
                x_label=x_label, y_label=y_label, dot_size=dot_size,
                color_bar=False,
                **kwargs)
        else:
            if gene_names is None:
                raise ValueError(f'gene name must be set if cluster_key is None')
            if len(gene_names) > 1:
                return multi_scatter(
                    res.values[:, 0],
                    res.values[:, 1],
                    hue=np.array(self.data.sub_by_name(gene_name=gene_names).exp_matrix).T,
                    palette=colors,
                    title=gene_names if title is None else title,
                    x_label=[x_label for i in range(len(gene_names))],
                    y_label=[y_label for i in range(len(gene_names))],
                    dot_size=dot_size,
                    color_bar=True,
                    **kwargs
                )
            else:
                return base_scatter(
                    res.values[:, 0],
                    res.values[:, 1],
                    hue=np.array(self.data.sub_by_name(gene_name=gene_names).exp_matrix[:, 0]),
                    palette=colors,
                    title=title, x_label=x_label, y_label=y_label, dot_size=dot_size,
                    color_bar=True,
                    **kwargs
                )

    def cluster_scatter(
            self,
            res_key='cluster',
            title: Optional[str] = None,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            dot_size: int = None,
            colors='stereo_30',
            invert_y: bool = True,
            **kwargs
    ):
        """
        spatial distribution ofter scatter

        :param res_key: cluster result key
        :param title: title
        :param x_label: x label
        :param y_label: y label
        :param dot_size: dot size
        :param colors: color list
        :param invert_y: whether to invert y-axis.

        """
        res = self.check_res_key(res_key)
        n = len(set(res['group']))
        ax = base_scatter(
            self.data.position[:, 0],
            self.data.position[:, 1],
            hue=np.array(res['group']),
            palette=conf.get_colors(colors, n=n),
            title=title, x_label=x_label, y_label=y_label, dot_size=dot_size, invert_y=invert_y,
            **kwargs
        )
        return ax
        # if file_path:
        #     plt.savefig(file_path)

    def marker_genes_text(
            self,
            res_key='marker_genes',
            groups: Union[str, Sequence[str]] = 'all',
            markers_num: int = 20,
            sort_key: str = 'scores',
            ascend: bool = False,
            fontsize: int = 8,
            ncols: int = 4,
            sharey: bool = True,
            **kwargs
    ):
        """
        maker genes plot

        :param res_key: marker genes result key
        :param groups: group name
        :param markers_num: top N genes to show in each cluster.
        :param sort_key: the sort key for getting top n marker genes, default `scores`.
        :param ascend: asc or dec.
        :param fontsize: font size.
        :param ncols: number of plot columns.
        :param sharey:
        :param kwargs:

        """
        from .marker_genes import marker_genes_text
        res = self.check_res_key(res_key)
        marker_genes_text(
            res,
            groups=groups,
            markers_num=markers_num,
            sort_key=sort_key,
            ascend=ascend,
            fontsize=fontsize,
            ncols=ncols,
            sharey=sharey,
            **kwargs
        )

    def marker_genes_heatmap(
            self,
            res_key='marker_genes',
            cluster_res_key='cluster',
            markers_num: int = 5,
            sort_key: str = 'scores',
            ascend: bool = False,
            show_labels: bool = True,
            show_group: bool = True,
            show_group_txt: bool = True,
            cluster_colors_array=None,
            min_value=None,
            max_value=None,
            gene_list=None,
            do_log=True
    ):
        """
        heatmap of maker genes

        :param res_key: results key
        :param cluster_res_key: cluster result key
        :param markers_num: top N maker
        :param sort_key: sorted by key
        :param ascend:
        :param show_labels:
        :param show_group:
        :param show_group_txt:
        :param cluster_colors_array:
        :param min_value:
        :param max_value:
        :param gene_list:
        :param do_log:

        """
        from .marker_genes import marker_genes_heatmap
        maker_res = self.check_res_key(res_key)
        cluster_res = self.check_res_key(cluster_res_key)
        cluster_res = cluster_res.set_index(['bins'])
        marker_genes_heatmap(
            self.data,
            cluster_res,
            maker_res,
            markers_num=markers_num,
            sort_key=sort_key,
            ascend=ascend,
            show_labels=show_labels,
            show_group=show_group,
            show_group_txt=show_group_txt,
            cluster_colors_array=cluster_colors_array,
            min_value=min_value,
            max_value=max_value,
            gene_list=gene_list,
            do_log=do_log
        )

    def check_res_key(self, res_key):
        """
        check if result exist

        :param res_key: result key

        :return: tool result
        """
        if res_key in self.data.tl.result:
            res = self.data.tl.result[res_key]
            return res
        else:
            raise ValueError(f'{res_key} result not found, please run tool before plot')

    def hotspot_local_correlations(self, res_key='spatial_hotspot', ):
        res = self.check_res_key(res_key)
        plt.rcParams['figure.figsize'] = (15.0, 12.0)
        res.plot_local_correlations()

    def hotspot_modules(
            self,
            res_key="spatial_hotspot",
            ncols=2,
            dot_size=None,
            palette='stereo',
            ** kwargs
    ):
        res = self.check_res_key(res_key)
        scores = [res.module_scores[module] for module in range(1, res.modules.max() + 1)]
        vmin = np.percentile(scores, 1)
        vmax = np.percentile(scores, 99)
        multi_scatter(
            x=res.latent.iloc[:, 0],
            y=res.latent.iloc[:, 1],
            hue=scores,
            # x_label=['spatial1', 'spatial1'],
            # y_label=['spatial2', 'spatial2'],
            title=[f"module {module}" for module in range(1, res.modules.max() + 1)],
            ncols=ncols,
            dot_size=dot_size,
            palette=palette,
            color_bar=True,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )

    def scenic_regulons(
            self,
            res_key="scenic",
            output=None,
    ):
        res = self.check_res_key(res_key)
        regulons=res["regulons"]
        auc_mtx=res["auc_mtx"]
        for tf in range(0, len(regulons)):
            scores = auc_mtx.iloc[:, tf]

            vmin = np.percentile(scores, 1)
            vmax = np.percentile(scores, 99)

            plt.scatter(x=self.data.position[:, 0],
                        y=self.data.position[:, 1],
                        s=8,
                        c=scores,
                        vmin=vmin,
                        vmax=vmax,
                        edgecolors='none'
                        )
            axes = plt.gca()
            for sp in axes.spines.values():
                sp.set_visible(False)
            plt.xticks([])
            plt.yticks([])
            plt.title('Regulon {}'.format(auc_mtx.columns[tf]))
            plt.show()

    def scenic_clustermap(
            self,
            res_key="scenic",
            output=None,
    ):
        res = self.check_res_key(res_key)
        auc_mtx = res["auc_mtx"]
        import seaborn as sns
        sns.clustermap(auc_mtx, figsize=(12, 12))
        plt.show()