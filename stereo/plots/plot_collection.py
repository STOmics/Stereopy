#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/31
"""
import os.path
from typing import Optional, Union, Sequence, Literal
from functools import partial, wraps
from natsort import natsorted
# import colorcet as cc
import panel as pn
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure
import numpy as np
from random import randint
from .scatter import base_scatter, multi_scatter, marker_gene_volcano, highly_variable_genes
from stereo.core.stereo_exp_data import StereoExpData
from stereo.stereo_config import stereo_conf
from stereo.log_manager import logger
from .plot_base import PlotBase
from .decorator import plot_scale, download

pn.param.ParamMethod.loading_indicator = True


class PlotCollection:
    """
    The plot collection for StereoExpData object.

    Parameters
    --------------
    data:
        - a StereoExpData object.

    """

    def __init__(
            self,
            data: StereoExpData
    ):
        self.data: StereoExpData = data
        self.result: dict = self.data.tl.result
        self.marker_gene_volcano = self.marker_genes_volcano

    def __getattr__(self, item):
        dict_attr = self.__dict__.get(item, None)
        if dict_attr:
            return dict_attr

        # start with __ may not be our algorithm function, and will cause import problem
        if item.startswith('__'):
            raise AttributeError

        new_attr = download(PlotBase.get_attribute_helper(item, self.data, self.result))
        if new_attr:
            self.__setattr__(item, new_attr)
            logger.info(f'register plot_func {item} to {self}')
            return new_attr

        raise AttributeError(
            f'{item} not existed, please check the function name you called!'
        )

    def interact_cluster(
            self,
            res_key: str='cluster', 
            inline: bool=True,
            width: int=700, 
            height: int=500
    ):
        """
        Interactive spatial scatter after clustering.

        :param res_key: the result key of clustering.
        :param inline: show in notebook.
        :param width: the figure width.
        :param height: the figure height.

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

    def interact_annotation_cluster(
            self,
            res_cluster_key='cluster',
            res_marker_gene_key='marker_genes',
            res_key = 'annotation',
            inline=True,
            width=700,
            height=500
    ):
        """
        Interactive spatial scatter after clustering.

        :param res_cluster_key: the result key of annotation.
        :param res_marker_gene_key: the result key of marker genes.
        :param res_key: the key for getting the result from the `self.result`.
        :param inline: show in notebook.
        :param width: the figure width.
        :param height: the figure height.

        """
        res = self.check_res_key(res_cluster_key)
        res_marker_gene = self.check_res_key(res_marker_gene_key)
        from .interact_plot.annotation_cluster import interact_spatial_cluster_annotation
        import pandas as pd
        df = pd.DataFrame({
            'x': self.data.position[:, 0],
            'y': self.data.position[:, 1],
            'bins': self.data.cell_names,
            'group': np.array(res['group'])
        })
        fig = interact_spatial_cluster_annotation(self.data, df, res_marker_gene, res_key, width=width, height=height)
        if not inline:
            fig.show()
        return fig

    @download
    def highly_variable_genes(
        self,
        res_key='highly_variable_genes',
        width=None,
        height=None
    ):
        """
        Scatter of highly variable genes

        :param res_key: the result key of highly variable genes.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        res = self.check_res_key(res_key)
        return highly_variable_genes(res, width=width, height=height)

    @download
    def marker_genes_volcano(
            self,
            group_name: str,
            res_key: str='marker_genes',
            hue_order=('down', 'normal', 'up'),
            colors: str=("#377EB8", "grey", "#E41A1C"),
            alpha: int=1, 
            dot_size: int=15,
            text_genes: Optional[list] = None,
            x_label: str='log2(fold change)', 
            y_label: str='-log10(pvalue)',
            vlines: bool=True,
            cut_off_pvalue: float=0.01,
            cut_off_logFC: int=1,
            width=None,
            height=None,
            **kwargs
    ):
        """
        Volcano plot of maker genes.

        :param group_name: the group name.
        :param res_key: the result key of marker gene.
        :param hue_order: the classification method.
        :param colors: the color set.
        :param alpha: the opacity.
        :param dot_size: the dot size.
        :param text_genes: show gene names.
        :param x_label: the x label.
        :param y_label: the y label.
        :param vlines: plot cutoff line or not.
        :param cut_off_pvalue: cut off of p-value to define gene type, p-values < cut_off and log2fc > cut_off_logFC
        define as up genes, p-values < cut_off and log2fc < -cut_off_logFC define as down genes.
        :param cut_off_logFC: cut off of log2fc to define gene type.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        res = self.check_res_key(res_key)[group_name]
        fig = marker_gene_volcano(
            res,
            text_genes=text_genes,
            cut_off_pvalue=cut_off_pvalue,
            cut_off_logFC=cut_off_logFC,
            hue_order=hue_order,
            palette=colors,
            alpha=alpha, s=dot_size,
            x_label=x_label, y_label=y_label,
            vlines=vlines,
            width=width,
            height=height,
            **kwargs
        )
        return fig

    @download
    def genes_count(
            self,
            x=["total_counts", "total_counts"],
            y=["pct_counts_mt", "n_genes_by_counts"],
            ncols: int=2,
            dot_size: int=None,
            width=None,
            height=None,
            **kwargs
    ):
        """
        Quality control index distribution visualization.

        :param x: list of x label.
        :param y: list of y label.
        :param ncols: the number of columns.
        :param dot_size: the dot size.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        import math
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        x = [x] if isinstance(x, str) else x
        y = [y] if isinstance(y, str) else y

        if width is None or height is None:
            width, height = 12, 6
        else:
            width = width / 100 if width >= 100 else 12
            height = height / 100 if height >= 100 else 6
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
        return fig

    @plot_scale
    @download
    def spatial_scatter(
            self,
            cells_key: list = ["total_counts", "n_genes_by_counts"],
            ncols: int=2,
            dot_size: int=None,
            palette: str='stereo',
            # invert_y=True,
            width: int=None,
            height: int=None,
            **kwargs
    ):
        """
        Spatial distribution of `total_counts` and `n_genes_by_counts`.

        :param cells_key: specified cells key list.
        :param ncols: the number of plot columns.
        :param dot_size: the dot size.
        :param palette: the color theme.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.
        :param show_plotting_scale: wheter to show the scale.

        """
        from .scatter import multi_scatter

        fig = multi_scatter(
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
            width=width,
            height=height,
            # show_plotting_scale=show_plotting_scale,
            # data_resolution=self.data.attr['resolution'] if self.data.bin_type == 'cell_bins' else self.data.attr['resolution'] * self.data.bin_size,
            **kwargs
        )
        return fig
    
    @plot_scale
    @download
    def spatial_scatter_by_gene(
            self,
            gene_name: Union[str, list, np.ndarray]=None,
            dot_size: int=None,
            palette: str='CET_L4',
            color_bar_reverse: bool=True,
            width: int=None,
            height: int=None,
            **kwargs
    ):
        """Draw the spatial distribution of expression quantity of the gene specified by gene names.

        :param gene_name: a gene or a list of genes you want to show.
        :param dot_size: the dot size, defaults to `None`.
        :param palette: the color theme, defaults to `'CET_L4'`.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.
        :param show_plotting_scale: wheter to show the scale.

        """

        self.data.array2sparse()
        if gene_name is None:

            raise Exception("The gene_name can not be None.")

        if isinstance(gene_name, str):
            gene_name = [gene_name]
        gene_idx = [np.argwhere(self.data.gene_names == gn)[0][0] for gn in gene_name]
        hue = self.data.exp_matrix[:, gene_idx].T

        fig = multi_scatter(
            x=self.data.position[:, 0],
            y=self.data.position[:, 1],
            hue=hue,
            x_label=['spatial1'] * len(gene_name),
            y_label=['spatial2'] * len(gene_name),
            title=gene_name,
            ncols=2,
            dot_size=dot_size,
            palette=palette,
            color_bar=True,
            color_bar_reverse=color_bar_reverse,
            width=width,
            height=height,
            **kwargs
        )
        
        return fig
    
    @plot_scale
    @download
    def gaussian_smooth_scatter_by_gene(
            self,
            gene_name: str=None,
            dot_size: int=None,
            palette: str='CET_L4',
            color_bar_reverse: bool=True,
            width: int=None,
            height: int=None,
            **kwargs
    ):
        """Draw the spatial distribution of expression quantity of the gene specified by gene names,
        just only for Gaussian smoothing, inluding the raw and smoothed.

        :param gene_name: specify the gene you want to draw, if `None` by default, will select randomly.
        :param dot_size: marker sizemarker size, defaults to `None`.
        :param palette: Color theme, defaults to `'CET_L4'`.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        # self.data.tl.raw.sparse2array()
        # self.data.sparse2array()
        if gene_name is None:
            idx = randint(0, len(self.data.tl.raw.gene_names) - 1)
            gene_name = self.data.gene_names[idx]
        else:
            # gene_names = self.data.gene_names.tolist()
            if gene_name not in self.data.gene_names:
                raise Exception(f'gene {gene_name} do not exist in expression matrix')
            # idx = gene_names.index(gene_name)
            idx = np.argwhere(self.data.gene_names == gene_name)[0][0]

        raw_exp_data = self.data.tl.raw.exp_matrix[:, idx].T
        exp_data = self.data.exp_matrix[:, idx].T
        hue_list = [raw_exp_data, exp_data]
        titles = [f'{gene_name}(raw)', f'{gene_name}(smoothed)']

        fig = multi_scatter(
            x=self.data.position[:, 0],
            y=self.data.position[:, 1],
            hue=hue_list,
            x_label=['spatial1', 'spatial1'],
            y_label=['spatial2', 'spatial2'],
            title=titles,
            ncols=2,
            dot_size=dot_size,
            palette=palette,
            color_bar=True,
            color_bar_reverse=color_bar_reverse,
            width=width,
            height=height,
            **kwargs
        )
        return fig

    @download
    def violin(self, width: int=None, height: int=None):
        """
        Violin plot to show index distribution of quality control.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        from .violin import violin_distribution
        fig = violin_distribution(self.data, width=width, height=height)
        return fig

    def interact_spatial_scatter(
            self, 
            inline: bool=True,
            width: Optional[int] = 600, 
            height: Optional[int] = 600,
            bgcolor: str='#2F2F4F',
            poly_select: bool=False
    ):
        """
        Interactive spatial distribution.

        :param inline: show in notebook.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.
        :param bgcolor: set background color.

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
    
    def batches_umap(
            self,
            res_key='umap',
            title: str = 'umap between batches',
            x_label: str = 'umap1',
            y_label: str = 'umap2',
            dot_size: int = 1,
            colors: Optional[Union[str, list]] = 'stereo_30',
            width: int = None,
            height: int = None
        ):
        import holoviews as hv
        import hvplot.pandas
        import panel as pn
        from bokeh.models import Title
        pn.extension()
        hv.extension('bokeh')
        
        assert self.data.cells.batch is not None, "there is no batches number list"
        if width is None or height is None:
            main_width, main_height = 500, 500
            sub_width, sub_height = 200, 200
        else:
            main_width = width
            main_height = height
            sub_width = np.ceil(width * 0.4).astype(np.int32)
            sub_height = np.ceil(height * 0.4).astype(np.int32)
        
        umap_res = self.check_res_key(res_key)
        umap_res = umap_res.rename(columns={0: 'x', 1: 'y'})
        umap_res['batch'] = self.data.cells.batch.astype(np.uint16)
        batch_number_unique = np.unique(umap_res['batch'])
        batch_count = len(batch_number_unique)
        cmap = stereo_conf.get_colors(colors, batch_count)
        fig_all = umap_res.hvplot.scatter(
            x='x', y='y',
            c='batch', cmap=cmap, cnorm='eq_hist',
            # datashade=True, dynspread=True
        ).opts(
            width=main_width,
            height=main_height,
            invert_yaxis=True,
            xlabel=x_label,
            ylabel=y_label, 
            size=dot_size,
            toolbar='disable',
            colorbar=False,
        )
        bfig_all = hv.render(fig_all)
        bfig_all.axis.major_tick_line_alpha = 0
        bfig_all.axis.minor_tick_line_alpha = 0
        bfig_all.axis.major_label_text_alpha = 0
        bfig_all.axis.axis_line_alpha = 0
        bfig_all.title = Title(text='all batches', align='center')
        bfig_batches = []
        pn_rows = []
        for i, bn, c in zip(range(batch_count), batch_number_unique, cmap):
            sub_umap_res = umap_res[umap_res.batch == bn]
            fig = sub_umap_res.hvplot.scatter(
                x='x', y='y',
                c='batch', color=c, cnorm='eq_hist',
                # datashade=True, dynspread=True
            ).opts(
                width=sub_width,
                height=sub_height,
                xaxis=None,
                yaxis=None,
                invert_yaxis=True,
                size=(dot_size / 3),
                toolbar='disable',
                colorbar=False,
            )
            bfig = hv.render(fig)
            bn = str(bn)
            bfig.title = Title(text=f'sn: {self.data.sn[bn]}', align='center')
            bfig_batches.append(bfig)
            if ((i + 1) % 2) == 0 or i == (batch_count - 1):
                pn_rows.append(pn.Row(*bfig_batches))
                bfig_batches.clear()

        return pn.Column(
            f"\n# {title}",
            pn.Row(
                pn.Column(bfig_all),
                pn.Column(*pn_rows)
            )
        )

    @download
    def umap(
            self,
            gene_names: Optional[Union[list, np.ndarray, str]] = None,
            res_key: str='umap',
            cluster_key=None,
            title: Optional[Union[str, list]] = None,
            x_label: Optional[Union[str, list]] = 'umap1',
            y_label: Optional[Union[str, list]] = 'umap2',
            dot_size: int = None,
            colors: Optional[Union[str, list]] = 'stereo',
            width: int = None,
            height: int = None,
            **kwargs
    ):
        """
        Scatter plot of UMAP after reducing dimensionalities.

        :param gene_names: the list of gene names.
        :param cluster_key: the result key of clustering.
        :param res_key: the result key of UMAP.
        :param title: the plot titles.
        :param x_label: the x label.
        :param y_label: the y label.
        :param dot_size: the dot size.
        :param colors: the color list.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        res = self.check_res_key(res_key)
        if cluster_key:
            cluster_res = self.check_res_key(cluster_key)
            n = len(set(cluster_res['group']))
            return base_scatter(
                res.values[:, 0],
                res.values[:, 1],
                # hue=np.array(cluster_res['group']),
                hue=cluster_res['group'],
                palette=stereo_conf.get_colors('stereo_30' if colors == 'stereo' else colors, n),
                title=cluster_key if title is None else title,
                x_label=x_label, y_label=y_label, dot_size=dot_size,
                color_bar=False,
                width=width, height=height,
                **kwargs)
        else:
            # self.data.sparse2array()
            self.data.array2sparse()
            if gene_names is None:
                raise ValueError(f'gene name must be set if cluster_key is None')
            if isinstance(gene_names, str):
                gene_names = [gene_names]
            return multi_scatter(
                res.values[:, 0],
                res.values[:, 1],
                # hue=np.array(self.data.sub_by_name(gene_name=gene_names).exp_matrix).T,
                hue=self.data.sub_exp_matrix_by_name(gene_name=gene_names).T,
                palette=colors,
                title=gene_names if title is None else title,
                x_label=[x_label for i in range(len(gene_names))],
                y_label=[y_label for i in range(len(gene_names))],
                dot_size=dot_size,
                color_bar=True,
                width=width,
                height=height,
                **kwargs
            )

    @plot_scale
    @download
    def cluster_scatter(
            self,
            res_key='cluster',
            groups: Union[str, list, np.ndarray] = None,
            title: Optional[str] = None,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            dot_size: int = None,
            colors='stereo_30',
            invert_y: bool = True,
            hue_order: set=None,
            width: int=None,
            height: int=None,
            **kwargs
    ):
        """
        Spatial distribution ofter scatter.

        :param res_key: cluster result key.
        :param title: the plot title.
        :param x_label: x label.
        :param y_label: y label.
        :param dot_size: dot size.
        :param colors: color list.
        :param invert_y: whether to invert y-axis.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        :return: Spatial scatter distribution of clusters.
        """
        res = self.check_res_key(res_key)
        group_list = res['group'].to_numpy()
        n = np.unique(group_list).size
        palette = stereo_conf.get_colors(colors, n=n)
        if groups is not None:
            # if not isinstance(groups, str):
            #     groups = str(groups)
            if isinstance(groups, str):
                groups = [groups]
            isin = np.in1d(group_list, groups)
            if not np.all(isin):
                group_list[~isin] = 'others'
                n = np.unique(group_list).size
                palette = palette[0:n - 1] + ['#828282']
                hue_order =  natsorted(np.unique(group_list[isin])) + ['others']
                # palette = ['#B3CDE3', '#FF7F00']
                # kwargs['show_legend'] = False

        
            
        fig = base_scatter(
            self.data.position[:, 0],
            self.data.position[:, 1],
            hue=group_list,
            palette=palette,
            title=title, x_label=x_label, y_label=y_label, dot_size=dot_size, invert_y=invert_y,
            hue_order=hue_order,
            width=width, height=height,
            **kwargs
        )
        return fig
        # if file_path:
        #     plt.savefig(file_path)

    @download
    def marker_genes_text(
            self,
            res_key: str='marker_genes',
            groups: Union[str, Sequence[str]] = 'all',
            markers_num: int = 20,
            sort_key: str = 'scores',
            ascend: bool = False,
            fontsize: int = 8,
            ncols: int = 4,
            sharey: bool = True,
            width: int = None,
            height: int = None,
            **kwargs
    ):
        """
        Scatter plot of maker genes. 

        :param res_key: the result key of marker genes.
        :param groups: the group names.
        :param markers_num: top N genes to show in each cluster.
        :param sort_key: the sort key for getting top N marker genes, default `'scores'`.
        :param ascend: whether to sort by ascending.
        :param fontsize: the font size.
        :param ncols: number of plot columns.
        :param sharey: share scale or not.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        from .marker_genes import marker_genes_text
        res = self.check_res_key(res_key)
        fig = marker_genes_text(
            res,
            groups=groups,
            markers_num=markers_num,
            sort_key=sort_key,
            ascend=ascend,
            fontsize=fontsize,
            ncols=ncols,
            sharey=sharey,
            width=width,
            height=height,
            **kwargs
        )
        return fig

    @download
    def marker_genes_heatmap(
            self,
            res_key: str='marker_genes',
            cluster_res_key: str='cluster',
            markers_num: int = 5,
            sort_key: str = 'scores',
            ascend: bool = False,
            show_labels: bool = True,
            show_group: bool = True,
            show_group_txt: bool = True,
            cluster_colors_array: bool = None,
            min_value: int = None,
            max_value: int = None,
            gene_list: list = None,
            do_log: bool = True,
            width: int = None,
            height: int = None
    ):
        """
        Heatmap plot of maker genes.

        :param res_key: the result key of marker genes.
        :param cluster_res_key: the result key of clustering.
        :param markers_num: top N maker genes.
        :param sort_key: sorted by which key.
        :param ascend: whether to sort by ascending.
        :param show_labels: show labels or not.
        :param show_group: show group or not.
        :param show_group_txt: show group names or not.
        :param cluster_colors_array: whether to show color scale.
        :param min_value: minimum value of scale.
        :param max_value: maximum value of scale.
        :param gene_list: gene name list.
        :param do_log: perform normalization if log1p before plotting, or not.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        from .marker_genes import marker_genes_heatmap
        maker_res = self.check_res_key(res_key)
        cluster_res_key = maker_res['parameters']['cluster_res_key']        
        cluster_res = self.check_res_key(cluster_res_key)
        cluster_res = cluster_res.set_index(['bins'])
        fig = marker_genes_heatmap(
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
            do_log=do_log,
            width=width,
            height=height
        )
        return fig

    @download
    def marker_genes_scatter(
        self,
        res_key: str = 'marker_genes',
        markers_num: int = 10,
        genes: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        values_to_plot: Optional[
            Literal[
                'scores',
                'logfoldchanges',
                'pvalues',
                'pvalues_adj',
                'log10_pvalues',
                'log10_pvalues_adj',
            ]
        ] = None,
        sort_by: Literal[
            'scores',
            'logfoldchanges',
            'pvalues',
            'pvalues_adj'
        ] = 'scores',
        width: int = None,
        height: int = None
    ):
        """Scatter of marker genes

        :param res_key: results key, defaults to 'marker_genes'.
        :param markers_num: top N makers, defaults to 10.
        :param genes: name of genes which would be shown on plot, markers_num is ignored if it is set, defaults to None.
        :param groups: cell types which would be shown on plot, all cell types would be shown if set it to None, defaults to None.
        :param values_to_plot: specify the value which color the plot, the mean expression in group would be set if set it to None defaults to None.
                        available values include: [scores, logfoldchanges, pvalues, pvalues_adj, log10_pvalues, log10_pvalues_adj].
        :param sort_by: specify the value which sort by when select top N markers, defaults to 'scores'
                        available values include: [scores, logfoldchanges, pvalues, pvalues_adj].
        """
        from .marker_genes import MarkerGenesScatterPlot
        marker_genes_res = self.check_res_key(res_key)
        mgsp = MarkerGenesScatterPlot(self.data, marker_genes_res)
        return mgsp.plot_scatter(
            markers_num=markers_num,
            genes=genes,
            groups=groups,
            values_to_plot=values_to_plot,
            sort_by=sort_by,
            width=width,
            height=height
        )

    def check_res_key(self, res_key):
        """
        Check if result exist

        :param res_key: result key

        :return: tool result
        """
        if res_key in self.data.tl.result:
            res = self.data.tl.result[res_key]
            return res
        else:
            raise ValueError(f'{res_key} result not found, please run tool before plot')

    @download
    def hotspot_local_correlations(
        self,
        res_key: str='spatial_hotspot',
        width: int=None,
        height: int=None
    ):
        """
        Visualize module scores with spatial position.

        :param res_key: the result key of spatial hotspot.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.
        """
        res = self.check_res_key(res_key)
        if width is None or height is None:
            width, height = 15, 12
        else:
            width = width / 100 if width >= 100 else 15
            height = height / 100 if height >= 100 else 12
        res.plot_local_correlations()
        fig = plt.gcf()
        fig.set_size_inches(width, height)
        return fig

    @download
    def hotspot_modules(
            self,
            res_key: str="spatial_hotspot",
            ncols: int=2,
            dot_size: int=None,
            palette: str='stereo',
            width: str=None,
            height: str=None,
            **kwargs
    ):
        """
        Plot hotspot modules

        :param ncols: the number of columns.
        :param dot_size: the dot size.
        :param res_key: the result key of spatial hotspot.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        """
        res = self.check_res_key(res_key)
        scores = [res.module_scores[module] for module in range(1, res.modules.max() + 1)]
        vmin = np.percentile(scores, 1)
        vmax = np.percentile(scores, 99)
        fig = multi_scatter(
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
            width=width,
            height=height,
            **kwargs
        )
        return fig

    def scenic_regulons(
            self,
            res_key="scenic",
            output=None,
    ):
        """
        Plot scenic regulons

        :return:
        """
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
        """
        Plot scenic cluster

        :return:
        """
        res = self.check_res_key(res_key)
        auc_mtx = res["auc_mtx"]
        import seaborn as sns
        sns.clustermap(auc_mtx, figsize=(12, 12))
        plt.show()

    def cells_plotting(self, cluster_res_key='cluster', bgcolor='#2F2F4F', width=None, height=None, fg_alpha=0.5, base_image=None):
        """Plot the cells.

        :param cluster_res_key: result key of clustering, defaults to `'cluster'`
                color by cluster result if cluster result is not None, or by `total_counts`.
        :param figure_size: the figure size is figure_size * figure_size, defaults to 500.
        :param fg_alpha: the alpha of foreground image, between 0 and 1, defaults to 0.8
                            this is the colored image of the cells.
        :param base_image: the path of the ssdna image after calibration, defaults to None
                            it will be located behide the image of the cells.
        :return: Cells distribution figure.
        """
        from .plot_cells import PlotCells
        pc = PlotCells(self.data, cluster_res_key=cluster_res_key, bgcolor=bgcolor, width=width, height=height, fg_alpha=fg_alpha, base_image=base_image)
        return pc.show()
