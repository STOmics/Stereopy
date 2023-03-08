#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/31
"""
from random import randint
from typing import Optional, Union, Sequence
from functools import partial, wraps
# import colorcet as cc
import panel as pn
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from random import randint
from .scatter import base_scatter, multi_scatter, marker_gene_volcano, highly_variable_genes
from stereo.stereo_config import stereo_conf

pn.param.ParamMethod.loading_indicator = True

def download(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        out_path = None
        if 'out_path' in kwargs:
            out_path = kwargs['out_path']
            del kwargs['out_path']
        fig = func(*args, **kwargs)
        if out_path is None:
            pn.extension()
            file_name_input = pn.widgets.TextInput(placeholder='Enter a file name...', width=200)
            export_button = pn.widgets.Button(name='download', button_type="primary", width=100)
            def _action(_, figure):
                export_button.loading = True
                try:
                    out_path = file_name_input.value
                    if out_path is not None and len(out_path) > 0:
                        out_path = f"{out_path}_{func.__name__}.png"
                        figure.savefig(out_path, bbox_inches='tight')
                finally:
                    export_button.loading = False
            action = partial(_action, figure=fig)
            export_button.on_click(action)
            return pn.Row(file_name_input, export_button)
        else:
            fig.savefig(out_path, bbox_inches='tight')
    return wrapped

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

    def __getattr__(self, item):
        dict_attr = self.__dict__.get(item, None)
        if dict_attr:
            return dict_attr

        # start with __ may not be our algorithm function, and will cause import problem
        if item.startswith('__'):
            raise AttributeError

        new_attr = PlotBase.get_attribute_helper(item, self.data, self.result)
        if new_attr:
            self.__setattr__(item, new_attr)
            logger.info(f'register plot_func {new_attr} to {self}')
            return new_attr

        raise AttributeError(
            f'{item} not existed, please check the function name you called!'
        )

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

    def interact_annotation_cluster(
            self,
            res_cluster_key='cluster',
            res_marker_gene_key='marker_genes',
            res_key = 'annotation',
            inline=True,
            width=700, height=500
    ):
        """
        interactive spatial scatter after clustering

        :param res_cluster_key: cluster result key
        :param res_marker_gene_key: marker gene result key
        :param res_key: The key for getting the result from the self.result.
        :param inline: show in notebook
        :param width: figure width
        :param height: figure height

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
    def highly_variable_genes(self, res_key='highly_variable_genes'):
        """
        scatter of highly variable genes

        :param res_key: result key

        """
        res = self.check_res_key(res_key)
        return highly_variable_genes(res)

    @download
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
        fig = marker_gene_volcano(
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
        return fig

    @download
    def genes_count(
            self,
            x=["total_counts", "total_counts"],
            y=["pct_counts_mt", "n_genes_by_counts"],
            ncols=2,
            dot_size=None,
            out_path=None,
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
        return fig

    @download
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
            **kwargs
        )
        return fig
    
    @download
    def spatial_scatter_by_gene(
            self,
            gene_name=None,
            dot_size=None,
            palette='CET_L4',
            ignore_no_expression=False,
            **kwargs
    ):
        """draw the spatial distribution of expression quantity of the gene specified by gene_name

        :param gene_name: specify the gene you want to draw, defaults to None, if none, will select randomly.
        :param dot_size: marker size, defaults to None
        :param palette: Color theme, defaults to 'CET_L4'
        :param ignore_no_expression: whether ignore the cells no expression, defaults to False
        """

        # self.data.sparse2array()
        self.data.array2sparse()
        if gene_name is None:
            idx = randint(0, len(self.data.gene_names))
            gene_name = self.data.gene_names[idx]
        else:
            gene_names = self.data.gene_names.tolist()
            if gene_name not in gene_names:
                raise Exception(f'gene {gene_name} do not exist in expression matrix')
            idx = gene_names.index(gene_name)

        exp_data = self.data.exp_matrix[:, idx]
        if ignore_no_expression:
            nonezero_idx = np.nonzero(exp_data)
            x = self.data.position[:, 0][nonezero_idx]
            y = self.data.position[:, 1][nonezero_idx]
            hue = exp_data[nonezero_idx]
        else:
            x = self.data.position[:, 0]
            y = self.data.position[:, 1]
            hue = exp_data
        
        hue = np.squeeze(hue.toarray())
        
        if 'color_bar_reverse' in kwargs:
            color_bar_reverse = kwargs['color_bar_reverse']
            del kwargs['color_bar_reverse']
        else:
            color_bar_reverse = True
        fig = base_scatter(
            x=x,
            y=y,
            hue=hue,
            title=gene_name,
            x_label='spatial1',
            y_label='spatial2',
            dot_size=dot_size,
            palette=palette,
            color_bar=True,
            color_bar_reverse=color_bar_reverse,
            **kwargs
        )
        return fig
    
    @download
    def gaussian_smooth_scatter_by_gene(
            self,
            gene_name=None,
            dot_size=None,
            palette='CET_L4',
            color_bar_reverse=True,
            **kwargs
    ):
        """draw the spatial distribution of expression quantity of the gene specified by gene_name,
        just only for gaussian smooth, inluding the raw and smoothed.

        :param gene_name: specify the gene you want to draw, defaults to None, if none, will select randomly.
        :param dot_size: marker sizemarker size, defaults to None
        :param palette: Color theme, defaults to 'CET_L4'
        """
        self.data.tl.raw.sparse2array()
        self.data.sparse2array()
        if gene_name is None:
            idx = randint(0, len(self.data.tl.raw.gene_names) - 1)
            gene_name = self.data.gene_names[idx]
        else:
            gene_names = self.data.gene_names.tolist()
            if gene_name not in gene_names:
                raise Exception(f'gene {gene_name} do not exist in expression matrix')
            idx = gene_names.index(gene_name)

        raw_exp_data = self.data.tl.raw.exp_matrix[:, idx]
        exp_data = self.data.exp_matrix[:, idx]
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
            **kwargs
        )
        return fig

    @download
    def violin(self):
        """
        violin plot showing quality control index distribution

        :return:
        """
        from .violin import violin_distribution
        fig = violin_distribution(self.data)
        return fig

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
    
    def batches_umap(
            self,
            res_key='umap',
            title: str = 'umap between batches',
            x_label: str = 'umap1',
            y_label: str = 'umap2',
            dot_size: int = 1,
            colors: Optional[Union[str, list]] = 'stereo_30'
        ):
        import holoviews as hv
        import hvplot.pandas
        import panel as pn
        from bokeh.models import Title
        pn.extension()
        hv.extension('bokeh')
        
        assert self.data.cells.batch is not None, "there is no batches number list"
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
            width=500,
            height=500,
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
                width=200,
                height=200,
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
        if cluster_key:
            cluster_res = self.check_res_key(cluster_key)
            n = len(set(cluster_res['group']))
            return base_scatter(
                res.values[:, 0],
                res.values[:, 1],
                hue=np.array(cluster_res['group']),
                palette=stereo_conf.get_colors('stereo_30' if colors == 'stereo' else colors, n),
                title=cluster_key if title is None else title,
                x_label=x_label, y_label=y_label, dot_size=dot_size,
                color_bar=False,
                **kwargs)
        else:
            self.data.sparse2array()
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

    @download
    def cluster_scatter(
            self,
            res_key='cluster',
            group_id: str = None,
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
        group_list = np.array(res['group'])
        n = len(set(group_list))
        palette = stereo_conf.get_colors(colors, n=n)
        if group_id is not None:
            if not isinstance(group_id, str):
                group_id = str(group_id)
            group_list = np.where(group_list == group_id, group_id, 0)
            palette = ['#B3CDE3', '#FF7F00']
            kwargs['show_legend'] = False
            
        fig = base_scatter(
            self.data.position[:, 0],
            self.data.position[:, 1],
            hue=group_list,
            palette=palette,
            title=title, x_label=x_label, y_label=y_label, dot_size=dot_size, invert_y=invert_y,
            **kwargs
        )
        return fig
        # if file_path:
        #     plt.savefig(file_path)

    @download
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
        fig = marker_genes_text(
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
        return fig

    @download
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
            do_log=do_log
        )
        return fig

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

    @download
    def hotspot_local_correlations(self, res_key='spatial_hotspot', ):
        """
        visualize module scores with spatial position

        :return:
        """
        res = self.check_res_key(res_key)
        plt.rcParams['figure.figsize'] = (15.0, 12.0)
        res.plot_local_correlations()
        return plt.gcf()

    @download
    def hotspot_modules(
            self,
            res_key="spatial_hotspot",
            ncols=2,
            dot_size=None,
            palette='stereo',
            ** kwargs
    ):
        """
        plot hotspot modules

        :return:
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
            **kwargs
        )
        return fig

    def scenic_regulons(
            self,
            res_key="scenic",
            output=None,
    ):
        """
        plot scenic regulons

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
        plot scenic cluster

        :return:
        """
        res = self.check_res_key(res_key)
        auc_mtx = res["auc_mtx"]
        import seaborn as sns
        sns.clustermap(auc_mtx, figsize=(12, 12))
        plt.show()

    def cells_plotting(self, cluster_res_key='cluster', figure_size=500, fg_alpha=0.8, base_image=None):
        """plot the cells

        :param cluster_res_key: result key of clustering, defaults to 'cluster'
                color by cluster result if cluster result is not None, or by total_counts
        :param figure_size: the figure size is figure_size * figure_size, defaults to 500
        :param fg_alpha: the alpha of foreground image, between 0 and 1, defaults to 0.8
                            this is the colored image of the cells
        :param base_image: the path of the ssdna image after calibration, defaults to None
                            it will be located behide the image of the cells
        :return: figure to show
        """
        from .plot_cells import PlotCells
        pc = PlotCells(self.data, cluster_res_key=cluster_res_key, figure_size=figure_size, fg_alpha=fg_alpha, base_image=base_image)
        return pc.show()
