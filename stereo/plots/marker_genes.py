#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:marker_genes.py
@time:2021/03/31
"""
from collections import OrderedDict
from typing import (
    Optional,
    Sequence,
    Union,
    Literal
)

import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.sparse import spmatrix

from ._plot_basic.heatmap_plt import (
    heatmap,
    plot_categories_as_colorblocks,
    plot_gene_groups_brackets
)
from ..core.stereo_exp_data import StereoExpData
from ..utils import data_helper
from ..utils import pipeline_utils


def marker_genes_text(
        marker_res,
        groups: Union[str, Sequence[str]] = 'all',
        markers_num: int = 20,
        sort_key: str = 'scores',
        ascend: bool = False,
        fontsize: int = 8,
        ncols: int = 4,
        sharey: bool = True,
        ax: Optional[Axes] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
):  # scatter plot, marker gene significance map, gravel-like map
    """
    marker gene scatter visualization.

    :param marker_res: the StereoResult of FindMarkers tool.
    :param groups: list of cluster ids or 'all' clusters, a cluster equal a group.
    :param markers_num: top N genes to show in each cluster.
    :param sort_key: the sort key for getting top n marker genes, default `scores`.
    :param ascend: asc or dec.
    :param fontsize: font size.
    :param ncols: number of plot columns.
    :param sharey: share scale or not
    :param ax: axes object
    :param width: the figure width.
    :param height: the figure height.
    :param kwargs: other args for plot.

    """
    # Adjust image panel/grid related parameters
    if 'n_panels_per_row' in kwargs:
        n_panels_per_row = kwargs['n_panels_per_row']
    else:
        n_panels_per_row = ncols
    if groups == 'all':
        group_names = [key for key in marker_res.keys() if '.vs.' in key]
    else:
        group_names = [groups] if isinstance(groups, str) else groups
    group_names = natsort.natsorted(group_names)
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)
    # initialize image
    if width is None or height is None:
        width = 10 * n_panels_x
        height = 10 * n_panels_y
    else:
        width = width / 100 if width >= 100 else 10 * n_panels_x
        height = height / 100 if height >= 100 else 10 * n_panels_y
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ymin = np.Inf
    ymax = -np.Inf
    for count, group_name in enumerate(group_names):
        result = data_helper.get_top_marker(g_name=group_name, marker_res=marker_res, sort_key=sort_key,
                                            ascend=ascend, top_n=markers_num)
        gene_names = result.genes.values
        scores = result.scores.values
        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(scores)) if scores.size > 0 else ymin
            ymax = max(ymax, np.max(scores)) if scores.size > 0 else ymax

            if ax0 is None:
                ax = fig.add_subplot(gs[count])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[count], sharey=ax0)
        else:
            ymin = np.min(scores) if scores.size > 0 else ymin
            ymax = np.max(scores) if scores.size > 0 else ymax
            ymax += 0.3 * (ymax - ymin)

            ax = fig.add_subplot(gs[count])
            if (not np.isinf(ymin)) and (not np.isinf(ymax)):
                ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, markers_num - 0.1)

        # Making labels
        for ig, gene_name in enumerate(gene_names):
            ax.text(
                ig,
                scores[ig],
                gene_name,
                rotation='vertical',
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=fontsize,
            )

        ax.set_title(group_name)
        if count >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel('ranking')

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax.set_ylabel(sort_key)

    if (sharey is True) and (not np.isinf(ymin)) and (not np.isinf(ymax)):
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)
    return fig


def make_draw_df(
        data: StereoExpData,
        group: pd.DataFrame,
        marker_res: dict,
        top_genes: int = 8,
        sort_key: str = 'scores',
        ascend: bool = False,
        gene_list: Optional[list] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
):
    gene_names_dict = get_groups_marker(marker_res, top_genes, sort_key, ascend, gene_list)
    gene_names = list()
    gene_group_labels = list()
    gene_group_positions = list()
    start = 0
    for label, gene_list in gene_names_dict.items():
        if isinstance(gene_list, str):
            gene_list = [gene_list]
        if len(gene_list) == 0:
            continue
        gene_names.extend(list(gene_list))
        gene_group_labels.append(label)
        gene_group_positions.append((start, start + len(gene_list) - 1))
        start += len(gene_list)
    if marker_res['parameters']['use_raw']:
        draw_df = data_helper.exp_matrix2df(data.raw, gene_name=np.array(gene_names))
    else:
        draw_df = data_helper.exp_matrix2df(data, gene_name=np.array(gene_names))
    draw_df = pd.concat([draw_df, group], axis=1)
    draw_df['group'] = draw_df['group'].astype('category')
    draw_df = draw_df.set_index(['group'])
    draw_df = draw_df.sort_index()
    if min_value is not None or max_value is not None:
        draw_df.clip(lower=min_value, upper=max_value, inplace=True)
    return draw_df, gene_group_labels, gene_group_positions


def get_groups_marker(
        marker_res: dict,
        top_genes: int = 8,
        sort_key: str = 'scores',
        ascend: bool = False,
        gene_list: Optional[list] = None
):
    groups = [key for key in marker_res.keys() if '.vs.' in key]
    groups = natsort.natsorted(groups)
    groups_genes = OrderedDict()
    if gene_list is not None:
        for g in groups:
            groups_genes[g] = gene_list
        return groups_genes
    for g in groups:
        res = data_helper.get_top_marker(g, marker_res, sort_key, ascend, top_genes)
        genes = res['genes'].values
        groups_genes[g] = genes
    return groups_genes


def plot_heatmap(
        df,
        show_xaxis=True,
        show_group=True,
        show_group_txt=True,
        group_position=None,
        group_labels=None,
        cluster_colors_array=None,
        width=None,
        height=None,
        **kwargs
):
    """
    heatmap

    :param df: the dataframe object. the index of df is group info if show group is True.
    :param show_labels: whether to show the labels of the heatmap plot.
    :param show_group: show the group info on the left of the heatmap.
    :param show_group_txt: show the group info on the top of the heatmap.
    :param group_position: the position of group txt, must to be set if show_group_txt is True.
    :param group_labels: the label of group, must to be set if show_group_txt is True.
    :param cluster_colors_array: he list of colors in the color block on the left of heatmap.
    :param kwargs: other args for plot.

    """
    kwargs.setdefault("colorbar_width", 0.2)
    colorbar_width = kwargs.get("colorbar_width")
    cluster_block_width = kwargs.setdefault("cluster_block_width", 0.2) if show_group else 0
    if width is None or height is None:
        height = 10
        if show_xaxis:
            heatmap_width = len(df.columns) * 0.3
        else:
            heatmap_width = 8
        width = heatmap_width + cluster_block_width
    else:
        width = width / 100 if width >= 100 else 10
        height = height / 100 if height >= 100 else 10
        heatmap_width = width - cluster_block_width
    if show_group_txt:
        # add some space in case 'brackets' want to be plotted on top of the image
        height_ratios = [0.15, height]
    else:
        height_ratios = [0, height]
    width_ratios = [
        cluster_block_width,
        heatmap_width,
        colorbar_width,
    ]
    fig = plt.figure(figsize=(width, height))
    axs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=width_ratios,
        wspace=0.15 / width,
        hspace=0.13 / height,
        height_ratios=height_ratios,
    )

    heatmap_ax = fig.add_subplot(axs[1, 1])

    width, height = fig.get_size_inches()
    max_cbar_height = 4.0
    if height > max_cbar_height:
        # to make the colorbar shorter, the
        # ax is split and the lower portion is used.
        axs2 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=axs[1, 2], height_ratios=[height - max_cbar_height, max_cbar_height],
        )
        heatmap_cbar_ax = fig.add_subplot(axs2[1])
    else:
        heatmap_cbar_ax = fig.add_subplot(axs[1, 2])

    heatmap(df=df, ax=heatmap_ax, norm=Normalize(vmin=None, vmax=None), plot_colorbar=True, colorbar_ax=heatmap_cbar_ax,
            show_xaxis=show_xaxis, show_yaxis=False, plot_hline=True)
    if show_group:
        plot_categories_as_colorblocks(
            fig.add_subplot(axs[1, 0]), df, colors=cluster_colors_array, orientation='left'
        )
    # plot cluster legends on top of heatmap_ax (if given)
    if show_group_txt:
        plot_gene_groups_brackets(
            fig.add_subplot(axs[0, 1], sharex=heatmap_ax),
            group_positions=group_position,
            group_labels=group_labels,
            rotation=None,
            left_adjustment=-0.3,
            right_adjustment=0.3,
        )
    return fig


def marker_genes_heatmap(
        data: StereoExpData,
        cluster_res: pd.DataFrame,
        marker_res: dict,
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
        do_log=True,
        width=None,
        height=None):
    """
    heatmap of marker genes

    :param data: StereoExpData object
    :param cluster_res: cluster result
    :param marker_res: maker genes result
    :param markers_num: top N maker genes
    :param sort_key: sorted key
    :param ascend: False or True
    :param show_labels: show labels or not
    :param show_group: show group or not
    :param show_group_txt: show group names or not
    :param cluster_colors_array: color values
    :param min_value: minus value
    :param max_value: max value
    :param gene_list: gene name list
    :param do_log: calculate log or not
    :param width: the figure width in pixels.
    :param height: the figure height in pixels.

    """
    draw_df, group_labels, group_position = make_draw_df(data=data, group=cluster_res, marker_res=marker_res,
                                                         top_genes=markers_num, sort_key=sort_key, ascend=ascend,
                                                         gene_list=gene_list, min_value=min_value, max_value=max_value)
    if do_log:
        draw_df = np.log1p(draw_df)
    return plot_heatmap(df=draw_df, show_labels=show_labels, show_group=show_group, show_group_txt=show_group_txt,
                        group_position=group_position, group_labels=group_labels,
                        cluster_colors_array=cluster_colors_array,
                        width=width, height=height)


class MarkerGenesScatterPlot:
    __title_font_size = 8
    __category_width = 0.37
    __category_height = 0.35
    __gene_groups_brackets_height = 0.3
    __legend_width = 4
    __color_map = 'Reds'

    def __init__(
            self,
            data: StereoExpData,
            marker_genes_res: dict,
    ):
        self.data = data
        self.marker_genes_res = marker_genes_res
        self.marker_genes_parameters = marker_genes_res['parameters']

    def _store_marker_genes_result_by_group(self):
        marker_genes_group_keys = natsort.natsorted([key for key in self.marker_genes_res.keys() if '.vs.' in key])
        res_dict = {}
        for mg_key in marker_genes_group_keys:
            group_name = mg_key.split('.vs.')[0]
            res_dict[group_name] = self.marker_genes_res[mg_key].set_index('genes')
        return res_dict

    def _get_dot_size_and_color(
            self,
            groups,
            gene_names,
            marker_genes_res_dict,
            mean_expressin_in_group,
            values_to_plot=None,
    ):
        original_marker_genes_key = self.marker_genes_parameters.get('marker_genes_res_key')
        if original_marker_genes_key is not None:
            pct: pd.DataFrame = self.data.tl.result[original_marker_genes_key]['pct']
        else:
            pct: pd.DataFrame = self.marker_genes_res['pct']
        pct = pct.set_index('genes')
        # marker_genes_res_dict = self._store_marker_genes_result_by_group()
        # mean_expressin_in_group = pipeline_utils.cell_cluster_to_gene_exp_cluster(
        #     self.data, self.marker_genes_parameters['cluster_res_key'], kind='mean')
        
        for g in groups:
            if g in marker_genes_res_dict and 'mean_count' not in marker_genes_res_dict[g].columns:
                genes = marker_genes_res_dict[g].index
                marker_genes_res_dict[g]['mean_count'] = mean_expressin_in_group[g].loc[genes].to_numpy()
            dot_size = pct[g].loc[gene_names].to_numpy() * 100
            if values_to_plot is None:
                # yield pct[g][gene_index].values * 100, mean_expressin_in_group[g][gene_index].values
                dot_color = mean_expressin_in_group[g].loc[gene_names].to_numpy()
            else:
                if values_to_plot == 'logfoldchanges':
                    column = 'log2fc'
                else:
                    column = values_to_plot
                column = column.replace('log10_', '')
                flag = gene_names.isin(marker_genes_res_dict[g].index).to_numpy()
                # dot_size = pct[g].loc[gene_names].to_numpy() * 100
                if not np.any(flag):
                    dot_color = 0
                else:
                    dot_color = np.zeros(gene_names.size, dtype=float)
                    gn = gene_names[flag]
                    if values_to_plot.startswith('log10'):
                        # yield pct[g][gene_index].values * 100, -1 * np.log10(marker_genes_res_dict[g][column][gene_index].values)
                        dot_color[flag] = -1 * np.log10(marker_genes_res_dict[g][column].loc[gn].to_numpy())
                        # yield pct[g].loc[gene_names].to_numpy() * 100, dot_color
                    else:
                        # yield pct[g][gene_index].values * 100, marker_genes_res_dict[g][column][gene_index].values
                        dot_color[flag] = marker_genes_res_dict[g][column].loc[gn].to_numpy()
            yield dot_size, dot_color

    def _create_plot_scatter_data(
            self,
            markers_num=5,
            genes=None,
            groups=None,
            values_to_plot=None,
            sort_by='scores'
    ):
        cluster_res = self.data.tl.result[self.marker_genes_parameters['cluster_res_key']]
        if values_to_plot is None:
            group_names = np.asarray(natsort.natsorted(cluster_res['group'].unique()))
        else:
            group_names = np.asarray(
                natsort.natsorted([key.split('.vs.')[0] for key in self.marker_genes_res.keys() if '.vs.' in key]))
        if group_names.size == 0:
            raise Exception('There is no group to show, please to check the parameter `groups`')

        marker_genes_group_keys = natsort.natsorted([key for key in self.marker_genes_res.keys() if '.vs.' in key])
        if groups is not None:
            if isinstance(groups, str):
                groups = [groups]
            marker_genes_group_keys = [key for key in marker_genes_group_keys if key.split('.vs.')[0] in groups]
        gene_names = []
        gene_intervals = []
        marker_genes_group_keys_to_show = []
        df_list = []
        if sort_by == 'logfoldchanges':
            sort_by = 'log2fc'
        marker_genes_res_dict = self._store_marker_genes_result_by_group()
        mean_expressin_in_group = pipeline_utils.cell_cluster_to_gene_exp_cluster(
            self.data, self.marker_genes_parameters['cluster_res_key'], kind='mean', filter_raw=False)
        for mg_key in marker_genes_group_keys:
            if genes is None:
                isin = self.marker_genes_res[mg_key]['genes'].isin(self.data.gene_names)
                topn_res = self.marker_genes_res[mg_key][isin].sort_values(by=sort_by, ascending=False).head(markers_num)
            else:
                if isinstance(genes, str):
                    genes = [genes]
                isin = self.marker_genes_res[mg_key]['genes'].isin(genes)
                topn_res = self.marker_genes_res[mg_key][isin].sort_values(by=sort_by, ascending=False)
            current_gene_names = topn_res['genes']
            # current_gene_index = topn_res.index
            # return current_gene_names, current_gene_index
            if current_gene_names.size == 0:
                continue
            current_gene_count = len(gene_names)
            current_gene_idx = list(range(current_gene_count, current_gene_count + current_gene_names.size))
            gene_names.extend(current_gene_names)
            gene_intervals.append((current_gene_idx[0], current_gene_idx[-1]))
            marker_genes_group_keys_to_show.append(mg_key)
            tmp = []
            for i, dot_style in enumerate(
                    self._get_dot_size_and_color(
                        group_names, current_gene_names, 
                        marker_genes_res_dict, mean_expressin_in_group, values_to_plot)
                    ):
                dot_size, dot_color = dot_style
                df = pd.DataFrame({
                    'x': current_gene_idx,
                    'y': i,
                    'dot_size': dot_size,
                    'dot_color': dot_color
                })
                tmp.append(df)
            df_list.append(pd.concat(tmp, axis=0))
            if genes is not None:
                break
        return pd.concat(df_list, axis=0), gene_names, group_names, marker_genes_group_keys_to_show, gene_intervals

    def _plot_gene_groups_brackets(
            self,
            ax: Axes,
            gene_intervals,
            marker_genes_group_keys_to_show
    ):
        ax.axis('off')
        verts = []
        codes = []
        for i, lr in enumerate(gene_intervals):
            left, right = lr
            verts.append((left, 0))
            verts.append((left, 0.5))
            verts.append((right, 0.5))
            verts.append((right, 0))
            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            text = marker_genes_group_keys_to_show[i].split('.vs.')[0]
            if len(text) > 4:
                text_position = left + (right - left) / 3
                rotation = 40
            else:
                text_position = left + (right - left) / 2
                rotation = 0
            ax.text(
                x=text_position,
                y=1,
                s=text,
                rotation=rotation
            )
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor='none', lw=1.5)
        ax.add_patch(patch)

    def _plot_gene_groups_scatter(
            self,
            ax: Axes,
            plot_data,
            gene_names,
            group_names
    ):
        ax.set_xlim(left=-1, right=len(gene_names))
        ax.xaxis.set_ticks(range(len(gene_names)), gene_names)
        ax.set_ylim(bottom=-1, top=len(group_names))
        ax.yaxis.set_ticks(range(len(group_names)), group_names)
        ax.tick_params(axis='x', labelrotation=90)
        return ax.scatter(
            x=plot_data['x'],
            y=plot_data['y'],
            s=plot_data['dot_size'],
            c=plot_data['dot_color'],
            cmap=self.__color_map
        )

    def _plot_colorbar(
            self,
            ax: Axes,
            im,
            values_to_plot
    ):
        if values_to_plot is None:
            colorbar_title = 'Mean expression in group'
        else:
            if values_to_plot == 'logfoldchanges':
                colorbar_title = 'log fold changes'
            else:
                colorbar_title = values_to_plot.replace('_', ' ')
        ax.set_title(colorbar_title, fontdict={'fontsize': self.__title_font_size})
        plt.colorbar(im, cax=ax, orientation='horizontal', ticklocation='bottom')

    def _plot_dot_size_map(
            self,
            ax: Axes
    ):
        ax.set_title('Fraction of cells in group(%)', fontdict={'fontsize': self.__title_font_size})
        ax.set_xlim(left=5, right=105)
        ax.set_ylim(bottom=0, top=1)
        ax.xaxis.set_ticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.axes.set_frame_on(False)
        ax.yaxis.set_tick_params(left=False, labelleft=False)
        ax.scatter(
            x=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            y=[0.4] * 10,
            s=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            c='grey'
        )

    def plot_scatter(
            self,
            markers_num: int = 10,
            genes: Union[Optional[Sequence[str]], str] = None,
            groups: Union[Optional[Sequence[str]], str] = None,
            values_to_plot: Optional[
                Literal[
                    'scores',
                    'logfoldchanges',
                    'pvalues',
                    'pvalues_adj',
                    'log10_pvalues',
                    'log10_pvalues_adj'
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
        if (values_to_plot is not None) and ('pvalues' in values_to_plot) and \
                self.marker_genes_parameters['method'] == 'logreg':
            raise Exception("Just only the t_test and wilcoxon_test method would output the pvalues and pvalues_adj.")

        plot_data, gene_names, group_names, marker_genes_group_keys_to_show, gene_intervals = \
            self._create_plot_scatter_data(markers_num, genes, groups, values_to_plot, sort_by)

        if width is None or height is None:
            main_area_width = self.__category_width * len(gene_names)
            if len(group_names) < 5:
                main_area_height = self.__category_height * 6
            else:
                main_area_height = self.__category_height * len(group_names)
        else:
            width /= 100
            height /= 100
            if width <= self.__legend_width:
                width = 50
            if height <= self.__gene_groups_brackets_height:
                height = 7
            main_area_width, main_area_height = width - self.__legend_width, height - self.__gene_groups_brackets_height
        width_ratios = [main_area_width, self.__legend_width]
        height_ratios = [self.__gene_groups_brackets_height + main_area_height]
        width = sum(width_ratios)
        height = sum(height_ratios)

        fig = plt.figure(figsize=(width, height))

        axs = gridspec.GridSpec(
            nrows=1,
            ncols=2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=(0.15 / main_area_width),
            hspace=0
        )

        axs_main = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=1,
            width_ratios=[main_area_width],
            height_ratios=[self.__gene_groups_brackets_height, main_area_height],
            wspace=0,
            hspace=(0.13 / main_area_height),
            subplot_spec=axs[0, 0]
        )

        ax_scatter = fig.add_subplot(axs_main[1, 0])
        main_im = self._plot_gene_groups_scatter(ax_scatter, plot_data, gene_names, group_names)

        if genes is None:
            ax_top = fig.add_subplot(axs_main[0, 0], sharex=ax_scatter)
            self._plot_gene_groups_brackets(ax_top, gene_intervals, marker_genes_group_keys_to_show)

        axs_on_right = gridspec.GridSpecFromSubplotSpec(
            nrows=4,
            ncols=1,
            height_ratios=[0.55, 0.05, 0.2, 0.1],
            subplot_spec=axs[0, 1],
            hspace=0.1
        )

        ax_colorbar = fig.add_subplot(axs_on_right[1, 0])
        self._plot_colorbar(ax_colorbar, main_im, values_to_plot)

        ax_dot_size_map = fig.add_subplot(axs_on_right[3, 0])
        self._plot_dot_size_map(ax_dot_size_map)

        return fig
