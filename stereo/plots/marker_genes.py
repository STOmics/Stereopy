#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:marker_genes.py
@time:2021/03/31
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from matplotlib.axes import Axes
from ._plot_basic.heatmap_plt import heatmap, plot_categories_as_colorblocks, plot_gene_groups_brackets
from ..core.stereo_exp_data import StereoExpData
from ..core.stereo_result import StereoResult
from ..utils import data_helper


def plot_marker_genes_text(
        marker_res: StereoResult,
        groups: Union[str, Sequence[str]] = 'all',
        markers_num: int = 20,
        sort_key: str = 'scores',
        ascend: bool = False,
        fontsize: int = 8,
        ncols: int = 4,
        sharey: bool = True,
        ax: Optional[Axes] = None,
        **kwargs,
):  # scatter plot, 差异基因显著性图，类碎石图
    """
    marker gene scatter visualization.

    :param marker_res: the StereoResult of FindMarkers tool.
    :param groups: list of cluster ids or 'all' clusters, a cluster equal a group.
    :param markers_num: top N genes to show in each cluster.
    :param sort_key: the sort key for getting top n marker genes, default `scores`.
    :param ascend: asc or dec.
    :param fontsize: font size.
    :param ncols: number of plot columns.
    :param sharey:
    :param ax:
    :param kwargs: other args for plot.
    :return:
    """
    # 调整图像 panel/grid 相关参数
    if 'n_panels_per_row' in kwargs:
        n_panels_per_row = kwargs['n_panels_per_row']
    else:
        n_panels_per_row = ncols
    # group_names = adata.uns[key]['names'].dtype.names if groups is None else groups
    if groups == 'all':
        group_names = list(set(marker_res.matrix['groups']))
    else:
        group_names = [groups] if isinstance(groups, str) else groups
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)
    # 初始化图像
    width = 10
    height = 10
    fig = plt.figure(
        figsize=(
            n_panels_x * width,  # rcParams['figure.figsize'][0],
            n_panels_y * height,  # rcParams['figure.figsize'][1],
        )
    )
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
            ymin = min(ymin, np.min(scores))
            ymax = max(ymax, np.max(scores))

            if ax0 is None:
                ax = fig.add_subplot(gs[count])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[count], sharey=ax0)
        else:
            ymin = np.min(scores)
            ymax = np.max(scores)
            ymax += 0.3 * (ymax - ymin)

            ax = fig.add_subplot(gs[count])
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
            ax.set_ylabel('score')

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)


def make_draw_df(data: StereoExpData, group: pd.DataFrame, marker_res: StereoResult, top_genes: int = 8,
                 sort_key: str = 'scores', ascend: bool = False):
    gene_names_dict = get_groups_marker(marker_res, top_genes, sort_key, ascend)
    gene_names = list()
    gene_group_labels = list()
    gene_group_positions = list()
    start = 0
    for label, gene_list in gene_names_dict.items():
        if isinstance(gene_list, str):
            gene_list = [gene_list]
        gene_names.extend(list(gene_list))
        gene_group_labels.append(label)
        gene_group_positions.append((start, start + len(gene_list) - 1))
        start += len(gene_list)
    draw_df = data_helper.exp_matrix2df(data, gene_name=np.array(gene_names))
    # add obs values
    cluster_data = group.set_index('bins')
    draw_df = pd.concat([draw_df, cluster_data], axis=1)
    draw_df = draw_df.set_index(draw_df['cluster'].astype('category'))
    draw_df = draw_df.sort_index()
    return draw_df, gene_group_labels, gene_group_positions


def get_groups_marker(marker_res: StereoResult, top_genes: int = 8, sort_key: str = 'scores', ascend: bool = False):
    groups = set(marker_res.matrix['groups'])
    groups_genes = {}
    for g in groups:
        res = data_helper.get_top_marker(g, marker_res, sort_key, ascend, top_genes)
        genes = res['genes'].values
        groups_genes[g] = genes
    return groups_genes


def plot_heatmap(
        df,
        show_labels=True,
        show_group=True,
        show_group_txt=True,
        group_position=None,
        group_labels=None,
        cluster_colors_array=None,
        **kwargs
):
    """

    :param df: the dataframe object. the index of df is group info if show group is True.
    :param show_labels: whether to show the labels of the heatmap plot.
    :param show_group: show the group info on the left of the heatmap.
    :param show_group_txt: show the group info on the top of the heatmap.
    :param group_position: the position of group txt, must to be set if show_group_txt is True.
    :param group_labels: the label of group, must to be set if show_group_txt is True.
    :param cluster_colors_array: he list of colors in the color block on the left of heatmap.
    :param kwargs: other args for plot.
    :return:
    """
    kwargs.setdefault("figsize", (10, 10))
    kwargs.setdefault("colorbar_width", 0.2)
    colorbar_width = kwargs.get("colorbar_width")
    figsize = kwargs.get("figsize")
    cluster_block_width = kwargs.setdefault("cluster_block_width", 0.2) if show_group else 0
    if figsize is None:
        height = 6
        if show_labels:
            heatmap_width = len(df.columns) * 0.3
        else:
            heatmap_width = 8
        width = heatmap_width + cluster_block_width
    else:
        width, height = figsize
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
        axs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=axs[1, 2],
                                                height_ratios=[height - max_cbar_height, max_cbar_height],
                                                )
        heatmap_cbar_ax = fig.add_subplot(axs2[1])
    else:
        heatmap_cbar_ax = fig.add_subplot(axs[1, 2])

    heatmap(df=df, ax=heatmap_ax,
            norm=Normalize(vmin=None, vmax=None), plot_colorbar=True, colorbar_ax=heatmap_cbar_ax,
            show_labels=show_labels, plot_hline=True)
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


def plot_marker_genes_heatmap(
        data: StereoExpData,
        cluster_res: StereoResult,
        marker_res: StereoResult,
        markers_num: int = 5,
        sort_key: str = 'scores',
        ascend: bool = False,
        show_labels: bool = True,
        show_group: bool = True,
        show_group_txt: bool = True,
        cluster_colors_array=None):
    draw_df, group_labels, group_position = make_draw_df(data=data, group=cluster_res.matrix, marker_res=marker_res,
                                                         top_genes=markers_num, sort_key=sort_key, ascend=ascend)
    plot_heatmap(df=draw_df, show_labels=show_labels, show_group=show_group, show_group_txt=show_group_txt,
                 group_position=group_position, group_labels=group_labels, cluster_colors_array=cluster_colors_array)
