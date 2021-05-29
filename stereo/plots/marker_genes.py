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
from ._plot_basic.get_stereo_data import get_degs_res
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Optional, Sequence, Union
from matplotlib.axes import Axes
from ._plot_basic.heatmap_plt import heatmap, plot_categories_as_colorblocks, plot_gene_groups_brackets
from ..log_manager import logger
from scipy.sparse import issparse
from ._plot_basic.get_stereo_data import get_find_marker_group, get_spatial_lag_group


def plot_marker_genes(
        adata: AnnData,
        groups: Union[str, Sequence[str]] = 'all',
        n_genes: int = 20,
        find_maker_name: Optional[str] = 'find_marker',
        fontsize: int = 8,
        ncols: int = 4,
        sharey: bool = True,
        ax: Optional[Axes] = None,
        **kwds,
):  # scatter plot, 差异基因显著性图，类碎石图
    """
    marker gene scatter visualization

    :param adata: anndata
    :param groups: list of cluster ids or 'all' clusters, a cluster equal a group
    :param n_genes: top N genes to show in each cluster
    :param find_maker_name: the task's name, defined when running 'FindMaker' tool by setting 'name' property.
    :param fontsize: font size
    :param ncols: number of columns
    :return: None
    """

    # 调整图像 panel/grid 相关参数
    if 'n_panels_per_row' in kwds:
        n_panels_per_row = kwds['n_panels_per_row']
    else:
        n_panels_per_row = ncols
    # group_names = adata.uns[key]['names'].dtype.names if groups is None else groups
    if groups == 'all':
        group_names = list(adata.uns[find_maker_name].keys())
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
        result = get_degs_res(adata, data_key=find_maker_name, group_key=group_name, top_k=n_genes)
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

        ax.set_xlim(-0.9, n_genes - 0.1)

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


def get_default_cluster(adata, tool_name):
    marker_res = adata.uns[tool_name]
    groups = get_find_marker_group(adata, tool_name) if isinstance(marker_res, dict) \
        else get_spatial_lag_group(adata, tool_name)
    return groups


def get_group_top_genes(adata, tool_name, top_k_gene, marker_clusters=None):
    marker_res = adata.uns[tool_name]
    default_cluster = get_default_cluster(adata, tool_name)
    if marker_clusters is None:
        marker_clusters = default_cluster
    if not set(marker_clusters).issubset(set(default_cluster)):
        marker_clusters = default_cluster

    gene_names_dict = {}  # dict in which each cluster is the keyand the num_show_gene are the values
    if isinstance(marker_res, dict):
        for cluster in marker_clusters:
            top_marker = marker_res[cluster].top_k_marker(top_k_genes=top_k_gene, sort_key='scores')
            genes_array = top_marker['genes'].values
            if len(genes_array) == 0:
                logger.warning("Cluster {} has no genes.".format(cluster))
                continue
            gene_names_dict[cluster] = genes_array
    else:
        spatial_lag_top = marker_res.top_markers(top_k=top_k_gene)
        for cluster in marker_clusters:
            genes_array = spatial_lag_top[cluster+'_lag_coeff']['genes'].values
            if len(genes_array) == 0:
                logger.warning("Cluster {} has no genes.".format(cluster))
                continue
            gene_names_dict[cluster] = genes_array
    return gene_names_dict


def plot_heatmap_marker_genes(
        adata: AnnData = None,
        cluster_name="clustering",
        fine_maker_name=None,
        num_show_gene=8,
        show_labels=True,
        order_cluster=True,
        marker_clusters=None,
        cluster_colors_array=None,
        **kwargs
):  # heatmap, 差异基因热图
    """
    marker gene heatmap.

    :param adata: AnnData object.
    :param cluster_name: the task's name, defined when running 'Clustering' tool by setting 'name' property.
    :param fine_maker_name: the task's name, defined when running 'FindMaker' tool by setting 'name' property.
    :param num_show_gene: number of genes to show in each cluster.
    :param show_labels: show gene name on axis.
    :param order_cluster: reorder the cluster list in plot (y axis).
    :param marker_clusters: the list of clusters to show on the heatmap.
    :param cluster_colors_array: the list of colors in the color block on the left of heatmap.
    :return: None

    Example:

    .. code:: python

        plot_heatmap_maker_genes(adata=adata, marker_uns_key = "rank_genes_groups", figsize = (20, 10))

    """
    gene_names_dict = get_group_top_genes(adata=adata, tool_name=fine_maker_name, top_k_gene=num_show_gene,
                                          marker_clusters=marker_clusters)
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

    # 此处获取所有绘图所需的数据 （表达量矩阵）
    uniq_gene_names = np.unique(gene_names)
    exp_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
    draw_df = pd.DataFrame(exp_matrix[:, adata.var.index.get_indexer(uniq_gene_names)],
                           columns=uniq_gene_names, index=adata.obs_names)
    # add obs values
    cluster_data = adata.uns[cluster_name].cluster.set_index('bins')
    draw_df = pd.concat([draw_df, cluster_data], axis=1)
    draw_df = draw_df[gene_names].set_index(draw_df['cluster'].astype('category'))
    if order_cluster:
        draw_df = draw_df.sort_index()
    kwargs.setdefault("figsize", (10, 10))
    kwargs.setdefault("colorbar_width", 0.2)
    colorbar_width = kwargs.get("colorbar_width")
    figsize = kwargs.get("figsize")

    cluster_block_width = kwargs.setdefault("cluster_block_width", 0.2) if order_cluster else 0
    if figsize is None:
        height = 6
        if show_labels:
            heatmap_width = len(gene_names) * 0.3
        else:
            heatmap_width = 8
        width = heatmap_width + cluster_block_width
    else:
        width, height = figsize
        heatmap_width = width - cluster_block_width

    if gene_group_positions is not None and len(gene_group_positions) > 0:
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

    heatmap(df=draw_df, ax=heatmap_ax,
            norm=Normalize(vmin=None, vmax=None), plot_colorbar=True, colorbar_ax=heatmap_cbar_ax,
            show_labels=True, plot_hline=True)

    if order_cluster:
        plot_categories_as_colorblocks(
            fig.add_subplot(axs[1, 0]), draw_df, colors=cluster_colors_array, orientation='left'
        )

    # plot cluster legends on top of heatmap_ax (if given)
    if gene_group_positions is not None and len(gene_group_positions) > 0:
        plot_gene_groups_brackets(
            fig.add_subplot(axs[0, 1], sharex=heatmap_ax),
            group_positions=gene_group_positions,
            group_labels=gene_group_labels,
            rotation=None,
            left_adjustment=-0.3,
            right_adjustment=0.3,
        )


def plot_heatmap_marker_genes_bak(
        adata: AnnData = None,
        cluster_name="phenograph",
        fine_maker_name=None,
        num_show_gene=8,
        show_labels=True,
        order_cluster=True,
        marker_clusters=None,
        cluster_colors_array=None,
        **kwargs
):  # heatmap, 差异基因热图
    """
    marker gene heatmap.

    :param adata: AnnData object.
    :param cluster_name: the task's name, defined when running 'Clustering' tool by setting 'name' property.
    :param fine_maker_name: the task's name, defined when running 'FindMaker' tool by setting 'name' property.
    :param num_show_gene: number of genes to show in each cluster.
    :param show_labels: show gene name on axis.
    :param order_cluster: reorder the cluster list in plot (y axis).
    :param marker_clusters: the list of clusters to show on the heatmap.
    :param cluster_colors_array: the list of colors in the color block on the left of heatmap.
    :return: None

    Example:

    .. code:: python

        plot_heatmap_maker_genes(adata=adata, marker_uns_key = "rank_genes_groups", figsize = (20, 10))

    """
    if fine_maker_name is None:
        fine_maker_name = 'marker_genes'  # "rank_genes_groups" in original scanpy pipeline
    marker_res = adata.uns[fine_maker_name]
    default_cluster = [i for i in marker_res.keys()]
    if marker_clusters is None:
        marker_clusters = default_cluster
    if not set(marker_clusters).issubset(set(default_cluster)):
        marker_clusters = default_cluster

    gene_names_dict = {}  # dict in which each cluster is the keyand the num_show_gene are the values

    for cluster in marker_clusters:
        top_marker = marker_res[cluster].top_k_marker(top_k_genes=num_show_gene, sort_key='scores')
        genes_array = top_marker['genes'].values
        if len(genes_array) == 0:
            logger.warning("Cluster {} has no genes.".format(cluster))
            continue
        gene_names_dict[cluster] = genes_array
    gene_names = []
    gene_group_labels = []
    gene_group_positions = []
    start = 0
    for label, gene_list in gene_names_dict.items():
        if isinstance(gene_list, str):
            gene_list = [gene_list]
        gene_names.extend(list(gene_list))
        gene_group_labels.append(label)
        gene_group_positions.append((start, start + len(gene_list) - 1))
        start += len(gene_list)

    # 此处获取所有绘图所需的数据 （表达量矩阵）
    uniq_gene_names = np.unique(gene_names)
    exp_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
    draw_df = pd.DataFrame(exp_matrix[:, adata.var.index.get_indexer(uniq_gene_names)],
                           columns=uniq_gene_names, index=adata.obs_names)
    # add obs values
    cluster_data = adata.uns[cluster_name].cluster.set_index('bins')
    draw_df = pd.concat([draw_df, cluster_data], axis=1)
    draw_df = draw_df[gene_names].set_index(draw_df['cluster'].astype('category'))
    if order_cluster:
        draw_df = draw_df.sort_index()
    kwargs.setdefault("figsize", (10, 10))
    kwargs.setdefault("colorbar_width", 0.2)
    colorbar_width = kwargs.get("colorbar_width")
    figsize = kwargs.get("figsize")

    cluster_block_width = kwargs.setdefault("cluster_block_width", 0.2) if order_cluster else 0
    if figsize is None:
        height = 6
        if show_labels:
            heatmap_width = len(gene_names) * 0.3
        else:
            heatmap_width = 8
        width = heatmap_width + cluster_block_width
    else:
        width, height = figsize
        heatmap_width = width - cluster_block_width

    if gene_group_positions is not None and len(gene_group_positions) > 0:
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

    heatmap(df=draw_df, ax=heatmap_ax,
            norm=Normalize(vmin=None, vmax=None), plot_colorbar=True, colorbar_ax=heatmap_cbar_ax,
            show_labels=True, plot_hline=True)

    if order_cluster:
        plot_categories_as_colorblocks(
            fig.add_subplot(axs[1, 0]), draw_df, colors=cluster_colors_array, orientation='left'
        )

    # plot cluster legends on top of heatmap_ax (if given)
    if gene_group_positions is not None and len(gene_group_positions) > 0:
        plot_gene_groups_brackets(
            fig.add_subplot(axs[0, 1], sharex=heatmap_ax),
            group_positions=gene_group_positions,
            group_labels=gene_group_labels,
            rotation=None,
            left_adjustment=-0.3,
            right_adjustment=0.3,
        )
