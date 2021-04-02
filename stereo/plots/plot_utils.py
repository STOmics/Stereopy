#!/usr/bin/env python3
# coding: utf-8
"""
@author: Shixu He  heshixu@genomics.cn
@last modified by: Shixu He
@file:plot_utils.py
@time:2021/03/15
"""

from anndata import AnnData
import pandas as pd
import numpy as np
import math

from matplotlib.colors import Normalize, ListedColormap
from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn

from ._plot_basic.scatter_plt import scatter, plot_cluster_result
from ._plot_basic.heatmap_plt import heatmap, _plot_categories_as_colorblocks, _plot_gene_groups_brackets

from typing import Optional, Sequence, Union

from ..log_manager import logger


def plot_spatial_distribution(
        adata: AnnData,
        obs_key: list = ["total_counts", "n_genes_by_counts"],
        ncols=2,
        dot_size=None,
        color_list=None,
        invert_y=False
):  # scatter plot, 表达矩阵空间分布
    """
    Plot spatial distribution of specified obs data.
    ============ Arguments ============
    :param adata: AnnData object.
    :param obs_key: specified obs key list, for example: ["total_counts", "n_genes_by_counts"]
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param cmap: Color map.
    :param invert_y: whether to invert y-axis.
    ============ Return ============
    None
    ============ Example ============
    plot_spatial_distribution(adata=adata)
    """
    # sc.pl.embedding(adata, basis="spatial", color=["total_counts", "n_genes_by_counts"],size=30)

    if dot_size is None:
        dot_size = 120000 / adata.shape[0]

    ncols = min(ncols, len(obs_key))
    nrows = np.ceil(len(obs_key) / ncols).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = plt.figure(figsize=(ncols * 10, nrows * 8))
    left = 0.2 / ncols
    bottom = 0.13 / nrows
    axs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        left=left,
        right=1 - (ncols - 1) * left - 0.01 / ncols,
        bottom=bottom,
        top=1 - (nrows - 1) * bottom - 0.1 / nrows,
        # hspace=hspace,
        # wspace=wspace,
    )

    if color_list is None:
        cmap = get_cmap()
    else:
        cmap = ListedColormap(color_list)
        # 把特定值改为 np.nan 之后，可以利用 cmap.set_bad("white") 来遮盖掉这部分数据

    # 散点图上每个点的坐标数据来自于 adata 的 obsm["spatial"]，每个点的颜色（数值）数据来自于 adata 的 obs_vector()
    for i, key in enumerate(obs_key):
        # color_data = np.asarray(adata.obs_vector(key), dtype=float)
        color_data = adata.obs_vector(key)
        order = np.argsort(~pd.isnull(color_data), kind="stable")
        spatial_data = np.array(adata.obsm["spatial"])[:, 0: 2]
        color_data = color_data[order]
        spatial_data = spatial_data[order, :]

        # color_data 是图像中各个点的值，也对应了每个点的颜色。data_points则对应了各个点的坐标
        ax = fig.add_subplot(axs[i])  # ax = plt.subplot(axs[i]) || ax = fig.add_subplot(axs[1, 1]))
        ax.set_title(key)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("spatial1")
        ax.set_ylabel("spatial2")
        pathcollection = scatter(
            spatial_data[:, 0],
            spatial_data[:, 1],
            ax=ax,
            marker=".",
            dot_colors=color_data,
            dot_size=dot_size,
            cmap=cmap,
        )
        plt.colorbar(
            pathcollection,
            ax=ax,
            pad=0.01,
            fraction=0.08,
            aspect=30,
        )
        ax.autoscale_view()
        if invert_y:
            ax.invert_yaxis()


def plot_spatial_cluster(
        adata: AnnData,
        obs_key: list = ["phenograph"],
        plot_cluster: list = None,
        bad_color="lightgrey",
        ncols=2,
        dot_size=None,
        invert_y=False,
        color_list=['violet', 'turquoise', 'tomato', 'teal',
                    'tan', 'silver', 'sienna', 'red', 'purple',
                    'plum', 'pink', 'orchid', 'orangered', 'orange',
                    'olive', 'navy', 'maroon', 'magenta', 'lime',
                    'lightgreen', 'lightblue', 'lavender', 'khaki',
                    'indigo', 'grey', 'green', 'gold', 'fuchsia',
                    'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral',
                    'chocolate', 'chartreuse', 'brown', 'blue', 'black',
                    'beige', 'azure', 'aquamarine', 'aqua']):  # scatter plot, 聚类后表达矩阵空间分布
    """
    Plot spatial distribution of specified obs data.
    ============ Arguments ============
    :param adata: AnnData object.
    :param obs_key: specified obs cluster key list, for example: ["phenograph"].
    :param plot_cluster: the name list of clusters to show.
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param cmap: Color map.
    :param invert_y: whether to invert y-axis.
    ============ Return ============
    None.
    ============ Example ============
    plot_spatial_cluster(adata = adata)
    """
    # sc.pl.embedding(adata, basis="spatial", color=["total_counts", "n_genes_by_counts"],size=30)

    if isinstance(obs_key, str):
        obs_key = ["obs_key"]

    plot_cluster_result(adata, obs_key=obs_key, pos_key="spatial", plot_cluster=plot_cluster, bad_color=bad_color,
                        ncols=ncols, dot_size=dot_size, invert_y=invert_y, color_list=color_list)


def plot_to_select_filter_value(
        adata: AnnData,
        x=["total_counts", "total_counts"],
        y=["pct_counts_mt", "n_genes_by_counts"],
        ncols=1,
        **kwargs):  # scatter plot, 线粒体分布图
    """
    Plot .
    ============ Arguments ============
    :param adata: AnnData object.
    :param x, y: obs key pairs for drawing. For example, assume x=["a", "a", "b"] and y=["c", "d", "e"], the output plots will include "a-c", "a-d", "b-e".
    ============ Return ============
    None.
    ============ Example ============
    plot_spatial_cluster(adata = adata)
    """
    # sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    # sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]

    width = 20
    height = 10
    nrows = math.ceil(len(x) / ncols)

    doc_color = "gray"

    fig = plt.figure(figsize=(width, height))
    axs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
    )
    for i, (xi, yi) in enumerate(zip(x, y)):
        draw_data = np.c_[adata.obs_vector(xi), adata.obs_vector(yi)]
        dot_size = 120000 / draw_data.shape[0]
        ax = fig.add_subplot(axs[i])
        # ax.set_title()
        # ax.set_yticks([])
        # ax.set_xticks([])
        ax.set_xlabel(xi)
        ax.set_ylabel(yi)
        scatter(
            draw_data[:, 0],
            draw_data[:, 1],
            ax=ax,
            marker=".",
            dot_colors=doc_color,
            dot_size=dot_size
        )


def plot_variable_gene(adata: AnnData, logarize=False):  # scatter plot, 表达量差异-均值图
    """
    Copied from scanpy and modified.
    """
    # 该图像需要前置数据处理：sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # 再画图：sc.pl.highly_variable_genes(adata)

    result = adata.var
    gene_subset = result.highly_variable
    means = result.means
    var_or_disp = result.dispersions
    var_or_disp_norm = result.dispersions_norm
    width = 10
    height = 10

    plt.figure(figsize=(2 * width, height))
    plt.subplots_adjust(wspace=0.3)
    for idx, d in enumerate([var_or_disp_norm, var_or_disp]):
        plt.subplot(1, 2, idx + 1)
        for label, color, mask in zip(
                ['highly variable genes', 'other genes'],
                ['black', 'grey'],
                [gene_subset, ~gene_subset],
        ):
            if False:
                means_, var_or_disps_ = np.log10(means[mask]), np.log10(d[mask])
            else:
                means_, var_or_disps_ = means[mask], d[mask]
            plt.scatter(means_, var_or_disps_, label=label, c=color, s=1)
        if logarize:  # there's a bug in autoscale
            plt.xscale('log')
            plt.yscale('log')
            y_min = np.min(var_or_disp)
            y_min = 0.95 * y_min if y_min > 0 else 1e-1
            plt.xlim(0.95 * np.min(means), 1.05 * np.max(means))
            plt.ylim(y_min, 1.05 * np.max(var_or_disp))
        if idx == 0:
            plt.legend()
        plt.xlabel(('$log_{10}$ ' if False else '') + 'mean expressions of genes')
        data_type = 'dispersions'
        plt.ylabel(
            ('$log_{10}$ ' if False else '')
            + '{} of genes'.format(data_type)
            + (' (normalized)' if idx == 0 else ' (not normalized)')
        )


def plot_cluster_umap(
        adata: AnnData,
        obs_key: list = ["phenograph"],
        plot_cluster: list = None,
        bad_color="lightgrey",
        ncols=2,
        dot_size=None,
        invert_y=False,
        color_list=['violet', 'turquoise', 'tomato', 'teal',
                    'tan', 'silver', 'sienna', 'red', 'purple',
                    'plum', 'pink', 'orchid', 'orangered', 'orange',
                    'olive', 'navy', 'maroon', 'magenta', 'lime',
                    'lightgreen', 'lightblue', 'lavender', 'khaki',
                    'indigo', 'grey', 'green', 'gold', 'fuchsia',
                    'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral',
                    'chocolate', 'chartreuse', 'brown', 'blue', 'black',
                    'beige', 'azure', 'aquamarine', 'aqua',
                    ]
):  # scatter plot，聚类结果PCA/umap图
    """
    Plot spatial distribution of specified obs data.
    ============ Arguments ============
    :param adata: AnnData object.
    :param obs_key: specified obs cluster key list, for example: ["phenograph"].
    :param plot_cluster: the name list of clusters to show.
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param cmap: Color map.
    :param invert_y: whether to invert y-axis.
    ============ Return ============
    None.
    ============ Example ============
    plot_cluster_umap(adata = adata)
    """

    if (isinstance(obs_key, str)):
        obs_key = [obs_key]

    plot_cluster_result(adata, obs_key=obs_key, pos_key="X_umap", plot_cluster=plot_cluster, bad_color=bad_color,
                        ncols=ncols, dot_size=dot_size, invert_y=invert_y, color_list=color_list)


def plot_expression_difference(
        adata: AnnData,
        groups: Union[str, Sequence[str]] = None,
        n_genes: int = 20,
        key: Optional[str] = 'rank_genes_groups',
        fontsize: int = 8,
        ncols: int = 4,
        sharey: bool = True,
        show: Optional[bool] = None,
        save: Optional[bool] = None,
        ax: Optional[Axes] = None,
        **kwds,
):  # scatter plot, 差异基因显著性图，类碎石图
    """
    Copied from scanpy and modified.
    """

    # 调整图像 panel/grid 相关参数
    if 'n_panels_per_row' in kwds:
        n_panels_per_row = kwds['n_panels_per_row']
    else:
        n_panels_per_row = ncols
    group_names = adata.uns[key]['names'].dtype.names if groups is None else groups
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
        gene_names = adata.uns[key]['names'][group_name][:n_genes]
        scores = adata.uns[key]['scores'][group_name][:n_genes]

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

        ax.set_title('{} vs. {}'.format(group_name, "Others"))
        if count >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel('ranking')

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax.set_ylabel('score')

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)


def plot_violin_distribution(adata):  # 小提琴统计图
    """
    绘制数据的分布小提琴图。
    ============ Arguments ============
    :param adata: AnnData object.
    ============ Return ============
    None
    """
    _, axs = plt.subplots(1, 3, figsize=(15, 4))
    seaborn.violinplot(y=adata.obs['total_counts'], ax=axs[0])
    seaborn.violinplot(y=adata.obs['n_genes_by_counts'], ax=axs[1])
    seaborn.violinplot(y=adata.obs['pct_counts_mt'], ax=axs[2])


def plot_heatmap_maker_genes(
        adata: AnnData = None,
        cluster_method="phenograph",
        marker_uns_key=None,
        num_show_gene=8,
        show_labels=True,
        order_cluster=True,
        marker_clusters=None,
        cluster_colors_array=None,
        **kwargs
):  # heatmap, 差异基因热图
    """
    绘制 Marker gene 的热图。热图中每一行代表一个 bin 的所有基因的表达量，所有的 bin 会根据所属的 cluster 进行聚集， cluster 具体展示在热图的左侧，用颜色区分。
    ============ Arguments ============
    :param adata: AnnData object.
    :param cluster_methpd: method used in clustering. for example: phenograph, leiden
    :param marker_uns_key: the key of adata.uns, the default value is "marker_genes"
    :param num_show_gene: number of genes to show in each cluster.
    :param show_labels: show gene name on axis.
    :param order_cluster: reorder the cluster list in plot (y axis).
    :param marker_clusters: the list of clusters to show on the heatmap.
    :param cluster_colors_array: the list of colors in the color block on the left of heatmap.
    ============ Return ============

    ============ Example ============
    plot_heatmap_maker_genes(adata=adata, marker_uns_key = "rank_genes_groups", figsize = (20, 10))
    """

    if marker_uns_key is None:
        marker_uns_key = 'marker_genes'  # "rank_genes_groups" in original scanpy pipeline

    # if cluster_method is None:
    #    cluster_method = str(adata.uns[marker_uns_key]['params']['groupby'])

    # cluster_colors_array = adata.uns["phenograph_colors"]

    if marker_clusters is None:
        marker_clusters = adata.uns[marker_uns_key]['names'].dtype.names

    if not set(marker_clusters).issubset(set(adata.uns[marker_uns_key]['names'].dtype.names)):
        marker_clusters = adata.uns[marker_uns_key]['names'].dtype.names

    gene_names_dict = {}  # dict in which each cluster is the keyand the num_show_gene are the values

    for cluster in marker_clusters:
        # get all genes that are 'non-nan'
        genes_array = adata.uns[marker_uns_key]['names'][cluster]
        genes_array = genes_array[~pd.isnull(genes_array)]

        if len(genes_array) == 0:
            logger.warning("Cluster {} has no genes.".format(cluster))
            continue
        gene_names_dict[cluster] = list(genes_array[:num_show_gene])

    adata._sanitize()

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
    draw_df = pd.DataFrame(index=adata.obs_names)
    uniq_gene_names = np.unique(gene_names)
    draw_df = pd.concat(
        [draw_df, pd.DataFrame(adata.X[tuple([slice(None), adata.var.index.get_indexer(uniq_gene_names)])],
                               columns=uniq_gene_names, index=adata.obs_names)],
        axis=1
    )

    # add obs values
    draw_df = pd.concat([draw_df, adata.obs[cluster_method]], axis=1)

    # reorder columns to given order (including duplicates keys if present)
    draw_df = draw_df[list([cluster_method]) + list(uniq_gene_names)]
    draw_df = draw_df[gene_names].set_index(draw_df[cluster_method].astype('category'))
    if order_cluster:
        draw_df = draw_df.sort_index()

    # From scanpy
    # define a layout of 2 rows x 3 columns
    # first row is for 'brackets' (if no brackets needed, the height of this row
    # is zero) second row is for main content. This second row is divided into
    # three axes:
    #   first ax is for the categories defined by `cluster_method`
    #   second ax is for the heatmap
    #   fourth ax is for colorbar

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
        _plot_categories_as_colorblocks(
            fig.add_subplot(axs[1, 0]), draw_df, colors=cluster_colors_array, orientation='left'
        )

    # plot cluster legends on top of heatmap_ax (if given)
    if gene_group_positions is not None and len(gene_group_positions) > 0:
        _plot_gene_groups_brackets(
            fig.add_subplot(axs[0, 1], sharex=heatmap_ax),
            group_positions=gene_group_positions,
            group_labels=gene_group_labels,
            rotation=None,
            left_adjustment=-0.3,
            right_adjustment=0.3,
        )

    # plt.savefig()
