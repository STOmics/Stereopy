#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:plots.py
@time:2021/03/31
"""
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex, Normalize
from matplotlib import gridspec
from ._plot_basic.get_stereo_data import get_cluster_res, get_reduce_x, get_position_array, get_degs_res
import numpy as np
import pandas as pd
from anndata import AnnData
from ._plot_basic.scatter_plt import scatter
import seaborn
from typing import Optional, Sequence, Union
from matplotlib.axes import Axes
from ._plot_basic.heatmap_plt import heatmap, plot_categories_as_colorblocks, plot_gene_groups_brackets
from ..log_manager import logger
from scipy.sparse import issparse


def plot_spatial_cluster(
        adata: AnnData,
        obs_key: list = ["phenograph"],
        pos_key: str = "spatial",
        plot_cluster: list = None,
        bad_color: str = "lightgrey",
        ncols: int = 2,
        dot_size: int = None,
        color_list=['violet', 'turquoise', 'tomato', 'teal','tan', 'silver', 'sienna', 'red', 'purple', 'plum', 'pink',
                    'orchid', 'orangered', 'orange', 'olive', 'navy', 'maroon', 'magenta', 'lime',
                    'lightgreen', 'lightblue', 'lavender', 'khaki', 'indigo', 'grey', 'green', 'gold', 'fuchsia',
                    'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral', 'chocolate', 'chartreuse', 'brown', 'blue', 'black',
                    'beige', 'azure', 'aquamarine', 'aqua',
                    ],
):  # scatter plot, 聚类后表达矩阵空间分布
    """
    Plot spatial distribution of specified obs data.
    ============ Arguments ============
    :param adata: AnnData object.
    :param obs_key: specified obs cluster key list, for example: ["phenograph"].
    :param pos_key: the coordinates of data points for scatter plots. the data points are stored in adata.obsm[pos_key]. choice: "spatial", "X_umap", "X_pca".
    :param plot_cluster: the name list of clusters to show.
    :param bad_color: the name list of clusters to show.
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param color_list: whether to invert y-axis.
    ============ Return ============
    None.
    ============ Example ============
    plot_spatial_cluster(adata = adata)
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
    cmap.set_bad(bad_color)
    # 把特定值改为 np.nan 之后，可以利用 cmap.set_bad("white") 来遮盖掉这部分数据

    for i, key in enumerate(obs_key):
        # color_data = adata.obs_vector(key)  # TODO  replace by get_cluster_res
        color_data = get_cluster_res(adata, data_key=key)
        pc_logic = False

        # color_data = np.asarray(color_data_raw, dtype=float)
        order = np.argsort(~pd.isnull(color_data), kind="stable")
#         spatial_data = np.array(adata.obsm[pos_key])[:, 0: 2]
        spatial_data = get_reduce_x(data=adata, data_key=pos_key)[:, 0:2] if pos_key != 'spatial' \
            else get_position_array(adata, pos_key)
        color_data = color_data[order]
        spatial_data = spatial_data[order, :]

        color_dict = {}
        has_na = False
        if pd.api.types.is_categorical_dtype(color_data):
            pc_logic = True
            if plot_cluster is None:
                plot_cluster = list(color_data.categories)
        if pc_logic:
            cluster_n = len(np.unique(color_data))
            if len(color_list) < cluster_n:
                color_list = color_list * cluster_n
                cmap = ListedColormap(color_list)
                cmap.set_bad(bad_color)
            if len(color_data.categories) > len(plot_cluster):
                color_data = color_data.replace(color_data.categories.difference(plot_cluster), np.nan)
                has_na = True
            color_dict = {str(k): to_hex(v) for k, v in enumerate(color_list)}
            print(color_dict)
            color_data = color_data.map(color_dict)
            if pd.api.types.is_categorical_dtype(color_data):
                color_data = pd.Categorical(color_data)
            if has_na:
                color_data = color_data.add_categories([to_hex(bad_color)])
                color_data = color_data.fillna(to_hex(bad_color))
                # color_dict["NA"]

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
            dot_size=dot_size
        )
        if pc_logic:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])
            # -------------modified by qiuping1@genomics.cn-------------
            # valid_cate = color_data.categories
            # cat_num = len(adata.obs_vector(key).categories)
            # for label in adata.obs_vector(key).categories:
            categories = get_cluster_res(adata, data_key=key).categories
            cat_num = len(categories)
            for label in categories:
                # --------modified end------------------
                ax.scatter([], [], c=color_dict[label], label=label)
            ax.legend(
                frameon=False,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                ncol=(1 if cat_num <= 14 else 2 if cat_num <= 30 else 3),
                # fontsize=legend_fontsize,
            )
        else:
            plt.colorbar(pathcollection, ax=ax, pad=0.01, fraction=0.08, aspect=30)
        ax.autoscale_view()


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


def plot_degs(
        adata: AnnData,
        groups: Union[str, Sequence[str]] = 'all',
        n_genes: int = 20,
        key: Optional[str] = 'find_marker',
        fontsize: int = 8,
        ncols: int = 4,
        sharey: bool = True,
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
    # group_names = adata.uns[key]['names'].dtype.names if groups is None else groups
    if groups == 'all':
        group_names = list(adata.uns[key].keys())
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
        result = get_degs_res(adata, data_key=key, group_key=group_name, top_k=n_genes)
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
    :param color_list: Color list.
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
    :param cluster_method: method used in clustering. for example: phenograph, leiden
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
    marker_res = adata.uns[marker_uns_key]
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
    cluster_data = adata.uns[cluster_method].cluster.set_index('bins')
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
