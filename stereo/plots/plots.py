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
from matplotlib.colors import ListedColormap, to_hex
from matplotlib import gridspec
from ._plot_basic.get_stereo_data import get_cluster_res, get_reduce_x, get_position_array, get_degs_res
import numpy as np
import pandas as pd
from anndata import AnnData
from ._plot_basic.scatter_plt import scatter
import seaborn
from typing import Optional, Sequence, Union
from matplotlib.axes import Axes


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
        result = get_degs_res(adata, data_key=key, group_key=group_name)
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

        ax.set_title('{} vs. {}'.format(group_name, "Others"))
        if count >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel('ranking')

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax.set_ylabel('score')

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)
