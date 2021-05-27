#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:qc.py
@time:2021/04/13
"""
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import numpy as np
import pandas as pd
import math
from anndata import AnnData
from ._plot_basic.scatter_plt import scatter
import seaborn


def plot_genes_count(
        adata: AnnData,
        x=["total_counts", "total_counts"],
        y=["pct_counts_mt", "n_genes_by_counts"],
        ncols=2,
        **kwargs):  # scatter plot, 线粒体分布图
    """
    quality control index distribution visualization

    :param adata: AnnData object.
    :param x: obs key pairs for drawing. For example, assume x=["a", "a", "b"], the output plots will \include "a-c", "a-d", "b-e".
    :param y: obs key pairs for drawing. For example, assume y=["c", "d", "e"].
    :param ncols: the columns of figures. the output plots will include "a-c", "a-d", "b-e".
    :return: None

    Example:

    .. code:: python

        plot_spatial_cluster(adata = adata)

    """
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


def plot_spatial_distribution(
        adata: AnnData,
        obs_key: list = ["total_counts", "n_genes_by_counts"],
        ncols=2,
        dot_size=None,
        color_list=None,
        invert_y=False
):  # scatter plot, 表达矩阵空间分布
    """
    spatial bin-cell distribution.

    :param adata: AnnData object.
    :param obs_key: specified obs key list, for example: ["total_counts", "n_genes_by_counts"]
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param color_list: Color list.
    :param invert_y: whether to invert y-axis.
    :return: None

    Example:

    .. code:: python

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


def plot_violin_distribution(adata):  # 小提琴统计图
    """
    violin plot showing quality control index distribution

    :param adata: AnnData object.

    :return: None
    """
    _, axs = plt.subplots(1, 3, figsize=(15, 4))
    seaborn.violinplot(y=adata.obs['total_counts'], ax=axs[0])
    seaborn.violinplot(y=adata.obs['n_genes_by_counts'], ax=axs[1])
    seaborn.violinplot(y=adata.obs['pct_counts_mt'], ax=axs[2])
