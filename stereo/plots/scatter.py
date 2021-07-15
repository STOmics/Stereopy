#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:scatter.py
@time:2021/04/14

change log:
    2021/07/12 params change. by: qindanhua.
"""
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex, Normalize
from matplotlib import gridspec
import numpy as np
import pandas as pd
from ._plot_basic.scatter_plt import scatter
from typing import Optional, Union


def plot_scatter(
        x: Optional[Union[np.ndarray, list]],
        y: Optional[Union[np.ndarray, list]],
        color_values: Optional[Union[np.ndarray, list]] = None,
        ax=None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color_bar: bool = False,
        plot_cluster: list = None,
        bad_color: str = "lightgrey",
        dot_size: int = None,
        color_list: Optional[Union[np.ndarray, list]] = None,
):  # scatter plot, 聚类后表达矩阵空间分布
    """
    scatter plotter

    :param x: x position values
    :param y: y position values
    :param color_values: each dot's values, use for color set, eg. ['1', '3', '1', '2']
    :param ax: matplotlib Axes object
    :param title: figure title
    :param x_label: x label
    :param y_label: y label
    :param color_bar: show color bar or not, color_values must be int array or list when color_bar is True
    :param plot_cluster: the name list of clusters to show.
    :param bad_color: the name list of clusters to show.
    :param dot_size: marker size.
    :param color_list: customized colors
    :return: matplotlib Axes object

    Example:
    -------

    >>> color_values = np.array(['g1', 'g3', 'g1', 'g2', 'g1'])
    >>> plot_scatter(np.array([2, 4, 5, 7, 9]), np.array([3, 4, 5, 6, 7]), color_values=color_values)

    OR
    >>> plot_scatter(np.array([2, 4, 5, 7, 9]), np.array([3, 4, 5, 6, 7]), color_values=np.array([0, 2, 3, 1, 1], color_bar=True)

    color_values must be int array or list when color_bar is True

    """
    if len(color_values) != len(x):
        raise ValueError(f'color values should have the same length with x, y')
    if dot_size is None:
        dot_size = 120000 / len(color_values)
    if ax is None:
        fig, ax = plt.subplots()
    if color_list is None:
        cmap = get_cmap()
        color_list = cmap.colors
    else:
        cmap = ListedColormap(color_list)
    cmap.set_bad(bad_color)
    # 把特定值改为 np.nan 之后，可以利用 cmap.set_bad("white") 来遮盖掉这部分数据

    spatial_data = np.array([x, y]).T

    group_category = pd.DataFrame(color_values)[0].astype(str).astype('category').values
    order = np.argsort(~pd.isnull(group_category), kind="stable")
    color_data = group_category[order]
    spatial_data = spatial_data[order, :]

    index2cate = {i: str(n) for n, i in enumerate(list(color_data.categories))}
    if not color_bar:
        color_data = color_data.map(index2cate)
        has_na = False
        if plot_cluster is None:
            plot_cluster = list(color_data.categories)
        cluster_n = len(np.unique(color_data))
        if len(color_list) < cluster_n:
            print(len(color_list))
            color_list = color_list * cluster_n
            print(len(color_list))
            cmap = ListedColormap(color_list)
            cmap.set_bad(bad_color)
        if len(color_data.categories) > len(plot_cluster):
            color_data = color_data.replace(color_data.categories.difference(plot_cluster), np.nan)
            has_na = True
        color_dict = {str(k): to_hex(v) for k, v in enumerate(color_list)}
        color_data = color_data.map(color_dict)

        if pd.api.types.is_categorical_dtype(color_data):
            color_data = pd.Categorical(color_data)
        if has_na:
            color_data = color_data.add_categories([to_hex(bad_color)])
            color_data = color_data.fillna(to_hex(bad_color))

    # color_data 是图像中各个点的值，也对应了每个点的颜色。data_points则对应了各个点的坐标
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    path_collection = scatter(
        spatial_data[:, 0],
        spatial_data[:, 1],
        ax=ax,
        marker=".",
        dot_colors=color_values if color_bar else color_data,
        dot_size=dot_size
    )
    if not color_bar:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.96, box.height])
        # -------------modified by qiuping1@genomics.cn-------------
        # valid_cate = color_data.categories
        # cat_num = len(adata.obs_vector(key).categories)
        # for label in adata.obs_vector(key).categories:
        categories = group_category.categories
        cat_num = len(categories)
        for label in categories:
            # --------modified end------------------
            ax.scatter([], [], c=color_dict[index2cate[label]], label=label)
        ax.legend(
            frameon=False,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if cat_num <= 14 else 2 if cat_num <= 30 else 3),
            # fontsize=legend_fontsize,
        )
    else:
        plt.colorbar(path_collection, ax=ax, pad=0.01, fraction=0.08, aspect=30)
    ax.autoscale_view()
    return ax
    # plt.show()


def plot_multi_scatter(
        x,
        y,
        color_values: Union[np.ndarray] = None,
        ncols: int = 2,
        title: Union[list, np.ndarray] = None,
        x_label: Union[list, np.ndarray] = None,
        y_label: Union[list, np.ndarray] = None,
        color_bar: bool = False,
        bad_color: str = "lightgrey",
        dot_size: int = None,
        color_list: Optional[Union[np.ndarray, list]] = None,
):
    """
    plot multiple scatters

    :param x: x position values
    :param y: y position values
    :param color_values: each dot's values, use for color set, eg. ['1', '3', '1', '2']
    :param ncols number of figure columns
    :param title: figure title
    :param x_label: x label
    :param y_label: y label
    :param color_bar: show color bar or not, color_values must be int array or list when color_bar is True
    :param bad_color: the name list of clusters to show.
    :param dot_size: marker size.
    :param color_list: customized colors

    :return: matplotlib Axes object

    """
    ncols = min(ncols, len(color_values))
    nrows = np.ceil(len(color_values) / ncols).astype(int)
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
    for i, cv in enumerate(color_values):
        ax = fig.add_subplot(axs[i])  # ax = plt.subplot(axs[i]) || ax = fig.add_subplot(axs[1, 1]))
        plot_scatter(x, y, cv,
                     ax=ax,
                     title=title[i] if title else None,
                     x_label=x_label[i] if x_label else None,
                     y_label=y_label[i] if y_label else None,
                     color_bar=color_bar,
                     bad_color=bad_color,
                     dot_size=dot_size,
                     color_list=color_list
                     )
    return fig
