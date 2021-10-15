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
import seaborn as sns


colors = ['violet', 'turquoise', 'tomato', 'teal', 'tan', 'silver', 'sienna', 'red', 'purple', 'plum', 'pink',
              'orchid', 'orangered', 'orange', 'olive', 'navy', 'maroon', 'magenta', 'lime',
              'lightgreen', 'lightblue', 'lavender', 'khaki', 'indigo', 'grey', 'green', 'gold', 'fuchsia',
              'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral', 'chocolate', 'chartreuse', 'brown', 'blue', 'black',
              'beige', 'azure', 'aquamarine', 'aqua',
              ]


def base_scatter(
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
        invert_y: bool = False
):  # scatter plot, 聚类后表达矩阵空间分布
    """
    scatter plotter

    :param invert_y: whether to invert y-axis.
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
    >>> base_scatter(np.array([2, 4, 5, 7, 9]), np.array([3, 4, 5, 6, 7]), color_values=color_values)

    OR

    >>> base_scatter(np.array([2, 4, 5, 7, 9]), np.array([3, 4, 5, 6, 7]), color_values=np.array([0, 2, 3, 1, 1], color_bar=True)

    color_values must be int array or list when color_bar is True

    """
    if len(color_values) != len(x):
        raise ValueError(f'color values should have the same length with x, y')
    if dot_size is None:
        dot_size = 120000 / len(color_values)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    if color_list is None:
        cmap = get_cmap()
        color_list = cmap.colors
    else:
        color_list = list(color_list)
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
            color_list = color_list * cluster_n
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
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title, fontsize=18)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
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
    if invert_y:
        ax.invert_yaxis()
    return ax
    # plt.show()


def multi_scatter(
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
        base_scatter(x, y, cv,
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


def volcano(
        data: Optional[pd.DataFrame], x: Optional[str], y: Optional[str], hue: Optional[str],
        hue_order=('down', 'normal', 'up'),
        palette=("#377EB8", "grey", "#E41A1C"),
        alpha=1, s=15,
        label: Optional[str] = None, text_visible: Optional[str] = None,
        x_label='log2(fold change)', y_label='-log10(pvalue)',
        vlines=True, cut_off_pvalue=0.01, cut_off_logFC=1,
):
    """
    volcano plot

    :param data: data frame
    :param x: key in data, variables that specify positions on the x axes.
    :param y: key in data, variables that specify positions on the y axes.
    :param hue: key in data, variables that specify maker gene.
    :param hue_order:
    :param palette: color set
    :param alpha: visible alpha
    :param s: dot size
    :param label: key in data, variables that specify dot label
    :param text_visible: key in data, variables that specify to show this dot's label or not
    :param x_label:
    :param y_label:
    :param cut_off_pvalue: used when plot vlines
    :param cut_off_logFC: used when plot vlines
    :param vlines: plot vlines or not
    :return:
    """
    ax = sns.scatterplot(
        data=data,
        x=x, y=y, hue=hue,
        hue_order=hue_order,
        palette=palette,
        alpha=alpha, s=s,
    )
    ax.spines['right'].set_visible(False)  # 去掉右边框
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.set_ylabel(y_label, fontweight='bold')  # 设置y轴标签
    ax.set_xlabel(x_label, fontweight='bold')  # 设置x轴标签

    if vlines:
        xmin = int(data['x'].min())
        xmax = int(np.percentile(np.array(data['x']), [90])[0])
        ymin = int(data['y'].min())
        ymax = int(np.percentile(np.array(data['y']), [90])[0])
        ax.vlines(-cut_off_logFC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1)  # 画竖直线
        ax.vlines(cut_off_logFC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1)  # 画竖直线
        ax.hlines(-np.log10(cut_off_pvalue), xmin, xmax, color='dimgrey', linestyle='dashed', linewidth=1)  # 画竖水平线
        # ax.set_xticks(range(xmin, xmax, 4))# 设置x轴刻度
        # ax.set_yticks(range(ymin, ymax, 2))# 设置y轴刻度
    if label and text_visible:
        for line in range(0, data.shape[0]):
            if data[text_visible][line]:
                ax.text(
                    data[x][line] + 0.01,
                    data[y][line],
                    data[label][line],
                    horizontalalignment='left',
                    size='medium',
                    color='black',
                    # weight='semibold'
                )
    return ax


def marker_gene_volcano(
        data,
        text_genes=None,
        cut_off_pvalue=0.01,
        cut_off_logFC=1,
        **kwargs
):
    df = data
    if 'log2fc' not in df.columns or 'pvalues' not in df.columns:
        raise ValueError(f'data frame should content log2fc and pvalues columns')
    df['x'] = df['log2fc']
    df['y'] = -df['pvalues'].apply(np.log10)
    df.loc[(df.x > cut_off_logFC) & (df.pvalues < cut_off_pvalue), 'group'] = 'up'
    df.loc[(df.x < -cut_off_logFC) & (df.pvalues < cut_off_pvalue), 'group'] = 'down'
    df.loc[(df.x >= -cut_off_logFC) & (df.x <= cut_off_logFC) | (df.pvalues >= cut_off_pvalue), 'group'] = 'normal'
    if text_genes is not None:
        # df['text'] = df['group'] == 'up'
        df['label'] = df['genes'].isin(text_genes)
        ax = volcano(df, x='x', y='y', hue='group', label='genes', text_visible='label',
                     cut_off_pvalue=cut_off_pvalue,
                     cut_off_logFC=cut_off_logFC,
                     **kwargs)
    else:
        ax = volcano(df, x='x', y='y', hue='group',
                     cut_off_pvalue=cut_off_pvalue, cut_off_logFC=cut_off_logFC,
                     **kwargs)
    return ax


def highly_variable_genes(
        data: Optional[pd.DataFrame]
):
    """
    scatter of highly variable genes

    :param data: pd.DataFrame

    :return: figure object
    """
    seurat_v3_flavor = "variances_norm" in data.columns
    if seurat_v3_flavor:
        y_label = 'variances'
    else:
        y_label = 'dispersions'
    data['gene type'] = ['highly variable genes' if i else 'other genes' for i in data['highly_variable']]
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sns.scatterplot(x="means", y=y_label+'_norm',
                    hue='gene type',
                    hue_order=('highly variable genes', 'other genes'),
                    palette=("black", "#ccc"),
                    alpha=1,
                    s=15,
                    data=data, ax=ax1
                    )
    sns.scatterplot(x="means", y=y_label,
                    hue='gene type',
                    hue_order=('highly variable genes', 'other genes'),
                    palette=("black", "#ccc"),
                    alpha=1,
                    s=15,
                    data=data, ax=ax2
                    )
    ax1.set_xlabel('mean expression of genes', fontsize=15)
    ax1.set_ylabel('dispersions of genes (normalized)', fontsize=15)
    ax2.set_xlabel('mean expression of genes', fontsize=15)
    ax2.set_ylabel('dispersions of genes (not normalized)', fontsize=15)
    return fig
