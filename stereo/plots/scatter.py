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
from matplotlib.colors import ListedColormap, to_hex, Normalize, LinearSegmentedColormap
from matplotlib import gridspec
import numpy as np
import pandas as pd
from typing import Optional, Union
import seaborn as sns
from ..config import StereoConfig

conf = StereoConfig()


def base_scatter(
        x: Optional[Union[np.ndarray, list]],
        y: Optional[Union[np.ndarray, list]],
        hue: Optional[Union[np.ndarray, list]] = None,
        ax=None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color_bar: bool = False,
        bad_color: str = "lightgrey",
        dot_size: int = None,
        palette: Optional[Union[str, list]] = 'stereo',
        invert_y: bool = True,
        legend_ncol=2,
        show_legend=True,
        show_ticks=False,
        vmin=None,
        vmax=None,
        SegmentedColormap=None,
):  # scatter plot, 聚类后表达矩阵空间分布
    """
    scatter plotter

    :param invert_y: whether to invert y-axis.
    :param x: x position values
    :param y: y position values
    :param hue: each dot's values, use for color set, eg. ['1', '3', '1', '2']
    :param ax: matplotlib Axes object
    :param title: figure title
    :param x_label: x label
    :param y_label: y label
    :param color_bar: show color bar or not, color_values must be int array or list when color_bar is True
    :param bad_color: the name list of clusters to show.
    :param dot_size: marker size.
    :param palette: customized colors
    :param legend_ncol: number of legend columns
    :param show_legend
    :param show_ticks
    :param vmin:
    :param vmax:

    :return: matplotlib Axes object

    color_values must be int array or list when color_bar is True

    """
    if not ax:
        _, ax = plt.subplots(figsize=(7, 7))
    dot_size = 120000 / len(hue) if dot_size is None else dot_size
    # add a color bar
    if color_bar:
        colors = conf.linear_colors(palette)
        cmap = ListedColormap(colors)
        cmap.set_bad(bad_color)

        sns.scatterplot(x=x, y=y, hue=hue, ax=ax, palette=cmap, size=hue, linewidth=0, marker="s",
                        sizes=(dot_size, dot_size), vmin=vmin, vmax=vmax)
        if vmin is None and vmax is None:
            norm = plt.Normalize(hue.min(), hue.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            ax.figure.colorbar(sm)
        ax.legend_.remove()
    else:
        from natsort import natsorted
        import collections
        g = natsorted(set(hue))
        colors = conf.get_colors(palette)
        color_dict = collections.OrderedDict(dict([(g[i], colors[i]) for i in range(len(g))]))
        sns.scatterplot(x=x, y=y, hue=hue, hue_order=g, linewidth=0, marker="s",
                        palette=color_dict, size=hue, sizes=(dot_size, dot_size), ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()
        ax.legend(handles, labels, ncol=legend_ncol, bbox_to_anchor=(1.02, 1),
                  loc='upper left', borderaxespad=0, frameon=False)
        for lh in ax.legend_.legendHandles:
            lh.set_alpha(1)
            lh._sizes = [40]
    if invert_y:
        ax.invert_yaxis()
    if not show_legend:
        ax.legend_.remove()

    if not show_ticks:
        ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=15)  # 设置y轴标签
    ax.set_xlabel(x_label, fontsize=15)  # 设置x轴标签
    if not show_ticks:
        ax.set_yticks([])
        ax.set_xticks([])
    return ax


def multi_scatter(
        x,
        y,
        hue: Union[np.ndarray, list] = None,
        ncols: int = 2,
        title: Union[list, np.ndarray] = None,
        x_label: Union[list, np.ndarray] = None,
        y_label: Union[list, np.ndarray] = None,
        color_bar: bool = False,
        bad_color: str = "lightgrey",
        dot_size: int = None,
        palette: Optional[Union[np.ndarray, list, str]] = 'stereo',
        vmin=None,
        vmax=None,
):
    """
    plot multiple scatters

    :param x: x position values
    :param y: y position values
    :param hue: each dot's values, use for color set, eg. ['1', '3', '1', '2']
    :param ncols number of figure columns
    :param title: figure title
    :param x_label: x label
    :param y_label: y label
    :param color_bar: show color bar or not, color_values must be int array or list when color_bar is True
    :param bad_color: the name list of clusters to show.
    :param dot_size: marker size.
    :param palette: customized colors
    :param vmin:
    :param vmax:

    :return: matplotlib Axes object

    """
    ncols = min(ncols, len(hue))
    nrows = np.ceil(len(hue) / ncols).astype(int)
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
    for i, cv in enumerate(hue):
        ax = fig.add_subplot(axs[i])  # ax = plt.subplot(axs[i]) || ax = fig.add_subplot(axs[1, 1]))
        base_scatter(x, y, cv,
                     ax=ax,
                     title=title[i] if title else None,
                     x_label=x_label[i] if x_label else None,
                     y_label=y_label[i] if y_label else None,
                     color_bar=color_bar,
                     bad_color=bad_color,
                     dot_size=dot_size,
                     palette=palette,
                     vmin=vmin,
                     vmax=vmax,
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
    sns.scatterplot(x="means", y=y_label + '_norm',
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
