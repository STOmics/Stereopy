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
from matplotlib.axes import Axes
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from typing import Optional, Union
import seaborn as sns
from ..stereo_config import stereo_conf


def base_scatter(
        x: Optional[Union[np.ndarray, list]],
        y: Optional[Union[np.ndarray, list]],
        hue: Optional[Union[np.ndarray, list]] = None,
        ax=None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color_bar: bool = False,
        color_bar_reverse: bool = False,
        bad_color: str = "lightgrey",
        dot_size: int = None,
        marker: str = 's',
        palette: Optional[Union[str, list]] = 'stereo_30',
        invert_y: bool = True,
        legend_ncol=2,
        show_legend=True,
        show_ticks=False,
        vmin=None,
        vmax=None,
        SegmentedColormap=None,
        hue_order=None,
        width=None,
        height=None,
        show_plotting_scale=False,
        plotting_scale_width=2000,
        data_resolution=None,
        data_bin_offset=1,
):  # scatter plot, Expression matrix spatial distribution after clustering
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
    :param color_bar_reverse: if True, reverse the color bar, defaults to False
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
        if width is None or height is None:
            figsize = (7, 7)
        else:
            width = width / 100 if width >= 100 else 7
            height = height / 100 if height >= 100 else 7
            figsize = (width, height)
        _, ax = plt.subplots(figsize=figsize)
    dot_size = 120000 / len(hue) if dot_size is None else dot_size
    # add a color bar

    if color_bar:
        colors = stereo_conf.linear_colors(palette, reverse=color_bar_reverse)
        cmap = ListedColormap(colors)
        cmap.set_bad(bad_color)

        sns.scatterplot(x=x, y=y, hue=hue, ax=ax, palette=cmap, size=hue, linewidth=0, marker=marker,
                        sizes=(dot_size, dot_size), vmin=vmin, vmax=vmax)
        if vmin is None and vmax is None:
            norm = plt.Normalize(hue.min(), hue.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            ax.figure.colorbar(sm)
        if ax.legend_ is not None:
            ax.legend_.remove()
    else:
        from natsort import natsorted
        import collections
        g = natsorted(set(hue))
        if hue_order is not None:
            g = hue_order
        colors = stereo_conf.get_colors(palette)
        color_dict = collections.OrderedDict(dict([(g[i], colors[i]) for i in range(len(g))]))
        sns.scatterplot(x=x, y=y, hue=hue, hue_order=g, linewidth=0, marker=marker,
                        palette=color_dict, size=hue, sizes=(dot_size, dot_size), ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()
        ax.legend(handles, labels, ncol=legend_ncol, bbox_to_anchor=(1.02, 1),
                  loc='upper left', borderaxespad=0, frameon=False)
        for lh in ax.legend_.legendHandles:
            lh.set_alpha(1)
            lh._sizes = [40]
    
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=15)  # set y-axis labels
    ax.set_xlabel(x_label, fontsize=15)  # set x-axis labels

    if invert_y:
        ax.invert_yaxis()

    if show_plotting_scale:
        min_x, max_x = np.min(x).astype(int), np.max(x).astype(int)
        min_y, max_y = np.min(y).astype(int), np.max(y).astype(int)

        ax_left, ax_right = ax.get_xlim()
        ax_bottom, ax_top = ax.get_ylim()

        plotting_scale_height = plotting_scale_width / 10

        horizontal_start_x = min_x
        bin_count = plotting_scale_width // data_bin_offset

        horizontal_end_x = horizontal_start_x + (bin_count - 1) * data_bin_offset
        horizontal_text_location_x = horizontal_start_x + plotting_scale_width / 2

        vertical_x_location = min_x - plotting_scale_height * 2
        vertical_text_location_x = vertical_x_location - plotting_scale_height
        # new_ax_left = vertical_text_location_x - plotting_scale_height * 3
        # ax.set_xlim(left=new_ax_left)
        if invert_y:
            horizontal_y_location = min_y - plotting_scale_height * 2
            vertical_start_y = min_y
            vertical_end_y = vertical_start_y + (bin_count - 1) * data_bin_offset
            vertical_text_location_y = vertical_start_y + plotting_scale_width / 2
            vertices = [
                (horizontal_start_x, horizontal_y_location - plotting_scale_height),
                (horizontal_start_x, horizontal_y_location),
                (horizontal_end_x, horizontal_y_location),
                (horizontal_end_x, horizontal_y_location - plotting_scale_height),
            ]
            horizontal_text_location_y = horizontal_y_location - plotting_scale_height
            # new_ax_top = horizontal_text_location_y - plotting_scale_height * 3
            # ax.set_ylim(top=new_ax_top)
        else:
            horizontal_y_location = max_y + plotting_scale_height * 2
            vertical_start_y = max_y
            vertical_end_y = vertical_start_y - (bin_count - 1) * data_bin_offset
            vertical_text_location_y = vertical_start_y - plotting_scale_width / 2
            vertices = [
                (horizontal_start_x, horizontal_y_location + plotting_scale_height),
                (horizontal_start_x, horizontal_y_location),
                (horizontal_end_x, horizontal_y_location),
                (horizontal_end_x, horizontal_y_location + plotting_scale_height),
            ]
            horizontal_text_location_y = horizontal_y_location + plotting_scale_height
            # new_ax_top = horizontal_text_location_y + plotting_scale_height * 3
            # ax.set_ylim(top=new_ax_top)

        vertices.extend([
            (vertical_x_location - plotting_scale_height, vertical_start_y),
            (vertical_x_location, vertical_start_y),
            (vertical_x_location, vertical_end_y),
            (vertical_x_location - plotting_scale_height, vertical_end_y)
        ])
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
        path = Path(vertices, codes)
        patch = PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)

        real_length = data_resolution * bin_count
        unit = 'nm'
        if real_length >= 1e9:
            real_length /= 1e9
            unit = 'm'
        elif real_length >= 1e6:
            real_length /= 1e6
            unit = 'mm'
        elif real_length >= 1e3:
            real_length /= 1e3
            unit = 'um'

                
        t1 = ax.text(
                x=horizontal_text_location_x,
                y=horizontal_text_location_y,
                s=f"{real_length}{unit}",
                # fontsize='small',
                rotation=0,
                horizontalalignment='center',
                verticalalignment='bottom'
        )
        t2 = ax.text(
                x=vertical_text_location_x,
                y=vertical_text_location_y,
                s=f"{real_length}{unit}",
                # fontsize='small',
                rotation=90,
                # rotation_mode='anchor',
                horizontalalignment='right',
                verticalalignment='center'
        )
        renderer = ax.get_figure().canvas.get_renderer()
        bbox  = t1.get_window_extent(renderer)
        trans = ax.transData.inverted()
        t1_top_left = trans.transform_point((bbox.x0, bbox.y1))
        if invert_y:
            if t1_top_left[1] <= ax_top:
                new_ax_top = ax_top - plotting_scale_height * 4
                ax.set_ylim(top=new_ax_top)
        else:
            if t1_top_left[1] >= ax_top:
                new_ax_top = ax_top + plotting_scale_height * 4
                ax.set_ylim(top=new_ax_top)

        bbox  = t2.get_window_extent(renderer)
        trans = ax.transData.inverted()
        t2_top_left = trans.transform_point((bbox.x0, bbox.y1))
        if t2_top_left[0] <= ax_left:
            new_ax_lef = ax_left - plotting_scale_height * 4
            ax.set_xlim(left=new_ax_lef)


    if not show_legend:
        ax.legend_.remove()

    if not show_ticks:
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_yticks([])
        ax.set_xticks([])
    
    return ax.get_figure()


def multi_scatter(
        x,
        y,
        hue: Union[np.ndarray, list] = None,
        ncols: int = 2,
        title: Union[list, np.ndarray] = None,
        x_label: Union[list, np.ndarray] = None,
        y_label: Union[list, np.ndarray] = None,
        color_bar: bool = False,
        color_bar_reverse: bool = False,
        bad_color: str = "lightgrey",
        dot_size: int = None,
        palette: Optional[Union[np.ndarray, list, str]] = 'stereo',
        vmin=None,
        vmax=None,
        width=None,
        height=None,
        show_plotting_scale=False,
        data_resolution=None,
        **kwargs
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
    hue_length = len(hue) if isinstance(hue, list) else hue.shape[0]
    ncols = min(ncols, hue_length)
    nrows = np.ceil(hue_length / ncols).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    if width is None or height is None:
        figsize = (ncols * 10, nrows * 8)
    else:
        width = width / 100 if width >= 100 else ncols * 8
        height = height / 100 if height >= 100 else nrows * 8
        figsize = (width, height)
    fig = plt.figure(figsize=figsize)
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
        if issparse(cv):
            cv = cv.toarray()[0]
        ax: Axes = fig.add_subplot(axs[i])  # ax = plt.subplot(axs[i]) || ax = fig.add_subplot(axs[1, 1]))
        base_scatter(x, y, cv,
                     ax=ax,
                     title=title[i] if title is not None and title != '' else None,
                     x_label=x_label[i] if x_label is not None and x_label != '' else None,
                     y_label=y_label[i] if y_label is not None and y_label != '' else None,
                     color_bar=color_bar,
                     color_bar_reverse=color_bar_reverse,
                     bad_color=bad_color,
                     dot_size=dot_size,
                     palette=palette,
                     vmin=vmin,
                     vmax=vmax,
                     show_plotting_scale=show_plotting_scale,
                     data_resolution=data_resolution,
                     **kwargs
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
        width=None, height=None
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
    if width is None or height is None:
        width, height = 6, 6
    else:
        width = width / 100 if width >= 100 else 6
        height = height / 100 if height >= 100 else 6
    fig, ax = plt.subplots(figsize=(width, height))
    ax: Axes = sns.scatterplot(
        data=data,
        x=x, y=y, hue=hue,
        hue_order=hue_order,
        palette=palette,
        alpha=alpha, s=s,
        ax=ax
    )
    # ax.figure.set_size_inches(width, height)
    ax.spines['right'].set_visible(False)  # remove right border
    ax.spines['top'].set_visible(False)  # remove top border
    ax.set_ylabel(y_label, fontweight='bold')  # set y-axis labels
    ax.set_xlabel(x_label, fontweight='bold')  # set x-axis labels

    if vlines:
        xmin = int(data['x'].min())
        xmax = int(np.percentile(np.array(data['x']), [90])[0])
        ymin = int(data['y'].min())
        ymax = int(np.percentile(np.array(data['y']), [90])[0])
        ax.vlines(-cut_off_logFC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1)  # draw vertical lines
        ax.vlines(cut_off_logFC, ymin, ymax, color='dimgrey', linestyle='dashed', linewidth=1)  # draw vertical lines
        ax.hlines(-np.log10(cut_off_pvalue), xmin, xmax, color='dimgrey', linestyle='dashed', linewidth=1)  # draw vertical lines
        # ax.set_xticks(range(xmin, xmax, 4))# set x-axis labels
        # ax.set_yticks(range(ymin, ymax, 2))# set y-axis labels
    if label and text_visible:
        data = data[data[text_visible]]
        for _, row in data.iterrows():
            ax.text(
                row[x] + 0.01,
                row[y],
                row[label],
                horizontalalignment='left',
                size='medium',
                color='black',
                # weight='semibold'
            )
    return fig


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
        fig = volcano(df, x='x', y='y', hue='group', label='genes', text_visible='label',
                     cut_off_pvalue=cut_off_pvalue,
                     cut_off_logFC=cut_off_logFC,
                     **kwargs)
    else:
        fig = volcano(df, x='x', y='y', hue='group',
                     cut_off_pvalue=cut_off_pvalue, cut_off_logFC=cut_off_logFC,
                     **kwargs)
    return fig


def highly_variable_genes(
        data: Optional[pd.DataFrame],
        width: int = None,
        height: int = None,
        xy_label: list = None,
        xyII_label: list = None
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
    if width is None or height is None:
        width, height = 12, 6
        height = 6
    else:
        width = width / 100 if width >= 100 else 12
        height = height / 100 if height >= 100 else 6
    fig = plt.figure(figsize=(width, height), clear=True)
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
    ax1.set_xlabel(xy_label[0], fontsize=15)
    ax1.set_ylabel(xy_label[1], fontsize=15)
    ax2.set_xlabel(xyII_label[0], fontsize=15)
    ax2.set_ylabel(xyII_label[1], fontsize=15)
    return fig
