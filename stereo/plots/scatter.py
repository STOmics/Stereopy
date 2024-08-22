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
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.sparse import issparse

from ..constant import PLOT_SCATTER_SIZE_FACTOR
from ..stereo_config import stereo_conf


def _plot_scale(
        x: np.ndarray,
        y: np.ndarray,
        ax: Axes,
        plotting_scale_width: int,
        data_bin_offset: int,
        data_resolution: int,
        invert_y: bool,
        boundary: list
):
    if boundary is None:
        min_x, max_x = np.min(x).astype(int), np.max(x).astype(int)
        min_y, max_y = np.min(y).astype(int), np.max(y).astype(int)
    else:
        min_x, max_x, min_y, max_y = boundary

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
    bbox = t1.get_window_extent(renderer)
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

    bbox = t2.get_window_extent(renderer)
    trans = ax.transData.inverted()
    t2_top_left = trans.transform_point((bbox.x0, bbox.y1))
    if t2_top_left[0] <= ax_left:
        new_ax_lef = ax_left - plotting_scale_height * 4
        ax.set_xlim(left=new_ax_lef)


def base_scatter(
        x: Optional[Union[np.ndarray, list]],
        y: Optional[Union[np.ndarray, list]],
        hue: Optional[Union[np.ndarray, list]] = None,
        ax: object = None,
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
        legend_ncol: int = 2,
        show_legend: bool = True,
        show_ticks: bool = False,
        vmin: float = None,
        vmax: float = None,
        hue_order: Union[list, np.ndarray] = None,
        width: float = None,
        height: float = None,
        boundary: list = None,
        show_plotting_scale: bool = False,
        plotting_scale_width: float = 2000,
        data_resolution: int = None,
        data_bin_offset: int = 1,
        foreground_alpha: float = None,
        base_image: np.ndarray = None,
        base_im_cmap: str = 'Greys',
        base_im_boundary: list = None,
        base_im_value_range: tuple = None,
        base_im_to_gray: bool = False
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
    :param dot_size: dot size.
    :param palette: customized colors
    :param legend_ncol: number of legend columns
    :param show_legend
    :param show_ticks
    :param vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
    :param vmax: The value representing the upper limit of the color scale. Values larger than vmax are plotted with the same color as vmax.
    :param width: the figure width in pixels.
    :param height: the figure height in pixels.

    :return: matplotlib Axes object

    color_values must be int array or list when color_bar is True

    """  # noqa
    if not ax:
        if width is None or height is None:
            figsize = (7, 7)
        else:
            width = width / 100 if width >= 100 else 7
            height = height / 100 if height >= 100 else 7
            figsize = (width, height)
        _, ax = plt.subplots(figsize=figsize)
    dot_size = PLOT_SCATTER_SIZE_FACTOR / len(hue) if dot_size is None else dot_size
    # add a color bar

    if invert_y:
        ax.invert_yaxis()

    if base_image is not None:
        if len(base_image.shape) == 3 and base_im_to_gray:
            from cv2 import cvtColor, COLOR_BGR2GRAY
            base_image = cvtColor(base_image[:, :, [2, 1, 0]], COLOR_BGR2GRAY)
        if len(base_image.shape) == 3 and base_image.dtype == np.uint16:
            if base_im_value_range is None:
                bmin, bmax = base_image.min(), base_image.max()
            else:
                bmin, bmax = base_im_value_range
            base_image = plt.Normalize(bmin, bmax)(base_image).data
        if len(base_image.shape) == 3:
            bg_pixel = np.array([0, 0, 0], dtype=base_image.dtype)
            if base_image.dtype == np.uint8:
                bg_value = 255
            else:
                bg_value = 1.0
            bg_mask = np.where(base_image == bg_pixel, bg_value, 0)
            base_image += bg_mask
        ax.imshow(base_image, cmap=base_im_cmap, extent=base_im_boundary)
        if foreground_alpha is None:
            foreground_alpha = 0.5
    else:
        if foreground_alpha is None:
            foreground_alpha = 1

    if color_bar:
        colors = stereo_conf.linear_colors(palette, reverse=color_bar_reverse)
        cmap = ListedColormap(colors)
        cmap.set_bad(bad_color)
        if vmin is None and vmax is None:
            norm = plt.Normalize(hue.min(), hue.max())
        else:
            vmin = hue.min() if vmin is None else vmin
            vmax = hue.max() if vmax is None else vmax
            norm = plt.Normalize(vmin, vmax)
        sns.scatterplot(x=x, y=y, hue=hue, ax=ax, palette=cmap, size=hue, linewidth=0, marker=marker,
                        sizes=(dot_size, dot_size), hue_norm=norm, alpha=foreground_alpha, legend=False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm)
        # if ax.legend_ is not None:
        #     ax.legend_.remove()
    else:
        from natsort import natsorted
        import collections
        g = natsorted(set(hue))
        if hue_order is None:
            hue_order = g
        # if isinstance(palette, (dict, collections.OrderedDict)):
        #     palette = [palette[i] for i in g if i in palette]
        # if len(palette) < len(g):
        #     colors = stereo_conf.get_colors(palette, n=len(g))
        # else:
        #     colors = palette
        # color_dict = collections.OrderedDict(dict([(g[i], colors[i]) for i in range(len(g))]))
        colors = stereo_conf.get_colors(palette, n=len(g), order=hue_order)
        color_dict = dict(zip(hue_order, colors))
        sns.scatterplot(x=x, y=y, hue=hue, hue_order=hue_order, linewidth=0, marker=marker,
                        palette=color_dict, size=hue, sizes=(dot_size, dot_size), ax=ax, alpha=foreground_alpha)
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend_.remove()
        legd = ax.legend(handles, labels, ncol=legend_ncol, bbox_to_anchor=(1.02, 1),
                  loc='upper left', borderaxespad=0, frameon=False)
        for lh in legd.legendHandles:
            # lh.set_alpha(1)
            lh._sizes = [40]

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=15)  # set y-axis labels
    ax.set_xlabel(x_label, fontsize=15)  # set x-axis labels

    if show_plotting_scale:
        _plot_scale(
            x, y, ax,
            plotting_scale_width, data_bin_offset,
            data_resolution, invert_y,
            boundary
        )

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
    :param vmin: min value to plot, default None means auto calculate.
    :param vmax: max value to plot, default None means auto calculate.
    :param width: the figure width in pixels.
    :param height: the figure height in pixels.

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
    )
    for i, cv in enumerate(hue):
        if issparse(cv):
            cv = cv.toarray()[0]
        ax: Axes = fig.add_subplot(axs[i])
        base_scatter(x,
                     y,
                     cv,
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
        data: Optional[pd.DataFrame],
        x: Optional[str],
        y: Optional[str],
        hue: Optional[str],
        hue_order=('down', 'normal', 'up'),
        palette=("#377EB8", "grey", "#E41A1C"),
        alpha=1, s=15,
        label: Optional[str] = None,
        text_visible: Optional[str] = None,
        x_label='log2(fold change)',
        y_label='-log10(pvalue)',
        vlines=True,
        cut_off_pvalue=0.01,
        cut_off_logFC=1,
        width=None,
        height=None
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
    :param x_label: the x label.
    :param y_label: the y label.
    :param cut_off_pvalue: used when plot vlines
    :param cut_off_logFC: used when plot vlines
    :param vlines: plot vlines or not
    :param width: the figure width in pixels.
    :param height: the figure height in pixels.

    :return: figure object
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
        ax.hlines(-np.log10(cut_off_pvalue), xmin, xmax, color='dimgrey', linestyle='dashed',
                  linewidth=1)  # draw vertical lines
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
            )
    return fig


def marker_gene_volcano(
        data,
        text_genes=None,
        cut_off_pvalue=0.01,
        cut_off_logFC=1,
        **kwargs
):
    """
    marker_gene_volcano plot

    :param data: data frame
    :param text_genes: show gene names.
    :param cut_off_pvalue: used when plot vlines
    :param cut_off_logFC: used when plot vlines
    :param kwargs:

    :return: figure object
    """
    df = data
    if 'log2fc' not in df.columns or 'pvalues' not in df.columns:
        raise ValueError('data frame should content log2fc and pvalues columns')
    drop_columns = ['x', 'y', 'group']
    df['x'] = df['log2fc']
    df['y'] = -df['pvalues'].apply(np.log10)
    df.loc[(df.x > cut_off_logFC) & (df.pvalues < cut_off_pvalue), 'group'] = 'up'
    df.loc[(df.x < -cut_off_logFC) & (df.pvalues < cut_off_pvalue), 'group'] = 'down'
    df.loc[(df.x >= -cut_off_logFC) & (df.x <= cut_off_logFC) | (df.pvalues >= cut_off_pvalue), 'group'] = 'normal'
    if text_genes is not None:
        df['label'] = df['genes'].isin(text_genes)
        fig = volcano(df, x='x', y='y', hue='group', label='genes', text_visible='label',
                      cut_off_pvalue=cut_off_pvalue, cut_off_logFC=cut_off_logFC, **kwargs)
        drop_columns.append('label')
    else:
        fig = volcano(df, x='x', y='y', hue='group', cut_off_pvalue=cut_off_pvalue, cut_off_logFC=cut_off_logFC,
                      **kwargs)
    df.drop(columns=drop_columns, inplace=True)
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
    :param width: the figure width in pixels.
    :param height: the figure height in pixels.
    :param xy_label: the x、y label of first figure.
    :param xyII_label: the x、y label of second figure.

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
    sns.scatterplot(
        x="means",
        y=y_label + '_norm',
        hue='gene type',
        hue_order=('highly variable genes', 'other genes'),
        palette=("black", "#ccc"),
        alpha=1,
        s=15,
        data=data,
        ax=ax1
    )
    sns.scatterplot(
        x="means",
        y=y_label,
        hue='gene type',
        hue_order=('highly variable genes', 'other genes'),
        palette=("black", "#ccc"),
        alpha=1,
        s=15,
        data=data,
        ax=ax2
    )
    ax1.set_xlabel(xy_label[0], fontsize=15)
    ax1.set_ylabel(xy_label[1], fontsize=15)
    ax2.set_xlabel(xyII_label[0], fontsize=15)
    ax2.set_ylabel(xyII_label[1], fontsize=15)
    return fig
