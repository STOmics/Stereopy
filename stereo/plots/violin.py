#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: wenzhenbin
@file:violin.py
@time:2023/12/06
"""
import collections.abc as cabc
from collections import OrderedDict
from typing import (
    Optional,
    Iterable,
    Tuple,
    Union,
    List,
    Literal
)
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import SubplotParams as sppars
from scipy.sparse import spmatrix

from stereo.constant import (
    TOTAL_COUNTS,
    PCT_COUNTS_MT,
    N_GENES_BY_COUNTS
)


def _check_indices(
        data_df: pd.DataFrame,
        alt_index: pd.Index,
        dim: Literal['cells', 'var'],
        keys: List[str],
        alias_index: Optional[pd.Index] = None,
        use_raw: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Common logic for checking indices for obs_df and var_df."""
    if use_raw:
        alt_repr = "adata.tl.raw"
    else:
        alt_repr = "adata"

    alt_dim = ("cells", "var")[dim == "cells"]

    if alias_index is not None:
        alt_names = pd.Series(alt_index, index=alias_index)
        alias_name = alias_index.name
        alt_search_repr = f"{alt_dim}['{alias_name}']"
    else:
        alt_names = pd.Series(alt_index, index=alt_index)
        alt_search_repr = f"{alt_dim}_names"

    col_keys = []
    index_keys = []
    index_aliases = []
    not_found = []

    # check that adata.obs does not contain duplicated columns
    # if duplicated columns names are present, they will
    # be further duplicated when selecting them.
    if not data_df.columns.is_unique:
        dup_cols = data_df.columns[data_df.columns.duplicated()].tolist()
        raise ValueError(
            f"adata.{dim} contains duplicated columns. Please rename or remove "
            "these columns first.\n`"
            f"Duplicated columns {dup_cols}"
        )

    if not alt_index.is_unique:
        raise ValueError(
            f"{alt_repr}.{alt_dim}_names contains duplicated items\n"
            f"Please rename these {alt_dim} names first for example using "
            f"`adata.{alt_dim}_names_make_unique()`"
        )

    # use only unique keys, otherwise duplicated keys will
    # further duplicate when reordering the keys later in the function
    for key in np.unique(keys):
        if key in data_df.columns:
            col_keys.append(key)
            if key in alt_names.index:
                raise KeyError(f"The key '{key}' is found in both adata.{dim} and {alt_repr}.{alt_search_repr}.")
        elif key in alt_names.index:
            val = alt_names[key]
            if isinstance(val, pd.Series):
                assert alias_index is not None
                raise KeyError(f"Found duplicate entries for '{key}' in {alt_repr}.{alt_search_repr}.")
            index_keys.append(val)
            index_aliases.append(key)
        else:
            not_found.append(key)
    if len(not_found) > 0:
        raise KeyError(
            f"Could not find keys '{not_found}' in columns of `adata.{dim}` or in"
            f" {alt_repr}.{alt_search_repr}."
        )

    return col_keys, index_keys, index_aliases

def _check_order(order, count):
    if order is None:
        return [None] * count
    elif not isinstance(order, (list, np.ndarray, pd.Index)):
        raise ValueError("order must be a list, np.ndarray, pd.Index or None")
    
    if isinstance(order, pd.Index):
        order = [order] * count
    elif isinstance(order, np.ndarray):
        if order.ndim == 1:
            order = [order] * count
        elif order.ndim == 2:
            if order.shape[0] < count:
                raise ValueError("order must have the same number of rows as keys")
        else:
            raise ValueError("order must be 1D or 2D")
    elif isinstance(order, list):
        if len(order) == 0:
            return [None] * count
        elif isinstance(order[0], (list, np.ndarray, pd.Index)):
            if len(order) != count:
                raise ValueError("order must have the same number of elements as keys")
        elif isinstance(order[0], (str, int, float, np.number)):
            order = [order] * count
    return order


def _get_array_values(
        X,
        dim_names: pd.Index,
        keys: List[str],
        axis: Literal[0, 1],
        backed: bool,
):
    mutable_idxer = [slice(None), slice(None)]
    idx = dim_names.get_indexer(keys)

    # for backed AnnData is important that the indices are ordered
    if backed:
        idx_order = np.argsort(idx)
        rev_idxer = mutable_idxer.copy()
        mutable_idxer[axis] = idx[idx_order]
        rev_idxer[axis] = np.argsort(idx_order)
        matrix = X[tuple(mutable_idxer)][tuple(rev_idxer)]
    else:
        mutable_idxer[axis] = idx
        matrix = X[tuple(mutable_idxer)]

    from scipy.sparse import issparse

    if issparse(matrix):
        matrix = matrix.toarray()

    return matrix


def get_data_attr(adata, keys, use_raw, gene_symbols: str = None, obsm_keys: Iterable[Tuple[str, int]] = ()):
    """Obtain relevant data from data and perform related processing."""
    if use_raw:
        var = pd.DataFrame(columns=[], index=adata.to_df().columns)
    else:
        var = adata.genes.to_df()

    if gene_symbols is not None:
        alias_index = pd.Index(var[gene_symbols])
    else:
        alias_index = None
    obs_cols, var_idx_keys, var_symbols = _check_indices(
        adata.cells.to_df(),
        var.index,
        "cells",
        keys,
        alias_index=alias_index,
        use_raw=use_raw,
    )

    # Make df
    df = pd.DataFrame(index=adata.cells.cell_name)

    # add var values
    if len(var_idx_keys) > 0:
        if use_raw:
            if not adata.tl.raw:
                raise Exception('The tl.raw_checkpoint() command is not executed, there is no data in raw!')
            df = pd.concat([df, adata.tl.raw.to_df()[var_idx_keys]], axis=1)
        else:
            df = pd.concat([df, adata.to_df()[var_idx_keys]], axis=1)

    # add obs values
    if len(obs_cols) > 0:
        df = pd.concat([df, adata.cells.to_df()[obs_cols]], axis=1)

    # reorder columns to given order (including duplicates keys if present)
    if keys:
        df = df[keys]

    for k, idx in obsm_keys:
        added_k = f"{k}-{idx}"
        val = adata.obsm[k]
        if isinstance(val, np.ndarray):
            df[added_k] = np.ravel(val[:, idx])
        elif isinstance(val, spmatrix):
            df[added_k] = np.ravel(val[:, idx].toarray())
        elif isinstance(val, pd.DataFrame):
            df[added_k] = val.loc[:, idx]

    return df


def violin_distribution(
        data,
        keys: Union[str, Iterable] = None,
        x_label: str = '',
        y_label: Optional[str] = None,
        show_stripplot: Optional[bool] = True,
        jitter: Optional[float] = 0.2,
        dot_size: Optional[float] = 0.8,
        log: Optional[bool] = False,
        rotation_angle: Optional[int] = 0,
        group_by: Optional[str] = None,
        multi_panel: bool = None,
        scale: Literal['area', 'count', 'width'] = 'width',
        ax: Optional[Axes] = None,
        order: Optional[Iterable[str]] = None,
        use_raw: Optional[bool] = False,
        palette: Optional[str] = None,
        title: Optional[str] = None
):  # Violin Statistics Chart
    """
    violin plot showing quality control index distribution

    :param data: StereoExpData object.
    :param keys: the figure width in pixels.
    :param x_label: x label.
    :param y_label: y label.
    :param show_stripplot: whether to overlay a stripplot of specific percentage values.
    :param jitter: adjust the dispersion of points.
    :param dot_size: dot size.
    :param log: plot a graph on a logarithmic axis.
    :param rotation_angle: rotation of xtick labels.
    :param group_by: the key of the observation grouping to consider.
    :param multi_panel: Display keys in multiple panels also when groupby is not None.
    :param scale: The method used to scale the width of each violin. If 'width' (the default), each violin will
            have the same width. If 'area', each violin will have the same area.
            If 'count', a violin's width corresponds to the number of observations.
    :param ax: a matplotlib axes object. only works if plotting a single component.
    :param order: Order in which to show the categories.
    :param use_raw: Whether to use raw attribute of adata. Defaults to True if .raw is present.
    :param palette: color theme.
    :param title: the title.

    :return: None
    """
    import seaborn as sns

    if isinstance(keys, str):
        keys = [keys]
    keys = list(OrderedDict.fromkeys(keys))
    if isinstance(y_label, (str, type(None))):
        y_label = [y_label] * (1 if group_by is None else len(keys))

    if group_by is None:
        if len(y_label) != 1:
            raise ValueError(f'Expected number of y-labels to be `1`, found `{len(y_label)}`.')
    elif len(y_label) != len(keys):
        raise ValueError(f'Expected number of y-labels to be `{len(keys)}`, found `{len(y_label)}`.')

    if group_by is not None:
        if group_by in [TOTAL_COUNTS, N_GENES_BY_COUNTS, PCT_COUNTS_MT]:
            raise Exception(
                f'group_by should not in `[{TOTAL_COUNTS, N_GENES_BY_COUNTS, PCT_COUNTS_MT}]`.'
            )
        obs_df = get_data_attr(data, keys=[group_by] + keys, use_raw=use_raw)
    else:
        obs_df = get_data_attr(data, keys=keys, use_raw=use_raw)

    if group_by is None:
        obs_df, x, ys = pd.melt(obs_df, value_vars=keys), 'variable', ['value']
    else:
        x, ys = group_by, keys

    index_ys = ys[0]
    if not isinstance(obs_df.iloc[0][index_ys], float):
        obs_df[index_ys] = pd.to_numeric(obs_df[index_ys], errors='coerce')

    if title and not isinstance(title, list):
        title = [title]

    if multi_panel and group_by is None and len(ys) == 1:
        y = ys[0]
        g = sns.catplot(
            y=y,
            data=obs_df,
            kind="violin",
            scale=scale,
            col=x,
            col_order=keys,
            sharey=False,
            order=keys,
            cut=0,
            inner=None,
            palette=palette
        )
        fig = g.figure

        if show_stripplot:
            grouped_df = obs_df.groupby(x)
            for ax_id, key in zip(range(g.axes.shape[1]), keys):
                sns.stripplot(
                    y=y,
                    data=grouped_df.get_group(key),
                    jitter=jitter,
                    size=dot_size,
                    color="black",
                    ax=g.axes[0, ax_id],
                )
        if log:
            g.set(yscale='log')
        g.set_titles(col_template='{col_name}').set_xlabels('')
        if title:
            for ax in g.axes[0]:
                ax.set_title(title.pop(0) if title else '')
        if rotation_angle is not None:
            for ax in g.axes[0]:
                ax.tick_params(axis='x', labelrotation=rotation_angle)
    else:
        if ax is None:
            axs = setup_axes(ax=ax, panels=['x'] if group_by is None else keys, show_ticks=True, right_margin=0.3, )[0]
        else:
            axs = [ax]
        fig = axs[0].figure
        orders = _check_order(order, len(ys))
        for ax, y, ylab, order in zip(axs, ys, y_label, orders):
            ax = sns.violinplot(x=x, y=y, data=obs_df, order=order, orient='vertical', scale=scale, ax=ax,
                                palette=palette)
            if show_stripplot:
                ax = sns.stripplot(
                    x=x,
                    y=y,
                    data=obs_df,
                    order=order,
                    jitter=jitter,
                    color='black',
                    size=dot_size,
                    ax=ax,
                )
            if x_label == '' and group_by is not None and rotation_angle is None:
                x_label = group_by.replace('_', ' ')
            ax.set_xlabel(x_label)
            if ylab is not None:
                ax.set_ylabel(ylab)
            if log:
                ax.set_yscale('log')
            if rotation_angle is not None:
                ax.tick_params(axis='x', labelrotation=rotation_angle)
            if title:
                ax.set_title(title.pop(0) if title else '')
    return fig
    # pl.show()


def setup_axes(
        ax: Union[Axes, Sequence[Axes]] = None,
        panels='blue',
        colorbars=False,
        right_margin=None,
        left_margin=None,
        projection: Literal['2d', '3d'] = '2d',
        show_ticks=False,
):
    """Grid of axes for plotting, legends and colorbars."""
    if projection not in {"2d", "3d"}:
        raise ValueError(f"Projection must be '2d' or '3d', was '{projection}'.")

    if left_margin is not None:
        raise NotImplementedError("We currently don't support `left_margin`.")

    if np.any(colorbars) and right_margin is None:
        right_margin = 1 - rcParams['figure.subplot.right'] + 0.21
    elif right_margin is None:
        right_margin = 1 - rcParams['figure.subplot.right'] + 0.06

    # make a list of right margins for each panel
    if not isinstance(right_margin, list):
        right_margin_list = [right_margin for i in range(len(panels))]
    else:
        right_margin_list = right_margin

    # make a figure with len(panels) panels in a row side by side
    top_offset = 1 - rcParams['figure.subplot.top']
    bottom_offset = 0.15 if show_ticks else 0.08
    left_offset = 1 if show_ticks else 0.3
    base_height = rcParams['figure.figsize'][1]
    height = base_height
    base_width = rcParams['figure.figsize'][0]
    if show_ticks:
        base_width *= 1.1

    draw_region_width = (base_width - left_offset - top_offset - 0.5)
    right_margin_factor = sum([1 + right_margin for right_margin in right_margin_list])
    width_without_offsets = (right_margin_factor * draw_region_width)
    right_offset = (len(panels) - 1) * left_offset
    figure_width = width_without_offsets + left_offset + right_offset
    draw_region_width_frac = draw_region_width / figure_width
    left_offset_frac = left_offset / figure_width
    if ax is None:
        pl.figure(figsize=(figure_width, height), subplotpars=sppars(left=0, right=1, bottom=bottom_offset))
    left_positions = [left_offset_frac, left_offset_frac + draw_region_width_frac]
    for i in range(1, len(panels)):
        right_margin = right_margin_list[i - 1]
        left_positions.append(left_positions[-1] + right_margin * draw_region_width_frac)
        left_positions.append(left_positions[-1] + draw_region_width_frac)
    panel_pos = [[bottom_offset], [1 - top_offset], left_positions]

    axs = []
    if ax is None:
        for icolor, color in enumerate(panels):
            left = panel_pos[2][2 * icolor]
            bottom = panel_pos[0][0]
            width = draw_region_width / figure_width
            height = panel_pos[1][0] - bottom
            if projection == '2d':
                ax = pl.axes([left, bottom, width, height])
            elif projection == '3d':
                ax = pl.axes([left, bottom, width, height], projection='3d')
            axs.append(ax)
    else:
        axs = ax if isinstance(ax, cabc.Sequence) else [ax]

    return axs, panel_pos, draw_region_width, figure_width
