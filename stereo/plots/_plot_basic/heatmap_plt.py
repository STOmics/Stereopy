#!/usr/bin/env python3
# coding: utf-8
"""
@author: Shixu He  heshixu@genomics.cn
@last modified by: Shixu He
@file:heatmap_plt.py
@time:2021/03/15
"""

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from typing import List, Iterable, Sequence, Optional, Tuple
from typing_extensions import Literal

from ...log_manager import logger


def heatmap(df: pd.DataFrame = None, ax: Axes = None, cmap=None, norm=None, plot_colorbar=False,
            colorbar_ax: Axes = None, show_labels=True, plot_hline=False, **kwargs):
    """
    :param df:
    :param ax:
    :param cmap:
    :param norm:
    :param plot_colorbar:
    :param colorbar_ax:
    :param show_labels:
    :param plot_hline:
    :param kwargs:
    :return:
    """

    if norm == None:
        norm = Normalize(vmin=None, vmax=None)
    if (plot_colorbar and colorbar_ax == None):
        logger.warning("Colorbar ax is not provided.")
        plot_colorbar = False

    kwargs.setdefault('interpolation', 'nearest')
    im = ax.imshow(df.values, aspect='auto', norm=norm, **kwargs)

    ax.set_ylim(df.shape[0] - 0.5, -0.5)
    ax.set_xlim(-0.5, df.shape[1] - 0.5)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.set_ylabel('')
    ax.grid(False)

    if show_labels:
        ax.tick_params(axis='x', labelsize='small')
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(list(df.columns), rotation=90)
    else:
        ax.tick_params(axis='x', labelbottom=False, bottom=False)

    if plot_colorbar:
        plt.colorbar(im, cax=colorbar_ax)

    if plot_hline:
        line_coord = (
                np.cumsum(df.index.value_counts(sort=False))[:-1] - 0.5
        )
        ax.hlines(
            line_coord,
            -0.5,
            df.shape[1] - 0.5,
            lw=1,
            color='black',
            zorder=10,
            clip_on=False,
        )


def plot_categories_as_colorblocks(
        groupby_ax: Axes,
        obs_tidy: pd.DataFrame,
        colors=None,
        orientation: Literal['top', 'bottom', 'left', 'right'] = 'left',
        cmap_name: str = 'tab20',
):
    """from scanpy"""

    groupby = obs_tidy.index.name
    from matplotlib.colors import ListedColormap, BoundaryNorm

    if colors is None:
        groupby_cmap = plt.get_cmap(cmap_name)
    else:
        groupby_cmap = ListedColormap(colors, groupby + '_cmap')
    norm = BoundaryNorm(np.arange(groupby_cmap.N + 1) - 0.5, groupby_cmap.N)

    # determine groupby label positions such that they appear
    # centered next/below to the color code rectangle assigned to the category
    value_sum = 0
    ticks = []  # list of centered position of the labels
    labels = []
    label2code = {}  # dictionary of numerical values asigned to each label
    for code, (label, value) in enumerate(
            obs_tidy.index.value_counts(sort=False).iteritems()
    ):
        ticks.append(value_sum + (value / 2))
        labels.append(label)
        value_sum += value
        label2code[label] = code

    groupby_ax.grid(False)

    if orientation == 'left':
        groupby_ax.imshow(
            np.array([[label2code[lab] for lab in obs_tidy.index]]).T,
            aspect='auto',
            cmap=groupby_cmap,
            norm=norm,
        )
        if len(labels) > 1:
            groupby_ax.set_yticks(ticks)
            groupby_ax.set_yticklabels(labels)

        # remove y ticks
        groupby_ax.tick_params(axis='y', left=False, labelsize='small')
        # remove x ticks and labels
        groupby_ax.tick_params(axis='x', bottom=False, labelbottom=False)

        # remove surrounding lines
        groupby_ax.spines['right'].set_visible(False)
        groupby_ax.spines['top'].set_visible(False)
        groupby_ax.spines['left'].set_visible(False)
        groupby_ax.spines['bottom'].set_visible(False)

        groupby_ax.set_ylabel(groupby)
    else:
        groupby_ax.imshow(
            np.array([[label2code[lab] for lab in obs_tidy.index]]),
            aspect='auto',
            cmap=groupby_cmap,
            norm=norm,
        )
        if len(labels) > 1:
            groupby_ax.set_xticks(ticks)
            if max([len(str(x)) for x in labels]) < 3:
                # if the labels are small do not rotate them
                rotation = 0
            else:
                rotation = 90
            groupby_ax.set_xticklabels(labels, rotation=rotation)

        # remove x ticks
        groupby_ax.tick_params(axis='x', bottom=False, labelsize='small')
        # remove y ticks and labels
        groupby_ax.tick_params(axis='y', left=False, labelleft=False)

        # remove surrounding lines
        groupby_ax.spines['right'].set_visible(False)
        groupby_ax.spines['top'].set_visible(False)
        groupby_ax.spines['left'].set_visible(False)
        groupby_ax.spines['bottom'].set_visible(False)

        groupby_ax.set_xlabel(groupby)

    return label2code, ticks, labels, groupby_cmap, norm


def plot_gene_groups_brackets(
        gene_groups_ax: Axes,
        group_positions: Iterable[Tuple[int, int]],
        group_labels: Sequence[str],
        left_adjustment: float = -0.3,
        right_adjustment: float = 0.3,
        rotation: Optional[float] = None,
        orientation: Literal['top', 'right'] = 'top',
):
    """from scanpy"""
    import matplotlib.patches as patches
    from matplotlib.path import Path

    # get the 'brackets' coordinates as lists of start and end positions

    left = [x[0] + left_adjustment for x in group_positions]
    right = [x[1] + right_adjustment for x in group_positions]

    # verts and codes are used by PathPatch to make the brackets
    verts = []
    codes = []
    if orientation == 'top':
        # rotate labels if any of them is longer than 4 characters
        if rotation is None and group_labels:
            if max([len(x) for x in group_labels]) > 4:
                rotation = 90
            else:
                rotation = 0
        for idx in range(len(left)):
            verts.append((left[idx], 0))  # lower-left
            verts.append((left[idx], 0.6))  # upper-left
            verts.append((right[idx], 0.6))  # upper-right
            verts.append((right[idx], 0))  # lower-right

            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)

            try:
                group_x_center = left[idx] + float(right[idx] - left[idx]) / 2
                gene_groups_ax.text(
                    group_x_center,
                    1.1,
                    group_labels[idx],
                    ha='center',
                    va='bottom',
                    rotation=rotation,
                )
            except:
                pass
    else:
        top = left
        bottom = right
        for idx in range(len(top)):
            verts.append((0, top[idx]))  # upper-left
            verts.append((0.15, top[idx]))  # upper-right
            verts.append((0.15, bottom[idx]))  # lower-right
            verts.append((0, bottom[idx]))  # lower-left

            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)

            try:
                diff = bottom[idx] - top[idx]
                group_y_center = top[idx] + float(diff) / 2
                if diff * 2 < len(group_labels[idx]):
                    # cut label to fit available space
                    group_labels[idx] = group_labels[idx][: int(diff * 2)] + "."
                gene_groups_ax.text(
                    0.6,
                    group_y_center,
                    group_labels[idx],
                    ha='right',
                    va='center',
                    rotation=270,
                    fontsize='small',
                )
            except Exception as e:
                print('problems {}'.format(e))
                pass

    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor='none', lw=1.5)

    gene_groups_ax.add_patch(patch)
    gene_groups_ax.grid(False)
    gene_groups_ax.axis('off')
    # remove y ticks
    gene_groups_ax.tick_params(axis='y', left=False, labelleft=False)
    # remove x ticks and labels
    gene_groups_ax.tick_params(
        axis='x', bottom=False, labelbottom=False, labeltop=False
    )


def _check_indices(
        dim_df: pd.DataFrame,
        alt_index: pd.Index,
        dim: "Literal['obs', 'var']",
        keys: List[str],
        alias_index: pd.Index = None,
        use_raw: bool = False,
):
    """from scanpy"""
    if use_raw:
        alt_repr = "adata.raw"
    else:
        alt_repr = "adata"

    alt_dim = ("obs", "var")[dim == "obs"]

    alias_name = None
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
    if not dim_df.columns.is_unique:
        dup_cols = dim_df.columns[dim_df.columns.duplicated()].tolist()
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
        if key in dim_df.columns:
            col_keys.append(key)
            if key in alt_names.index:
                raise KeyError(
                    f"The key '{key}' is found in both adata.{dim} and {alt_repr}.{alt_search_repr}."
                )
        elif key in alt_names.index:
            val = alt_names[key]
            if isinstance(val, pd.Series):
                # while var_names must be unique, adata.var[gene_symbols] does not
                # It's still ambiguous to refer to a duplicated entry though.
                assert alias_index is not None
                raise KeyError(
                    f"Found duplicate entries for '{key}' in {alt_repr}.{alt_search_repr}."
                )
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
