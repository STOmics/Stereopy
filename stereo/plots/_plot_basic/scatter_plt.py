#!/usr/bin/env python3
# coding: utf-8
"""
@author: Shixu He  heshixu@genomics.cn
@last modified by: Shixu He
@file:scatter_plt.py
@time:2021/03/15

change log:
    2021/04/02 14:07  modified by qiuping, change the way of getting plot data of plot_cluster_result,
                      and remove it to ..marker_genes.py.
"""
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from ...log_manager import logger


def scatter(
        x: list,
        y: list,
        ax: Axes = None,
        dot_colors=None,
        marker=".",
        dot_size=1,
        cmap=None,
        plotnonfinite=True,
        **kwargs
):
    """
    Simplified scatter plot function, which wraps matplotlib.axes.Axes.scatter .
    :param x: Data position list.
    :param y: Data position list.
    :param ax: plot axes.
    :param dot_colors: list of colors.
    :param marker: marker type.
    :param dot_size: marker size.
    :param cmap: Color map.
    :param plotnonfinite: whether to plot bad point.
    :return: plot axes.
    """

    if len(x) <= 0 or len(y) <= 0:
        logger.warning("x position or y position has no data.")
    if dot_colors is None:
        dot_colors = ["gray"]
    if isinstance(dot_colors, str):
        dot_colors = [dot_colors]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if cmap is None:
        cmap = get_cmap()
    pathcollection = ax.scatter(
        x, y,
        c=dot_colors,
        marker=marker,
        s=dot_size,
        cmap=cmap,
        plotnonfinite=plotnonfinite,
        **kwargs,
    )

    return pathcollection
