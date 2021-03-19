#!/usr/bin/env python3
# coding: utf-8
"""
@author: Shixu He  heshixu@genomics.cn
@last modified by: Shixu He
@file:scatter_plt.py
@time:2021/03/15
"""

from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from ...log_manager import logger

def scatter(x: list, y: list, ax: Axes = None, dot_colors=None, marker=".", dot_size=1, cmap = None, plotnonfinite=True, **kwargs):
    """
    Simplified scatter plot function, which wraps matplotlib.axes.Axes.scatter .
    :param x,y: Data position list.
    :param ax: plot axes.
    :param color: list of colors.
    :param marker: marker type.
    :param size: marker size.
    :param cmap: Color map.
    :param plotnonfinite: whether to plot bad point.
    :return: plot axes.
    """


    if (len(x)<=0 or len(y)<=0):
        logger.warning("x position or y position has no data.")
    if (dot_colors == None):
        dot_colors = ["gray"]
    if (isinstance(dot_colors, str)):
        dot_colors = [dot_colors]

    if (ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if (cmap == None):
        cmap = get_cmap()

    pathcollection = ax.scatter(
        x, y,
        c = dot_colors,
        marker = marker,
        s = dot_size,
        cmap = cmap,
        plotnonfinite = plotnonfinite,
        **kwargs,
    )

    return pathcollection

