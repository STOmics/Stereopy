#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:violin.py
@time:2021/11/02
"""
import matplotlib.pyplot as plt
import seaborn

from stereo.constant import N_GENES_BY_COUNTS
from stereo.constant import PCT_COUNTS_MT
from stereo.constant import TOTAL_COUNTS


def violin_distribution(data, width=None, height=None, y_label=None):  # Violin Statistics Chart
    """
    violin plot showing quality control index distribution

    :param data: StereoExpData object.
    :param width: the figure width in pixels.
    :param height: the figure height in pixels.
    :param y_label: y label

    :return: None
    """
    if width is None or height is None:
        figsize = (18, 4)
    else:
        width = width / 100 if width >= 100 else 18
        height = height / 100 if height >= 100 else 4
        figsize = (width, height)
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    seaborn.violinplot(y=data.cells.get_property(TOTAL_COUNTS), ax=axs[0])
    seaborn.violinplot(y=data.cells.get_property(N_GENES_BY_COUNTS), ax=axs[1])
    seaborn.violinplot(y=data.cells.get_property(PCT_COUNTS_MT), ax=axs[2])
    axs[0].set_ylabel(y_label[0], fontsize=15)
    axs[1].set_ylabel(y_label[1], fontsize=15)
    axs[2].set_ylabel(y_label[2], fontsize=15)
    return fig


def save_fig(output):
    plt.savefig(output, bbox_inches="tight")
