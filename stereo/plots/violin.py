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


def violin_distribution(data):  # 小提琴统计图
    """
    violin plot showing quality control index distribution

    :param data: StereoExpData object.

    :return: None
    """
    _, axs = plt.subplots(1, 3, figsize=(18, 4))
    # plt.ylabel("total_counts")
    seaborn.violinplot(y=data.cells.get_property('total_counts'), ax=axs[0])
    # plt.ylabel("n_genes_by_counts")
    seaborn.violinplot(y=data.cells.get_property('n_genes_by_counts'), ax=axs[1])
    # plt.ylabel("pct_counts_mt")
    seaborn.violinplot(y=data.cells.get_property('pct_counts_mt'), ax=axs[2])
    axs[0].set_ylabel('total counts', fontsize=15)
    axs[1].set_ylabel('n genes by counts', fontsize=15)
    axs[2].set_ylabel('pct counts mt', fontsize=15)


def save_fig(output):
    plt.savefig(output, bbox_inches="tight")
