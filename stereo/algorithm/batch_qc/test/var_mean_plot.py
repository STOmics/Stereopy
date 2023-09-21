#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:38
# @Author  : zhangchao
# @File    : var_mean_plot.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from io import BytesIO

import matplotlib.pyplot as plt


def var_mean_plot(merge_data, batch_key="batch"):
    plt.figure(figsize=(12.8, 7.2))
    plt.axline((0, 0), (1, 1), c="r", label="Ref Line")
    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)
    for c in merge_data.obs["batch"].cat.categories:
        tmp_data = merge_data[merge_data.obs[batch_key] == c]
        gene_mean = tmp_data.X.toarray().mean(0)
        gene_var = tmp_data.X.toarray().var(0)
        plt.scatter(gene_mean.ravel(), gene_var.ravel(), s=5, label=f"Batch{c}")
    plt.xticks(fontsize=20, rotation=-35)
    plt.yticks(fontsize=20)
    plt.xlabel("gene mean".title(), fontsize=20)
    plt.ylabel("gene variance".title(), fontsize=20)
    # plt.title("mean-variance scatter plot", pad=15)
    plt.legend(loc="best", fontsize=20)
    plt.grid()
    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
    return fig_buffer
