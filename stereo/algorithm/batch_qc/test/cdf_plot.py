#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:33
# @Author  : zhangchao
# @File    : cdf_plot.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cdf_plot(merge_data, batch_key="batch", use_key="total_counts"):
    plt.figure(figsize=(12.8, 7.2))
    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)
    batch_categories = list(merge_data.obs[batch_key].cat.categories)
    dfs = []
    for c in batch_categories:
        df = np.cumsum(
            pd.Series(merge_data[merge_data.obs[batch_key] == c].obs[use_key]).value_counts(
                sort=False, normalize=True).sort_index(ascending=True))
        plt.plot(df.index, df, mfc="white", ms=5, label=f"Batch{c}")
        dfs.append(df)

    for i in range(len(batch_categories)):
        for j in range(i + 1, len(batch_categories)):
            temp_idx = list(set(dfs[i].index).intersection(dfs[j].index))
            diff = np.abs(dfs[i].loc[temp_idx].values - dfs[j].loc[temp_idx].values)
            idx = np.argmax(diff)
            plt.errorbar(x=temp_idx[idx], y=(dfs[i].loc[temp_idx[idx]] + dfs[j].loc[temp_idx[idx]]) / 2,
                         yerr=diff[idx] / 2, capsize=5, mew=3)
            plt.annotate(f"Maximum Difference: {diff[idx] / 2:.4f}",
                         xy=(temp_idx[idx], (dfs[i].loc[temp_idx[idx]] + dfs[j].loc[temp_idx[idx]]) / 2),
                         xytext=(temp_idx[idx], dfs[i].loc[temp_idx[idx]]),
                         fontsize=16,
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.3"))
    plt.grid(ls="--", lw=0.25, color="#4E616C")
    plt.xticks(fontsize=20, rotation=-35)
    plt.xlabel("UMICount", fontsize=20)
    plt.ylabel("Cumulative Density", fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="best", fontsize=16)
    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
    return fig_buffer
