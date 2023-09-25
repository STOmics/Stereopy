#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:42
# @Author  : zhangchao
# @File    : kernel_plot.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sn


def kernel_plot(merge_data, batch_key="batch", test_key="total_counts"):
    plt.figure(figsize=(12.8, 7.2))
    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)
    sn.kdeplot(x=test_key, data=merge_data.obs, hue=batch_key, common_norm=False)
    plt.xticks(fontsize=20, rotation=-35)
    plt.yticks(fontsize=20)
    plt.xlabel(test_key.title(), fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.grid()
    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
    return fig_buffer
