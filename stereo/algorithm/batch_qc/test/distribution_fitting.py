#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:39
# @Author  : zhangchao
# @File    : distribution_fitting.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from collections import defaultdict
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats


def distribution_fitting(merge_data, batch_key="batch", fit_key="total_counts"):
    buffer_dict = defaultdict()
    for i, c in enumerate(sorted(merge_data.obs[batch_key].cat.categories)):
        tm_data = merge_data.obs[merge_data.obs[batch_key] == c][fit_key].values
        plt.figure(figsize=(12.8, 7.2))
        plt.subplot(121)
        sn.distplot(tm_data, fit=stats.norm)
        plt.xlabel("UMICount", fontsize=20)
        plt.ylabel("Density", fontsize=20)
        plt.xticks(fontsize=20, rotation=-35)
        plt.yticks(fontsize=20)
        plt.title(f"Batch{i} UMICount Histogram Curve", fontsize=20, pad=30)
        plt.subplot(122)
        stats.probplot(tm_data, plot=plt)
        plt.xlabel("Theoretocal Quantiles", fontsize=20)
        plt.ylabel("Ordered Values", fontsize=20)
        plt.xticks(fontsize=20, rotation=-35)
        plt.yticks(fontsize=20)
        plt.title(f"Batch{i} Probability Fitting Curve", fontsize=20, pad=30)
        plt.subplots_adjust(wspace=0.5, hspace=0)
        fig_buffer = BytesIO()
        plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
        buffer_dict[f"batch{c}"] = fig_buffer
    return buffer_dict
