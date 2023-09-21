#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:40
# @Author  : zhangchao
# @File    : qq_plot.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from collections import defaultdict
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.stats import stats


def qq_plot(merge_data, batch_key="batch", test_key="total_counts"):
    buffer_dict = defaultdict()
    batch_list = merge_data.obs[batch_key].cat.categories.tolist()
    static_df = pd.DataFrame(columns=["n-samples", "k-s stat", "p-value"])
    for i in range(len(batch_list)):
        for j in range(i + 1, len(batch_list)):
            if j == len(batch_list):
                break
            data = merge_data[merge_data.obs[batch_key].isin([batch_list[i], batch_list[j]])]
            data1 = data[data.obs[batch_key] == batch_list[i]].obs[test_key].values
            data2 = data[data.obs[batch_key] == batch_list[j]].obs[test_key].values
            df_pct = pd.DataFrame()
            df_pct[f'batch{i}'] = np.percentile(data1, range(100))
            df_pct[f'batch{j}'] = np.percentile(data2, range(100))
            plt.figure(figsize=(12.8, 7.2))

            plt.subplot(121)
            ax = plt.gca()
            ax.spines[["top", "right"]].set_visible(False)
            plt.scatter(x=f'batch{i}', y=f'batch{j}', data=df_pct, label='Actual fit')
            sn.lineplot(x=f'batch{i}', y=f'batch{i}', data=df_pct, color='r', label='Line of perfect fit')
            plt.xlabel(f'Quantile of UMICount, Batch{batch_list[i]}', fontsize=20)
            plt.ylabel(f'Quantile of UMICount, Batch{batch_list[j]}', fontsize=20)
            plt.xticks(fontsize=20, rotation=-35)
            plt.yticks(fontsize=20)
            plt.legend(loc="best", fontsize=14)
            plt.title("QQ plot", fontsize=20, pad=30)
            plt.grid()

            plt.subplot(122)
            ax = plt.gca()
            ax.spines[["top", "right"]].set_visible(False)
            tmp_df = pd.DataFrame()
            tmp_df[test_key] = np.sort(data.obs[test_key].unique())
            tmp_df[f"batch{batch_list[i]}"] = tmp_df[test_key].apply(
                lambda x: np.mean(data[data.obs[batch_key] == batch_list[i]].obs[test_key] <= x))
            tmp_df[f"batch{batch_list[j]}"] = tmp_df[test_key].apply(
                lambda x: np.mean(data[data.obs[batch_key] == batch_list[j]].obs[test_key] <= x))
            k = np.argmax(np.abs(tmp_df[f"batch{batch_list[i]}"] - tmp_df[f"batch{batch_list[j]}"]))
            ks_stats, p_val = stats.kstest(data[data.obs[batch_key] == batch_list[i]].obs[test_key],
                                           data[data.obs[batch_key] == batch_list[j]].obs[test_key])
            static_df.loc[f"batch{batch_list[i]}-{batch_list[j]}"] = [
                data.shape[0], np.around(ks_stats, decimals=4), np.around(p_val, decimals=4)]
            y = (tmp_df[f"batch{batch_list[i]}"][k] + tmp_df[f"batch{batch_list[j]}"][k]) / 2
            plt.plot(test_key, f"batch{batch_list[i]}", data=tmp_df, label=f"batch{batch_list[i]}")
            plt.plot(test_key, f"batch{batch_list[j]}", data=tmp_df, label=f"batch{batch_list[j]}")
            plt.errorbar(x=tmp_df[test_key][k], y=y, yerr=ks_stats / 2, color='r',
                         capsize=5, mew=3, label=f"Test statistic: {ks_stats:.4f}")
            plt.legend(loc='best', fontsize=14)
            plt.ylabel('Cumulative Density', fontsize=20)
            plt.xlabel('UMICount', fontsize=20)
            plt.xticks(fontsize=20, rotation=-35)
            plt.yticks(fontsize=20)
            plt.title("Kolmogorov-Smirnov Test", fontsize=20, pad=30)
            plt.grid()

            plt.subplots_adjust(wspace=0.5, hspace=0)
            fig_buffer = BytesIO()
            plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
            buffer_dict[f"batch{batch_list[i]}-{batch_list[j]}"] = fig_buffer
    static_df[
        "describe_note"] = "Kolmogorov-smirnov test (K-S Test) checks whether two data distributions are consistent."
    return buffer_dict, static_df
