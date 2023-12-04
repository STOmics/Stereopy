#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:30
# @Author  : zhangchao
# @File    : variance_test.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from anndata import AnnData
from scipy import stats


def variance_test(merge_data: AnnData, batch_key: str = "batch", test_key: str = "total_counts"):
    """Calculate the F-test
    Parameters
    ---------------
    merge_data: ``AnnData``
    batch_key: ``str``
        Used to mark data batches.
    test_key: ``str``
        Marks the value to be used for calculation.

    Returns
    ---------------
    res_df: ``DataFrame``
        the results of F-test
    fig_buffer: ``BytesIO``
        Boxplot stream
    """
    n_batch = merge_data.obs[batch_key].cat.categories.size
    n_sample = merge_data.shape[0]

    statistic, p = stats.kruskal(*[np.log1p(
        (merge_data[merge_data.obs[batch_key] == c].obs[test_key].values -
         merge_data[merge_data.obs[batch_key] == c].obs[test_key].values.min()) / (
                merge_data[merge_data.obs[batch_key] == c].obs[test_key].values.max() -
                merge_data[merge_data.obs[batch_key] == c].obs[test_key].values.min())) for c in
        merge_data.obs[batch_key].cat.categories.tolist()])

    f_test = stats.f.ppf(0.95, n_batch - 1, n_sample - n_batch)
    res_df = pd.DataFrame({"n_batch": n_batch,
                           "n_sample": n_sample,
                           "F": statistic,
                           "p value": np.around(p, decimals=4),
                           f"F Ref {n_batch - 1, n_sample - n_batch}": f_test}, index=["Score"])

    # UMICount Boxplot
    plt.figure(figsize=(12.8, 7.2))
    sn.boxplot(x="batch", y=test_key, data=merge_data.obs.sort_values("batch"))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Batch".title(), fontsize=20)
    plt.ylabel(f"{test_key}".title(), fontsize=20)
    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
    return res_df, fig_buffer
