#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:17
# @Author  : zhangchao
# @File    : metric_score.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import numpy as np
import pandas as pd

from ..score import get_kbet, get_lisi, get_ksim


def metric_score(merge_data, n_neighbor=100, batch_key="batch", metric_pos="X_umap", celltype_key=None):
    reject_score, stat_mean, pvalue_mean, accept_rate = get_kbet(merge_data, key=batch_key, use_rep=metric_pos,
                                                                 alpha=0.05,
                                                                 n_neighbors=n_neighbor)
    kbet_df = pd.DataFrame(data={"Chi Mean": stat_mean, "95% P Value": pvalue_mean, "Accept Rate": accept_rate,
                                 "Reject Rate": reject_score}, index=["Score"])
    kbet_df["describe_note"] = f"Local area Chi2 test. Local sample richness test. " \
                               f"By default, UMAP coordinates are used to calculate region neighbors. " \
                               f"default neighbors: {n_neighbor}"
    lisi_mean, lower, upper = get_lisi(merge_data, key=batch_key, use_rep=metric_pos, n_neighbors=n_neighbor)
    lisi_df = pd.DataFrame(data={"LISI Mean": lisi_mean, "95%CI Lower": lower, "95%CI Upper": upper}, index=[batch_key])

    if celltype_key is not None:
        lisi_mean, lower, upper = get_lisi(merge_data, key=celltype_key, use_rep=metric_pos, n_neighbors=n_neighbor)
        lisi_type = [lisi_mean, lower, upper]
    else:
        lisi_type = [np.NAN, np.NAN, np.NAN]
    lisi_df.loc[f"{celltype_key}".title()] = lisi_type
    lisi_df["describe_note"] = f"By default, UMAP coordinates are used to calculate region neighbors, " \
                               f"default n_neighbor: {n_neighbor}. For batch, a larger value indicates that " \
                               f"data of different batches in a local area is mixed more evenly."
    if celltype_key is not None:
        ksim_mean, ksim_accept_rate = get_ksim(merge_data, key=celltype_key, use_rep=metric_pos, beta=0.9,
                                               n_neighbors=n_neighbor)
        ksim_df = pd.DataFrame(
            data={"KSIM Mean": ksim_mean, "Accept Rate": ksim_accept_rate}, index=[f"{celltype_key}".title()])
    else:
        ksim_df = pd.DataFrame(data={"KSIM Mean": np.NAN, "Accept Rate": np.NAN}, index=[f"{celltype_key}".title()])
    ksim_df["describe_note"] = "The kSIM acceptance rate requires ground truth cell type information and measures " \
                               "whether the neighbors of a cell have the same cell type as it does. If a method " \
                               "overcorrects the batch effects, it will have a low kSIM acceptance rate."
    return {"kbet_df": kbet_df, "lisi_df": lisi_df, "ksim_df": ksim_df}
