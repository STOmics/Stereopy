#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: data_helper.py
@time: 2021/3/14 16:11
"""
from scipy.sparse import issparse
import pandas as pd


def select_group(andata, groups, clust_key):
    if clust_key not in andata.obs_names:
        raise ValueError(f" '{clust_key}' is not in andata.")
    all_groups = set(andata.obs[clust_key].values)
    groups = [groups] if isinstance(groups, str) else groups
    for g in groups:
        if g not in all_groups:
            raise ValueError(f"cluster {g} is not in all cluster.")
    group_index = andata.obs[clust_key].isin(groups)
    exp_matrix = andata.X.toarray() if issparse(andata.X) else andata.X
    group_sub = exp_matrix[group_index, :]
    obs = andata.obs_names[group_index]
    return pd.DataFrame(group_sub, index=obs, columns=list(andata.var_names))
