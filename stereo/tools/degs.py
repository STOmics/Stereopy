#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: degs.py
@time: 2021/3/14 14:52
"""
from ..utils.data_helper import select_group
from scipy import stats
import pandas as pd
from statsmodels.stats.multitest import multipletests
import numpy as np


def cal_degs(andata, cluster, groups='all', other_groups='rest', method='t-test', corr_method=None, result_key=None):
    if cluster not in andata.obs_names:
        raise ValueError(f" '{cluster}' is not in andata.")
    all_groups = set(andata.obs[cluster].values)
    groups = all_groups if groups == 'all' else [groups]
    result_info = {}
    for g in groups:
        if other_groups == 'rest':
            other_g = all_groups.copy()
            other_g.remove(g)
        else:
            other_g = other_groups
        g_data = select_group(andata=andata, groups=g, clust_key=cluster)
        other_data = select_group(andata=andata, groups=other_g, clust_key=cluster)
        if method == 't-test':
            result = t_test(g_data, other_data, corr_method)
        else:
            result = wilcoxon_test(g_data, other_data, corr_method)
        g_name = f"{g}.vs.{other_groups}"
        result_info[g_name] = result
    result_key = result_key if result_key else 'DEGs'
    andata.uns[result_key] = result_info


def t_test(group, other_group, corr_method=None):
    scores, pvals = stats.ttest_ind(group.values, other_group.values, axis=0, equal_var=False)
    result = {'genes': group.columns, 'scores': scores, 'pvalues': pvals}
    n_genes = len(group.columns)
    pvals_adj = corr_method(pvals, corr_method, n_genes)
    if pvals_adj:
        result['pvals_adj'] = pvals_adj
    return pd.DataFrame(result)


def corr_pvalues(pvals, method, n_genes):
    pvals_adj = None
    if method == 'benjamini-hochberg':
        pvals[np.isnan(pvals)] = 1
        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    elif method == 'bonferroni':
        pvals_adj = np.minimum(pvals * n_genes, 1.0)
    return pvals_adj


def wilcoxon_test(group, other_group, corr_method=None):
    result = group.apply(lambda x: pd.Series(stats.mannwhitneyu(x, other_group[x.name])), axis=0)
    result.columns = ['scores', 'pvalues']
    result['genes'] = group.columns
    n_genes = len(group.columns)
    pvals_adj = corr_method(result['pvalues'], corr_method, n_genes)
    if pvals_adj:
        result['pvals_adj'] = pvals_adj
    return pd.DataFrame(result)

