#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: find_markers.py
@time: 2021/3/14 14:52
"""
from ..utils.data_helper import select_group
from scipy import stats
import pandas as pd
from statsmodels.stats.multitest import multipletests
import numpy as np


def find_cluster_marker(andata, cluster, groups='all', other_groups='rest', method='t-test', corr_method=None,
                        result_key=None):
    if cluster not in andata.obs_keys():
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
    result_key = result_key if result_key else 'marker_genes'
    andata.uns[result_key] = result_info
    return andata


def t_test(group, other_group, corr_method=None):
    scores, pvals = stats.ttest_ind(group.values, other_group.values, axis=0, equal_var=False)
    result = {'genes': group.columns, 'scores': scores, 'pvalues': pvals}
    n_genes = len(group.columns)
    pvals_adj = corr_pvalues(pvals, corr_method, n_genes)
    if pvals_adj is not None:
        result['pvalues_adj'] = pvals_adj
    result['log2fc'] = cal_log2fc(group, other_group)
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
    result = group.apply(lambda x: pd.Series(stats.mannwhitneyu(x, other_group[x.name])), axis=0).transpose()
    result.columns = ['scores', 'pvalues']
    result['genes'] = list(result.index)
    n_genes = result.shape[0]
    pvals_adj = corr_pvalues(result['pvalues'], corr_method, n_genes)
    if pvals_adj is not None:
        result['pvalues_adj'] = pvals_adj
    result['log2fc'] = cal_log2fc(group, other_group)
    return pd.DataFrame(result)


def cal_log2fc(group, other_group):
    g_mean = np.mean(group.values, axis=0)
    other_mean = np.mean(other_group.values, axis=0)
    log2fc = g_mean - np.log2(other_mean + 10e-5)
    return log2fc


def top_k_marker(andata, group, other_group='rest', top_k_genes=10, result_key=None,
                 sort_key='pvalues', sort_order='decreasing'):
    result_key = result_key if result_key else 'marker_genes'
    if result_key not in andata.uns.keys():
        andata = find_cluster_marker(andata, group, other_group)
    data = andata.uns[result_key][f'{group}.vs.{other_group}']
    ascend = False if sort_order == 'decreasing' else True
    top_k_data = data.sort_values(by=sort_key, ascending=ascend).head(top_k_genes)
    return top_k_data
