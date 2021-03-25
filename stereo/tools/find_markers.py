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
from ..core.tool_base import ToolBase
from ..log_manager import logger
from ..core.stereo_result import FindMarkerResult
from tqdm import tqdm


class FindMarker(ToolBase):
    def __init__(self, data, cluster, test_groups='all', control_groups='rest', method='t-test', corr_method=None, name=None):
        super(FindMarker, self).__init__(data, method, name)
        self.params = locals()
        self.corr_method = corr_method.lower()
        self.test_group = test_groups
        self.control_group = control_groups
        self.cluster = cluster

    def check_param(self):
        """
        Check whether the parameters meet the requirements.
        :return:
        """
        super(FindMarker, self).check_param()
        if self.method not in ['t-test', 'wilcoxon']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')
        if self.corr_method not in ['bonferroni', 'benjamini-hochberg']:
            logger.error(f'{self.corr_method} is out of range, please check.')
            raise ValueError(f'{self.corr_method} is out of range, please check.')
        if self.cluster not in self.data.obs_keys():
            logger.error(f" '{self.cluster}' is not in andata.")
            raise ValueError(f" '{self.cluster}' is not in andata.")

    def fit(self):
        all_groups = set(self.data.obs[self.cluster].values)
        groups = all_groups if self.test_group == 'all' else [self.test_group]
        result_info = {}
        for g in tqdm(groups, desc='Find marker gene: '):
            if self.control_group == 'rest':
                other_g = all_groups.copy()
                other_g.remove(g)
            else:
                other_g = self.control_group
            g_data = select_group(andata=self.data, groups=g, clust_key=self.cluster)
            other_data = select_group(andata=self.data, groups=other_g, clust_key=self.cluster)
            if self.method == 't-test':
                result = t_test(g_data, other_data, self.corr_method)
            else:
                result = wilcoxon_test(g_data, other_data, self.corr_method)
            g_name = f"{g}.vs.{self.control_group}"
            params = self.params.copy()
            params['test_groups'] = g
            result_info[g_name] = FindMarkerResult(name=self.name, param=params, degs_data=result)
        self.add_result(result=result_info, key_added=self.name)


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
