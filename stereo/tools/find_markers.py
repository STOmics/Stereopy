#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: find_markers.py
@time: 2021/3/14 14:52

change log:
    2021/05/20 rst supplement. by: qindanhua.
    2021/06/20 adjust for restructure base class . by: qindanhua.
"""
from ..utils.data_helper import select_group
from scipy import stats
import pandas as pd
from statsmodels.stats.multitest import multipletests
import numpy as np
from ..core.tool_base import ToolBase
from ..log_manager import logger
from ..core.stereo_result import StereoResult
from .clustering import Clustering
from tqdm import tqdm


class FindMarker(ToolBase):
    """
    a tool for finding maker gene
    define a cluster as group, find statistical test different genes between one group and the others using t-test or wilcoxon_test

    :param data: anndata
    # :param cluster: the 'Clustering' tool name, defined when running 'Clustering' tool
    :param test_groups: default all clusters
    :param control_groups: rest of groups
    :param method: t-test or wilcoxon_test
    :param corr_method: correlation method
    # :param name: name of this tool and will be used as a key when adding tool result to andata object.
    """
    def __init__(
            self,
            data=None,
            method: str = 't-test',
            cluster=None,
            test_groups: str = 'all',
            control_groups: str = 'rest',
            corr_method: str = 'bonferroni',
    ):
        super(FindMarker, self).__init__(data=data, method=method)
        # self.params = self.get_params(locals())
        self.corr_method = corr_method.lower()
        self.test_group = test_groups
        self.control_group = control_groups
        self._cluster = cluster
        self._check_params()

    @property
    def cluster(self):
        return self._cluster

    @cluster.setter
    def cluster(self, cluster):
        self._cluster = self.check_input_data(cluster)

    def run_cluster(self, method='louvain'):
        ct = Clustering(self.data, method=method)
        ct.fit()
        self.cluster = ct.result.matrix

    def _check_params(self):
        """
        Check whether the parameters meet the requirements.
        """
        # super(FindMarker, self).check_param()
        if self.method not in ['t-test', 'wilcoxon']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')
        if self.corr_method not in ['bonferroni', 'benjamini-hochberg']:
            logger.error(f'{self.corr_method} is out of range, please check.')
            raise ValueError(f'{self.corr_method} is out of range, please check.')
        # if self.cluster is None:
        #     logger.error(f" '{self.cluster}' is not in andata.")
        #     raise ValueError(f" '{self.cluster}' is not in andata.")

    def fit(self):
        """
        run
        """
        if self.cluster is None:
            self.run_cluster()
        all_groups = set(self.cluster['cluster'])
        groups = all_groups if self.test_group == 'all' else [self.test_group]
        result_info = {}
        for g in tqdm(groups, desc='Find marker gene: '):
            if self.control_group == 'rest':
                other_g = all_groups.copy()
                other_g.remove(g)
            else:
                other_g = self.control_group
            g_data = select_group(andata=self.data, groups=g, cluster=self.cluster)
            other_data = select_group(andata=self.data, groups=other_g, cluster=self.cluster)
            g_data, other_data = self.merge_groups_data(g_data, other_data)
            if self.method == 't-test':
                result = t_test(g_data, other_data, self.corr_method)
            else:
                result = wilcoxon_test(g_data, other_data, self.corr_method)
            # g_name = f"{g}.vs.{self.control_group}"
            # params = self.params.copy()
            # params['test_groups'] = g
            # result_info[g_name] = StereoResult(data=result)
            self.result.matrix = result
        return self.result.matrix

    @staticmethod
    def merge_groups_data(g1, g2):
        """
        drop duplicated and the columns that all the values are 0

        :param g1:
        :param g2:
        :return:
        """
        g1 = g1.loc[:, ~g1.columns.duplicated()]
        g2 = g2.loc[:, ~g2.columns.duplicated()]
        zeros = list(set(g1.columns[g1.sum(axis=0) == 0]) & set(g2.columns[g2.sum(axis=0) == 0]))
        g1.drop(zeros, axis=1, inplace=True)
        g2.drop(zeros, axis=1, inplace=True)
        return g1, g2


def t_test(group, other_group, corr_method=None):
    """
    student t test

    :param group:
    :param other_group:
    :param corr_method:
    :return:
    """
    scores, pvals = stats.ttest_ind(group.values, other_group.values, axis=0, equal_var=False)
    result = {'genes': group.columns, 'scores': scores, 'pvalues': pvals}
    n_genes = len(group.columns)
    pvals_adj = corr_pvalues(pvals, corr_method, n_genes)
    if pvals_adj is not None:
        result['pvalues_adj'] = pvals_adj
    result['log2fc'] = cal_log2fc(group, other_group)
    return pd.DataFrame(result)


def corr_pvalues(pvals, method, n_genes):
    """
    calculate correlation's p values

    :param pvals:
    :param method:
    :param n_genes:
    :return:
    """
    pvals_adj = None
    if method == 'benjamini-hochberg':
        pvals[np.isnan(pvals)] = 1
        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    elif method == 'bonferroni':
        pvals_adj = np.minimum(pvals * n_genes, 1.0)
    return pvals_adj


def wilcoxon_test(group, other_group, corr_method=None):
    """
    wilcoxon_test

    :param group:
    :param other_group:
    :param corr_method:
    :return:
    """
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
