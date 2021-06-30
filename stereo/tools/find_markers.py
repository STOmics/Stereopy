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
from ..core.tool_base import ToolBase
from ..log_manager import logger
from .clustering import Clustering
from tqdm import tqdm
from ..algorithm.statistics import t_test, wilcoxon_test


class FindMarker(ToolBase):
    """
    a tool of finding maker gene
    treat a cluster as a group, find statistical test different genes between one group and the others using t-test or wilcoxon_test

    :param data: expression matrix, pd.Dataframe or StereoExpData object
    :param test_groups: default all clusters
    :param control_groups: rest of groups
    :param method: t-test or wilcoxon_test
    :param corr_method: correlation method
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
        self.corr_method = corr_method.lower()
        self.test_group = test_groups
        self.control_group = control_groups
        self.cluster = self.result if cluster is None else cluster

    @property
    def cluster(self):
        return self._cluster

    @cluster.setter
    def cluster(self, cluster):
        self._cluster = self._check_input_data(cluster)
        if not self._cluster.check_columns(['cluster']):
            logger.error('cluster matrix should content a cluster columns')
            self._cluster = self.cluster

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['t-test', 'wilcoxon']
        self._method_check(method, m_range)

    @property
    def corr_method(self):
        return self._corr_method

    @corr_method.setter
    def corr_method(self, corr_method):
        if corr_method.lower() not in ['bonferroni', 'benjamini-hochberg']:
            logger.error(f'{self.corr_method} is out of range, please check.')
            raise ValueError(f'{self.corr_method} is out of range, please check.')
        else:
            self._corr_method = corr_method

    def run_cluster(self, method='louvain'):
        ct = Clustering(self.data, method=method)
        ct.fit()
        self.cluster = ct.result

    def fit(self):
        """
        run
        """
        if self.cluster.is_empty:
            self.run_cluster()
        cluster_matrix = self.cluster.matrix
        all_groups = set(cluster_matrix['cluster'])
        groups = all_groups if self.test_group == 'all' else [self.test_group]
        result_info = {}
        for g in tqdm(groups, desc='Find marker gene: '):
            if self.control_group == 'rest':
                other_g = all_groups.copy()
                other_g.remove(g)
            else:
                other_g = self.control_group
            g_data = select_group(st_data=self.data, groups=g, cluster=cluster_matrix)
            other_data = select_group(st_data=self.data, groups=other_g, cluster=cluster_matrix)
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
