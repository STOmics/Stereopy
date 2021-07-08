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
from typing import Union, Optional
import numpy as np
import pandas as pd
from ..core.stereo_result import StereoResult


class FindMarker(ToolBase):
    """
    a tool of finding maker gene
    for each group, find statistical test different genes between one group and the rest groups using t-test or wilcoxon_test

    :param data: expression matrix, StereoExpData object
    :param groups: group information matrix, at least two columns, treat first column as sample name, and the second as
    group name e.g pd.Dataframe({'bin_cell': ['cell_1', 'cell_2'], 'cluster': ['1', '2']})
    :param case_groups: default all clusters
    :param control_groups: rest of groups
    :param method: t-test or wilcoxon_test
    :param corr_method: correlation method

    Examples
    --------

    >>> from stereo.tools.find_markers import FindMarker
    >>> fm = FindMarker()
    """
    def __init__(
            self,
            data=None,
            groups=None,
            method: str = 't-test',
            case_groups: Union[str, np.ndarray] = 'all',
            control_groups: Union[str, np.ndarray] = 'rest',
            corr_method: str = 'bonferroni',
    ):
        super(FindMarker, self).__init__(data=data, groups=groups, method=method)
        self.corr_method = corr_method.lower()
        self.case_groups = case_groups
        self.control_group = control_groups

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
        self.groups = ct.result.matrix
        return ct.result.matrix

    def fit(self):
        """
        run
        """
        if self.groups is None:
            self.run_cluster()
        group_info = self.groups
        all_groups = set(group_info['group'].values)
        case_groups = all_groups if self.case_groups == 'all' else set(self.case_groups)
        for g in tqdm(case_groups, desc='Find marker gene: '):
            if self.control_group == 'rest':
                other_g = all_groups.copy()
                other_g.remove(g)
            else:
                other_g = self.control_group
            g_data = select_group(st_data=self.data, groups=g, cluster=group_info)
            other_data = select_group(st_data=self.data, groups=other_g, cluster=group_info)
            g_data, other_data = self.merge_groups_data(g_data, other_data)
            if self.method == 't-test':
                result = t_test(g_data, other_data, self.corr_method)
            else:
                result = wilcoxon_test(g_data, other_data, self.corr_method)
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
