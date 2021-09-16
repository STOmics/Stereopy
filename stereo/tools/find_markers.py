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
import pandas as pd

from ..utils.data_helper import select_group
from ..core.tool_base import ToolBase
from ..log_manager import logger
from tqdm import tqdm
from typing import Union, Sequence
import numpy as np
from ..plots.marker_genes import marker_genes_text, marker_genes_heatmap


class FindMarker(ToolBase):
    """
    a tool of finding maker gene
    for each group, find statistical test different genes between one group and the rest groups using t-test or wilcoxon_test

    :param data: expression matrix, StereoExpData object
    :param groups: group information matrix, at least two columns, treat first column as sample name, and the second as
    group name e.g pd.DataFrame({'bin_cell': ['cell_1', 'cell_2'], 'cluster': ['1', '2']})
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
            method: str = 't_test',
            case_groups: Union[str, np.ndarray] = 'all',
            control_groups: str = 'rest',
            corr_method: str = 'bonferroni',
    ):
        super(FindMarker, self).__init__(data=data, groups=groups, method=method)
        self.corr_method = corr_method.lower()
        self.case_groups = case_groups
        self.control_group = control_groups
        self.fit()

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['t_test', 'wilcoxon_test']
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

    @ToolBase.fit_log
    def fit(self):
        """
        run
        """
        self.data.sparse2array()
        if self.groups is None:
            raise ValueError(f'group information must be set')
        group_info = self.groups
        all_groups = set(group_info['group'].values)
        if isinstance(self.case_groups, np.ndarray):
            case_groups = set(self.case_groups)
        elif self.case_groups == 'all':
            case_groups = all_groups
        else:
            case_groups = [self.case_groups]
        control_str = self.control_group if isinstance(self.control_group, str) else \
            '-'.join([str(i) for i in self.control_group])
        self.result = {}
        logres_score = self.logres_score() if self.method == 'logres' else None
        for g in tqdm(case_groups, desc='Find marker gene: '):
            if self.control_group == 'rest':
                other_g = all_groups.copy()
                other_g.remove(g)
            else:
                other_g = [self.control_group]
            other_g = list(other_g)
            g_data = select_group(st_data=self.data, groups=g, cluster=group_info)
            other_data = select_group(st_data=self.data, groups=other_g, cluster=group_info)
            g_data, other_data = self.merge_groups_data(g_data, other_data)
            result = self.get_func_by_path('stereo.algorithm.statistics', self.method)(g_data, other_data,
                                                                                       self.corr_method) \
                if logres_score is None else self.run_logres(logres_score, g_data, other_data, g)
            self.result[f"{g}.vs.{control_str}"] = result

    def logres_score(self):
        from ..algorithm.statistics import logreg
        x = self.data.exp_matrix
        y = self.groups['group'].values
        if (isinstance(self.case_groups, str) and self.case_groups != 'all') or isinstance(self.case_groups, np.ndarray):
            use_group = [self.case_groups] if isinstance(self.case_groups, str) else list(self.case_groups)
            use_group.append(self.control_group)
            group_index = self.groups['group'].isin(use_group)
            x = x[group_index, :]
            y = y[group_index]
        score_df = logreg(x, y)
        score_df.columns = self.data.gene_names
        return score_df

    def run_logres(self, score_df, g_data, other_data, group_name):
        from ..algorithm.statistics import cal_log2fc
        res = pd.DataFrame()
        res['genes'] = g_data.columns
        gene_index = score_df.columns.isin(g_data.columns)
        scores = score_df.loc[group_name].values if score_df.shape[0] > 1 else score_df.values[0]
        res['scores'] = scores[gene_index]
        res['log2fc'] = cal_log2fc(g_data, other_data)
        return res

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

    def plot_marker_text(self,
                         groups: Union[str, Sequence[str]] = 'all',
                         markers_num: int = 20,
                         sort_key: str = 'scores',
                         ascend: bool = False,
                         fontsize: int = 8,
                         ncols: int = 4, ):
        marker_genes_text(self.result, groups, markers_num, sort_key, ascend, fontsize, ncols)

    def plot_heatmap(self,
                     markers_num: int = 5,
                     sort_key: str = 'scores',
                     ascend: bool = False,
                     show_labels: bool = True,
                     show_group: bool = True,
                     show_group_txt: bool = True,
                     cluster_colors_array=None,
                     min_value=None,
                     max_value=None,
                     gene_list=None, do_log=True):
        marker_genes_heatmap(data=self.data, cluster_res=self.groups, marker_res=self.result,
                                  markers_num=markers_num, sort_key=sort_key, ascend=ascend, show_labels=show_labels,
                                  show_group=show_group, show_group_txt=show_group_txt,
                                  cluster_colors_array=cluster_colors_array, min_value=min_value, max_value=max_value,
                                  gene_list=gene_list, do_log=do_log)
