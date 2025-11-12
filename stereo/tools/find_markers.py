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
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
from joblib import cpu_count
from natsort import natsorted
from scipy import stats

from ..algorithm import mannwhitneyu
from ..algorithm import statistics
from ..core.tool_base import ToolBase
from ..log_manager import logger
from ..utils.data_helper import select_group
from ..utils.time_consume import log_consumed_time


class FindMarker(ToolBase):
    """
    a tool of finding maker gene
    for each group, find statistical test different genes between one group and the rest groups using t-test or wilcoxon_test # noqa

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
            case_groups: Union[str, np.ndarray, list, tuple] = 'all',
            control_groups: str = 'rest',
            corr_method: str = 'benjamini-hochberg',
            tie_term: bool = False,
            # raw_data=None,
            sort_by='scores',
            n_genes: Union[str, int] = 'all',
            ascending: bool = False,
            n_jobs: int = 4,
            pct: pd.DataFrame = None,
            pct_rest: pd.DataFrame = None,
            mean_count: pd.DataFrame = None
    ):
        super(FindMarker, self).__init__(data=data, groups=groups, method=method)
        self.corr_method = corr_method.lower()
        self.case_groups = case_groups
        self.control_groups = control_groups
        self.tie_term = tie_term
        # self.raw_data = raw_data
        self.sort_by = sort_by
        self.n_genes = n_genes
        self.ascending = ascending
        self.n_jobs = n_jobs
        self.result = {}
        self.result['pct'] = pct
        self.result['pct_rest'] = pct_rest
        self.result['mean_count'] = mean_count        
        self.fit()

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['t_test', 'wilcoxon_test', 'logreg']
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

    @property
    def control_groups(self):
        return self._control_groups

    @control_groups.setter
    def control_groups(self, control_groups):
        if isinstance(control_groups, str):
            self._control_groups = [control_groups] if control_groups != 'rest' else 'rest'
        elif isinstance(control_groups, int):
            self._control_groups = [str(control_groups)]
        elif isinstance(control_groups, (np.ndarray, list, tuple)):
            control_groups = list(set(control_groups))
            self._control_groups = [str(g) if isinstance(g, int) else g for g in control_groups]
        else:
            raise TypeError("The type of control_groups must be one of str, int, numpy.ndarray, list or tuple")
        if isinstance(self._control_groups, list):
            self._control_groups = natsorted(self._control_groups)

    @property
    def case_groups(self):
        return self._case_groups

    @case_groups.setter
    def case_groups(self, case_groups):
        if isinstance(case_groups, str):
            self._case_groups = [case_groups] if case_groups != 'all' else 'all'
        elif isinstance(case_groups, int):
            self._case_groups = [str(case_groups)]
        elif isinstance(case_groups, (np.ndarray, list, tuple)):
            case_groups = list(set(case_groups))
            self._case_groups = [str(g) if isinstance(g, int) else g for g in case_groups]
        else:
            raise TypeError("The type of case_groups must be one of str, int, numpy.ndarray, list or tuple")

    def handle_result(self, g, group_info, all_groups, ranks=None, tie_term=None, control_str='rest'):
        if self.control_groups == 'rest':
            other_g = all_groups.copy()
            other_g.remove(g)
        else:
            other_g = self.control_groups.copy()
            if g in other_g:
                other_g.remove(g)
        if len(other_g) <= 0:
            return

        g_index = select_group(groups=g, cluster=group_info, all_groups=all_groups)
        g_data = self.data.exp_matrix[g_index]
        if self.control_groups == 'rest':
            others_data = self.data.exp_matrix[~g_index]
        else:
            others_index = select_group(groups=other_g, cluster=group_info, all_groups=all_groups)
            others_data = self.data.exp_matrix[others_index]
        if self.method == 't_test':
            result = statistics.ttest(g_data, others_data, self.corr_method)
        elif self.method == 'logreg':
            if self.temp_logres_score is None:
                self.temp_logres_score = self.logres_score()
            result = self.run_logres(
                self.temp_logres_score,
                g_data,
                others_data,
                g
            )
        else:
            if self.control_groups != 'rest' and self.tie_term:
                xy = np.vstack((g_data, others_data))
                ranks = stats.rankdata(xy, axis=-1)
                tie_term = mannwhitneyu.cal_tie_term(ranks)
            result = statistics.wilcoxon(
                g_data,
                others_data,
                self.corr_method,
                ranks,
                tie_term,
                g_index
            )
        result['genes'] = self.data.gene_names
        if self.data.genes.real_gene_name is not None:
            result['gene_name'] = self.data.genes.real_gene_name
        result.sort_values(by=self.sort_by, ascending=self.ascending, inplace=True, ignore_index=True)

        if self.n_genes != 'all':
            if self.n_genes == 'auto':
                to = int(10000 / self.len_case_groups ** 2)
            else:
                to = self.n_genes
            to = min(max(to, 1), 50)
            result = result[:to]

        if self.control_groups != 'rest':
            control_str = '-'.join(other_g)
        self.result[f"{g}.vs.{control_str}"] = result
        pct = self.result['pct'].set_index('genes')
        pct_rest = self.result['pct_rest'].set_index('genes')
        self.result[f"{g}.vs.{control_str}"]['pct'] = pct.loc[result['genes']][g].to_numpy()
        self.result[f"{g}.vs.{control_str}"]['pct_rest'] = pct_rest.loc[result['genes']][g].to_numpy()
        self.result[f"{g}.vs.{control_str}"]['mean_count'] = self.result['mean_count'].loc[result['genes']][g].to_numpy()

    @ToolBase.fit_log
    def fit(self):
        """
        run
        """
        if self.n_genes == 0 or self.n_genes is None:
            raise ValueError('self.n_genes can not be zero')
        if self.sort_by not in {'scores', 'log2fc'}:
            raise ValueError('sort_by must be in {\'scores\', \'log2fc\'}')
        if self.method == 'wilcoxon_test':
            self.data.sparse2array()
        if self.groups is None:
            raise ValueError('group information must be set')
        group_info = self.groups
        all_groups = set(group_info['group'].values)
        if self.case_groups == 'all':
            case_groups = all_groups
        else:
            case_groups = self.case_groups
        case_groups = natsorted(case_groups)
        control_str = self.control_groups if isinstance(self.control_groups, str) else '-'.join(self.control_groups)
        # self.result = {}
        # only used when method is wilcoxon
        ranks = None
        tie_term = None
        if self.method == 'wilcoxon_test' and self.control_groups == 'rest':
            self.logger.info('cal rankdata')
            ranks = stats.rankdata(self.data.exp_matrix.T, axis=-1)
            self.logger.info('cal tie_term')
            if self.tie_term:
                tie_term = mannwhitneyu.cal_tie_term(ranks)
            self.logger.info('cal tie_term end')
        self.temp_logres_score = None
        if self.case_groups == 'all' and self.control_groups == 'rest' and self.method == 'logreg':
            self.temp_logres_score = self.logres_score()
        # self.result['pct'], self.result['pct_rest'] = self.calc_pct_and_pct_rest()
        from joblib import Parallel, delayed
        self.len_case_groups = len(case_groups)
        n_jobs = min(cpu_count(), self.n_jobs)
        Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.handle_result)(g, group_info, all_groups, ranks, tie_term, control_str)
            for g in case_groups
        )
        del self.temp_logres_score

    # @log_consumed_time
    # def calc_pct_and_pct_rest(self):
    #     raw_cells_isin_data = np.isin(self.raw_data.cell_names, self.data.cell_names)
    #     raw_genes_isin_data = np.isin(self.raw_data.gene_names, self.data.gene_names)
    #     raw_exp_matrix = self.raw_data.exp_matrix[np.ix_(raw_cells_isin_data, raw_genes_isin_data)]
    #     exp_matrix_one_hot = (raw_exp_matrix > 0).astype(np.uint8)
    #     cluster_result: pd.DataFrame = self.groups.copy()
    #     cluster_result.reset_index(drop=True, inplace=True)
    #     cluster_result.reset_index(inplace=True)
    #     cluster_result.sort_values(by=['group', 'index'], inplace=True)
    #     group_index = cluster_result.groupby('group').agg(cell_index=('index', list))
    #     group_check = group_index.apply(lambda x: 1 if len(x[0]) <= 0 else 0, axis=1, result_type='broadcast')
    #     group_empty_index_list = group_check[group_check['cell_index'] == 1].index.tolist()
    #     group_index.drop(index=group_empty_index_list, inplace=True)

    #     def _calc(a, exp_matrix_one_hot):
    #         cell_index = a[0]
    #         if isinstance_ndarray:
    #             sub_exp = exp_matrix_one_hot[cell_index].sum(axis=0)
    #             sub_exp_rest = exp_matrix_one_hot_number - sub_exp
    #         else:
    #             sub_exp = exp_matrix_one_hot[cell_index].sum(axis=0).A[0]
    #             sub_exp_rest = exp_matrix_one_hot_number - sub_exp
    #         sub_pct = sub_exp / len(cell_index)
    #         sub_pct_rest = sub_exp_rest / (cell_names_size - len(cell_index))
    #         return sub_pct, sub_pct_rest

    #     cell_names_size = self.data.cell_names.size
    #     exp_matrix_one_hot_number = exp_matrix_one_hot.sum(axis=0)
    #     isinstance_ndarray = isinstance(exp_matrix_one_hot, np.ndarray)
    #     if not isinstance_ndarray:
    #         exp_matrix_one_hot_number = exp_matrix_one_hot_number.A[0]
    #     pct_all = np.apply_along_axis(_calc, 1, group_index.values, exp_matrix_one_hot)
    #     pct = pd.DataFrame(pct_all[:, 0], columns=self.data.gene_names, index=group_index.index).T
    #     pct_rest = pd.DataFrame(pct_all[:, 1], columns=self.data.gene_names, index=group_index.index).T
    #     pct.columns.name = None
    #     pct.reset_index(inplace=True)
    #     pct.rename(columns={'index': 'genes'}, inplace=True)
    #     pct_rest.columns.name = None
    #     pct_rest.reset_index(inplace=True)
    #     pct_rest.rename(columns={'index': 'genes'}, inplace=True)
    #     return pct, pct_rest

    def logres_score(self):
        from ..algorithm.statistics import logreg
        x = self.data.exp_matrix
        y = self.groups['group'].values
        if self.case_groups != 'all':
            use_groups = self.case_groups
            if self.control_groups != 'rest':
                use_groups = set(self.case_groups + self.control_groups)
            group_index = self.groups['group'].isin(use_groups)
            x = x[group_index, :]
            y = y[group_index]
        score_df = logreg(x, y)
        score_df.columns = self.data.gene_names
        return score_df

    def run_logres(self, score_df, g_data, other_data, group_name):
        from ..algorithm.statistics import cal_log2fc
        res = pd.DataFrame()
        gene_index = score_df.columns.isin(self.data.gene_names)
        scores = score_df.loc[str(group_name)].values if score_df.shape[0] > 1 else score_df.values[0]
        res['scores'] = scores[gene_index]
        res['log2fc'] = cal_log2fc(g_data, other_data)
        return res

    @staticmethod
    def merge_groups_data(g1, g2):
        """
        drop duplicated and the columns that all the values are 0
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
                         ncols: int = 4):
        from ..plots.marker_genes import marker_genes_text

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
        from ..plots.marker_genes import marker_genes_heatmap

        marker_genes_heatmap(
            data=self.data,
            cluster_res=self.groups,
            marker_res=self.result,
            markers_num=markers_num,
            sort_key=sort_key,
            ascend=ascend,
            show_labels=show_labels,
            show_group=show_group,
            show_group_txt=show_group_txt,
            cluster_colors_array=cluster_colors_array,
            min_value=min_value,
            max_value=max_value,
            gene_list=gene_list,
            do_log=do_log
        )
