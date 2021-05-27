#!/usr/bin/env python3
# coding: utf-8
"""
@author: Leying Wang wangleying@genomics.cn
@last modified by: Leying Wang
@file: spatial_pattern_score.py
@time: 2021/4/19 17:20
"""

from anndata import AnnData
import pandas as pd
import numpy as np
import statistics
import scipy.stats as stats
from ..core.tool_base import ToolBase
from ..core.stereo_result import SpatialPatternScoreResult


class SpatialPatternScore(ToolBase):
    """
    calculate spatial pattern score
    """
    def __init__(self, data: AnnData, method='enrichment',
                 name='spatial_pattern_score'):
        self.params = self.get_params(locals())
        super(SpatialPatternScore, self).__init__(data=data, method=method, name=name)
        self.check_param()
        self.result = SpatialPatternScoreResult(name=name, param=self.params)

    def check_param(self):
        """
        Check whether the parameters meet the requirements.
        """
        super(SpatialPatternScore, self).check_param()

    def fit(self):
        """
        run
        """
        report = []
        for gene in self.data.var.index:
            gene_expression = pd.DataFrame(self.data[:, gene].X, columns=['values'],
                                           index=list(self.data.obs.index))
            gene_expression = gene_expression[gene_expression['values'] > 0]
            report.append(get_enrichment_score(gene, gene_expression))
        report = pd.DataFrame(report, columns=['gene', 'E10', 'C50', 'total_count'])
        tmp = report[report['total_count'] > 300]
        e10_cutoff = find_cutoff(list(tmp['E10']), 0.9)
        c50_cutoff = find_cutoff(list(tmp['C50']), 0.1)
        pattern = tmp[(tmp['E10'] > e10_cutoff) & (tmp.C50 < c50_cutoff)]
        no_pattern = tmp.drop(pattern.index, inplace=False)
        low_exp = report.drop(tmp.index, inplace=False)
        report_out = pd.concat([pattern, no_pattern, low_exp])[["gene", "E10"]]
        report_out["attribute"] = \
            pattern.shape[0] * ["pattern"] \
            + no_pattern.shape[0] * ["no_pattern"] \
            + low_exp.shape[0] * ["low_exp"]
        report_out.index = report_out['gene']
        report_out = report_out.reindex(self.data.var.index)
        result = SpatialPatternScoreResult(name=self.name, param=self.params, pattern_info=report_out)
        self.add_result(result, key_added=self.name)
        # TODO  added for spatial pattern score
        self.data.var['E10'] = report_out['E10']
        self.data.var['pattern_attribute'] = report_out['attribute']


def get_enrichment_score(gene, gene_expression):
    """
    calculate enrichment score E10 and C50.

    :param gene: gene name
    :param gene_expression: expression data for the input gene
    :return: list of gene name, E10 score, C50 score and total MID counts of input gene
    """
    gene_expression = gene_expression.sort_values(by='values', ascending=False).reset_index(drop=True)
    count_list = list(gene_expression['values'])
    total_count = np.sum(count_list)
    e10 = np.around(100 * (np.sum(count_list[:int(len(count_list) * 0.1)]) / total_count), 2)
    cdf = np.cumsum(count_list)
    count_fraction_list = cdf / total_count
    c50 = np.around((next(idx for idx, count_fraction in enumerate(count_fraction_list) if count_fraction > 0.5)
                     / len(count_fraction_list)) * 100, 2)
    return gene, e10, c50, total_count


def find_cutoff(score_list, p):
    """
    find cutoff

    :param score_list:
    :param p: expression data for gene
    :return: the cutoff of E10 or C50
    """
    curve = score_list
    mu = np.mean(curve)
    sd = statistics.stdev(curve)
    cutoff = stats.norm.ppf(p) * sd + mu
    return cutoff
