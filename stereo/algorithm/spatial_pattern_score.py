#!/usr/bin/env python3
# coding: utf-8
"""
@file: spatial_pattern_score.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/08/27  create file.
"""

import pandas as pd
import numpy as np
import statistics
import scipy.stats as stats
from tqdm import tqdm


def spatial_pattern_score(exp_df: pd.DataFrame):
    """
    calculate the spatial pattern score.
    :param exp_df: dataframe of spatial express matrix which columns is genes, and rows is cells.
    :return:
    """
    tqdm.pandas(desc="calculating enrichment score")
    report = exp_df.progress_apply(get_enrichment_score, axis=0)
    report = report.T.reset_index()
    report.columns = ['gene', 'E10', 'C50', 'total_count']
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
    return report_out


def get_enrichment_score(gene_expression):
    """
    calculate enrichment score E10 and C50.
    :param gene_expression: expression data for the input gene
    :return: list E10 score, C50 score and total MID counts of input gene
    """
    gene_expression = gene_expression[gene_expression > 0]
    gene_expression = gene_expression.sort_values(ascending=False).reset_index(drop=True)
    count_list = gene_expression.values
    total_count = np.sum(count_list)
    e10 = np.around(100 * (np.sum(count_list[:int(len(count_list) * 0.1)]) / total_count), 2)
    cdf = np.cumsum(count_list)
    count_fraction_list = cdf / total_count
    c50 = np.around((next(idx for idx, count_fraction in enumerate(count_fraction_list) if count_fraction > 0.5)
                     / len(count_fraction_list)) * 100, 2)
    return e10, c50, total_count


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
