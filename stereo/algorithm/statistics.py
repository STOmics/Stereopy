#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
"""


import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


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
    # result = group.apply(lambda x: pd.Series(stats.mannwhitneyu(x, other_group[x.name])), axis=0).transpose()
    g_num = group.shape[0]
    x_array = np.hstack((group.values.T, other_group.values.T))
    result = np.apply_along_axis(lambda x: stats.mannwhitneyu(x[0: g_num], x[g_num:]), 1, x_array)
    result = pd.DataFrame(result, columns=['scores', 'pvalues'])
    result['genes'] = list(group.columns)
    n_genes = result.shape[0]
    pvals_adj = corr_pvalues(result['pvalues'], corr_method, n_genes)
    if pvals_adj is not None:
        result['pvalues_adj'] = pvals_adj
    result['log2fc'] = cal_log2fc(group, other_group)
    return pd.DataFrame(result)


def cal_log2fc(group, other_group):
    g_mean = np.mean(group.values, axis=0)
    other_mean = np.mean(other_group.values, axis=0)
    log2fc = np.log2(g_mean/other_mean + 10e-5)
    return log2fc


def logreg(x, y, **kwds):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(**kwds)
    clf.fit(x, y)
    scores_all = clf.coef_
    group = [str(i) for i in clf.classes_] if len(clf.classes_) > 2 else None
    res = pd.DataFrame(scores_all, index=group)
    return res


# def t_test_overestim_var():
#     if method == 't-test':
#         ns_rest = ns_other
#     elif method == 't-test_overestim_var':
#         # hack for overestimating the variance for small groups
#         ns_rest = ns_group
#     else:
#         raise ValueError('Method does not exist.')
#
#     # TODO: Come up with better solution. Mask unexpressed genes?
#     # See https://github.com/scipy/scipy/issues/10269
#     with np.errstate(invalid="ignore"):
#         scores, pvals = stats.ttest_ind_from_stats(
#             mean1=mean_group,
#             std1=np.sqrt(var_group),
#             nobs1=ns_group,
#             mean2=mean_rest,
#             std2=np.sqrt(var_rest),
#             nobs2=ns_rest,
#             equal_var=False,  # Welch's
#         )
