#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
"""


import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from .mannwhitneyu import mannwhitneyu


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


def cal_log2fc(group, other_group):
    g_mean = np.mean(group, axis=0) + 1e-9
    other_mean = np.mean(other_group, axis=0) + 1e-9
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


def wilcoxon(group, other_group, corr_method=None, ranks=None, tie_term=None, x_mask=None):
    """
    wilcoxon_test

    :param group:
    :param other_group:
    :param corr_method:
    :param ranks:
    :param tie_term:
    :param x_mask:
    :return:
    """
    s, p = mannwhitneyu(group, other_group, ranks=ranks, tie_term=tie_term, x_mask=x_mask)
    result = pd.DataFrame({'scores': s, 'pvalues': p})
    # result['genes'] = list(group.columns)
    n_genes = result.shape[0]
    pvals_adj = corr_pvalues(result['pvalues'], corr_method, n_genes)
    if pvals_adj is not None:
        result['pvalues_adj'] = pvals_adj
    result['log2fc'] = cal_log2fc(group, other_group)
    return pd.DataFrame(result)


def ttest(group, other_group, corr_method=None):
    mean_group, var_group = get_mean_var(group)
    mean_rest, var_rest = get_mean_var(other_group)
    with np.errstate(invalid="ignore"):
        scores, pvals = stats.ttest_ind_from_stats(
            mean1=mean_group,
            std1=np.sqrt(var_group),
            nobs1=group.shape[0],
            mean2=mean_rest,
            std2=np.sqrt(var_rest),
            nobs2=other_group.shape[0],
            equal_var=False,  # Welch's
        )
    scores[np.isnan(scores)] = 0
    pvals[np.isnan(pvals)] = 1
    n_genes = group.shape[1]
    pvals_adj = corr_pvalues(pvals, corr_method, n_genes)
    result = {'scores': scores, 'pvalues': pvals}
    if pvals_adj is not None:
        result['pvalues_adj'] = pvals_adj
    result['log2fc'] = cal_log2fc(group, other_group)
    return pd.DataFrame(result)


def get_mean_var(x, *, axis=0):
    mean = np.mean(x, axis=axis, dtype=np.float64)
    mean_sq = np.multiply(x, x).mean(axis=axis, dtype=np.float64)
    var = mean_sq - mean ** 2
    # enforce R convention (unbiased estimator) for variance
    var *= x.shape[axis] / (x.shape[axis] - 1)
    return mean, var