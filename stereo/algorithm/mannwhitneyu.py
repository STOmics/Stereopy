#!/usr/bin/env python3
# coding: utf-8
"""
@file: mannwhitneyu.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/11/05  create file.
"""
import numpy as np
from dataclasses import make_dataclass
from collections import namedtuple
from scipy import special
from scipy import stats


def _broadcast_concatenate(x, y, axis):
    '''Broadcast then concatenate arrays, leaving concatenation axis last'''
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    z = np.broadcast(x[..., 0], y[..., 0])
    x = np.broadcast_to(x, z.shape + (x.shape[-1],))
    y = np.broadcast_to(y, z.shape + (y.shape[-1],))
    z = np.concatenate((x, y), axis=-1)
    return x, y, z


class _MWU:
    '''Distribution of MWU statistic under the null hypothesis'''
    # Possible improvement: if m and n are small enough, use integer arithmetic

    def __init__(self):
        '''Minimal initializer'''
        self._fmnks = -np.ones((1, 1, 1))

    def pmf(self, k, m, n):
        '''Probability mass function'''
        self._resize_fmnks(m, n, np.max(k))
        # could loop over just the unique elements, but probably not worth
        # the time to find them
        for i in np.ravel(k):
            self._f(m, n, i)
        return self._fmnks[m, n, k] / special.binom(m + n, m)

    def cdf(self, k, m, n):
        '''Cumulative distribution function'''
        # We could use the fact that the distribution is symmetric to avoid
        # summing more than m*n/2 terms, but it might not be worth the
        # overhead. Let's leave that to an improvement.
        pmfs = self.pmf(np.arange(0, np.max(k) + 1), m, n)
        cdfs = np.cumsum(pmfs)
        return cdfs[k]

    def sf(self, k, m, n):
        '''Survival function'''
        # Use the fact that the distribution is symmetric; i.e.
        # _f(m, n, m*n-k) = _f(m, n, k), and sum from the left
        k = m*n - k
        # Note that both CDF and SF include the PMF at k. The p-value is
        # calculated from the SF and should include the mass at k, so this
        # is desirable
        return self.cdf(k, m, n)

    def _resize_fmnks(self, m, n, k):
        '''If necessary, expand the array that remembers PMF values'''
        # could probably use `np.pad` but I'm not sure it would save code
        shape_old = np.array(self._fmnks.shape)
        shape_new = np.array((m+1, n+1, k+1))
        if np.any(shape_new > shape_old):
            shape = np.maximum(shape_old, shape_new)
            fmnks = -np.ones(shape)             # create the new array
            m0, n0, k0 = shape_old
            fmnks[:m0, :n0, :k0] = self._fmnks  # copy remembered values
            self._fmnks = fmnks

    def _f(self, m, n, k):
        '''Recursive implementation of function of [3] Theorem 2.5'''

        # [3] Theorem 2.5 Line 1
        if k < 0 or m < 0 or n < 0 or k > m*n:
            return 0

        # if already calculated, return the value
        if self._fmnks[m, n, k] >= 0:
            return self._fmnks[m, n, k]

        if k == 0 and m >= 0 and n >= 0:  # [3] Theorem 2.5 Line 2
            fmnk = 1
        else:   # [3] Theorem 2.5 Line 3 / Equation 3
            fmnk = self._f(m-1, n, k-n) + self._f(m, n-1, k)

        self._fmnks[m, n, k] = fmnk  # remember result

        return fmnk


# Maintain state for faster repeat calls to mannwhitneyu w/ method='exact'
_mwu_state = _MWU()


def _tie_term(ranks):
    """Tie correction term"""
    # element i of t is the number of elements sharing rank i
    _, t = np.unique(ranks, return_counts=True, axis=-1)
    return (t**3 - t).sum(axis=-1)


def cal_tie_term(ranks):
    tie_term = np.apply_along_axis(_tie_term, -1, ranks)
    return tie_term


def _get_mwu_z(U, n1, n2, tie_term, continuity=True):
    '''Standardized MWU statistic'''
    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2
    # Tie correction according to [2]
    # tie_term = np.apply_along_axis(_tie_term, -1, ranks)
    if tie_term is None:
        s = np.sqrt(n1*n2*(n+1)/12)
    else:
        s = np.sqrt(n1*n2/12 * ((n + 1) - tie_term/(n*(n-1))))
    numerator = U - mu
    if continuity:
        numerator -= 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z


def _mwu_input_validation(x, y, use_continuity, alternative, axis, method):
    ''' Input validation and standardization for mannwhitneyu '''
    # Would use np.asarray_chkfinite, but infs are OK
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')

    bools = {True, False}
    if use_continuity not in bools:
        raise ValueError(f'`use_continuity` must be one of {bools}.')

    alternatives = {"two-sided", "less", "greater"}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')

    methods = {"asymptotic", "exact", "auto"}
    method = method.lower()
    if method not in methods:
        raise ValueError(f'`method` must be one of {methods}.')

    return x, y, use_continuity, alternative, axis_int, method


def _tie_check(xy):
    """Find any ties in data"""
    _, t = np.unique(xy, return_counts=True, axis=-1)
    return np.any(t != 1)


def _mwu_choose_method(n1, n2, xy, method):
    """Choose method 'asymptotic' or 'exact' depending on input size, ties"""

    # if both inputs are large, asymptotic is OK
    if n1 > 8 and n2 > 8:
        return "asymptotic"

    # if there are any ties, asymptotic is preferred
    if np.apply_along_axis(_tie_check, -1, xy).any():
        return "asymptotic"

    return "exact"


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))


def mannwhitneyu(x, y, use_continuity=True, alternative="two-sided",
                 axis=0, method="auto", ranks=None, tie_term=None, x_mask=None):
    r'''Perform the Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of
    of difference in location between distributions.

    Parameters
    ----------
    x, y : array-like
        N-d arrays of samples. The arrays must be broadcastable except along
        the dimension given by `axis`.
    use_continuity : bool, optional
            Whether a continuity correction (1/2) should be applied.
            Default is True when `method` is ``'asymptotic'``; has no effect
            otherwise.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        Let *F(u)* and *G(u)* be the cumulative distribution functions of the
        distributions underlying `x` and `y`, respectively. Then the following
        alternative hypotheses are available:

        * 'two-sided': the distributions are not equal, i.e. *F(u) ≠ G(u)* for
          at least one *u*.
        * 'less': the distribution underlying `x` is stochastically less
          than the distribution underlying `y`, i.e. *F(u) > G(u)* for all *u*.
        * 'greater': the distribution underlying `x` is stochastically greater
          than the distribution underlying `y`, i.e. *F(u) < G(u)* for all *u*.

        Under a more restrictive set of assumptions, the alternative hypotheses
        can be expressed in terms of the locations of the distributions;
        see [5] section 5.1.
    axis : int, optional
        Axis along which to perform the test. Default is 0.
    method : {'auto', 'asymptotic', 'exact'}, optional
        Selects the method used to calculate the *p*-value.
        Default is 'auto'. The following options are available.

        * ``'asymptotic'``: compares the standardized test statistic
          against the normal distribution, correcting for ties.
        * ``'exact'``: computes the exact *p*-value by comparing the observed
          :math:`U` statistic against the exact distribution of the :math:`U`
          statistic under the null hypothesis. No correction is made for ties.
        * ``'auto'``: chooses ``'exact'`` when the size of one of the samples
          is less than 8 and there are no ties; chooses ``'asymptotic'``
          otherwise.

    Returns
    -------
    res : MannwhitneyuResult
        An object containing attributes:

        statistic : float
            The Mann-Whitney U statistic corresponding with sample `x`. See
            Notes for the test statistic corresponding with sample `y`.
        pvalue : float
            The associated *p*-value for the chosen `alternative`.
    '''

    x, y, use_continuity, alternative, axis_int, method = (
        _mwu_input_validation(x, y, use_continuity, alternative, axis, method))

    x, y, xy = _broadcast_concatenate(x, y, axis)

    n1, n2 = x.shape[-1], y.shape[-1]

    if method == "auto":
        method = _mwu_choose_method(n1, n2, xy, method)

    # Follows [2]
    if ranks is None:
        ranks = stats.rankdata(xy, axis=-1)  # method 2, step

    R1 = ranks[..., :n1].sum(axis=-1) if x_mask is None else ranks[..., x_mask].sum(axis=-1)   # method 2, step 2
    U1 = R1 - n1*(n1+1)/2                # method 2, step 3
    U2 = n1 * n2 - U1                    # as U1 + U2 = n1 * n2

    if alternative == "greater":
        U, f = U1, 1  # U is the statistic to use for p-value, f is a factor
    elif alternative == "less":
        U, f = U2, 1  # Due to symmetry, use SF of U2 rather than CDF of U1
    else:
        U, f = np.maximum(U1, U2), 2  # multiply SF by two for two-sided test

    if method == "exact":
        p = _mwu_state.sf(U.astype(int), n1, n2)
    elif method == "asymptotic":
        # if tie_term is None:
        #     tie_term = cal_tie_term(ranks)
        z = _get_mwu_z(U, n1, n2, tie_term, continuity=use_continuity)
        p = stats.norm.sf(z)
    p *= f
    p = np.clip(p, 0, 1)

    return MannwhitneyuResult(z, p)
