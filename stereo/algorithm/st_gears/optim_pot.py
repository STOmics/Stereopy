"""
an optimizer module revised from POT, to suit the project's need
"""

import numpy as np
import warnings

from ot.lp import emd
from ot.utils import list_to_array
from ot.backend import get_backend

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from scipy.optimize import scalar_search_armijo
    except ImportError:
        from scipy.optimize.linesearch import scalar_search_armijo

from stereo.log_manager import logger


def line_search_armijo(
    f, xk, pk, gfk, old_fval, args=(), c1=1e-4,
    alpha0=0.99, alpha_min=None, alpha_max=None
):
    r"""
    Armijo linesearch function that works with matrices

    Find an approximate minimum of :math:`f(x_k + \alpha \cdot p_k)` that satisfies the
    armijo conditions.

    Parameters
    ----------
    f : callable
        loss function
    xk : array-like
        initial position
    pk : array-like
        descent direction
    gfk : array-like
        gradient of `f` at :math:`x_k`
    old_fval : float
        loss value at :math:`x_k`
    args : tuple, optional
        arguments given to `f`
    c1 : float, optional
        :math:`c_1` const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
    alpha_min : float, optional
        minimum value for alpha
    alpha_max : float, optional
        maximum value for alpha

    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float
        loss value at step alpha

    """

    xk, pk, gfk = list_to_array(xk, pk, gfk)
    nx = get_backend(xk, pk)

    if len(xk.shape) == 0:
        xk = nx.reshape(xk, (-1,))

    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval

    derphi0 = nx.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0)

    if alpha is None:
        return 0., fc[0], phi0
    else:
        if alpha_min is not None or alpha_max is not None:
            alpha = np.clip(alpha, alpha_min, alpha_max)
        return float(alpha), fc[0], phi1


def solve_linesearch(
    cost, G, deltaG, Mi, f_val, armijo=True, C1=None, C2=None,
    reg=None, constC=None, M=None, alpha_min=None, alpha_max=None
):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------
    cost : method
        Cost in the FW for the linesearch
    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : array-like (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val : float
        Value of the cost at `G`
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    C1 : array-like (ns,ns), optional
        Structure matrix in the source domain. Only used and necessary when armijo=False
    C2 : array-like (nt,nt), optional
        Structure matrix in the target domain. Only used and necessary when armijo=False
    reg : float, optional
        Regularization parameter. Only used and necessary when armijo=False
    Gc : array-like (ns,nt)
        Optimal map found by linearization in the FW algorithm. Only used and necessary when armijo=False
    constC : array-like (ns,nt)
        Constant for the gromov cost. See :ref:`[24] <references-solve-linesearch>`. Only used and necessary when armijo=False
    M : array-like (ns,nt), optional
        Cost matrix between the features. Only used and necessary when armijo=False
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha

    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    f_val : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    def torch_np2value(x):
        if type(x).__module__ == np.__name__:  # isinstance(x, np.ndarray):
            return x.item()
        else:
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.cpu().item()
            except:
                logger.warning('need to install torch')
                quit()

    if armijo:
        alpha, fc, f_val = line_search_armijo(
            cost, G, deltaG, Mi, f_val, alpha_min=alpha_min, alpha_max=alpha_max
        )
    else:  # requires symetric matrices
        G, deltaG, C1, C2, constC, M = list_to_array(G, deltaG, C1, C2, constC, M)
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2, constC)
        else:
            nx = get_backend(G, deltaG, C1, C2, constC, M)

        dot = nx.dot(nx.dot(C1, deltaG), C2)
        a = -2 * reg * nx.sum(dot * deltaG)
        b = nx.sum((M + reg * constC) * deltaG) - 2 * reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2) * deltaG))
        c = cost(G)

        a = torch_np2value(a)
        b = torch_np2value(b)
        c = torch_np2value(c)

        # print('a, b, c during line search', a, b, c)
        # print('b sub-terms', nx.sum((M + reg * constC) * deltaG), - 2 * reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2) * deltaG)))

        alpha = solve_1d_linesearch_quad(a, b, c)
        if alpha_min is not None or alpha_max is not None:
            alpha = np.clip(alpha, alpha_min, alpha_max)
        fc = None
        f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val


def cg(a, b, M, reg, f, df, G0=None, numItermax=200, numItermaxEmd=100000,
       stopThr=1e-12, stopThr2=1e-12, verbose=False, log=False, **kwargs):  # old thrval = 1e-9

    r"""
    Solve the general regularized OT problem with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`


    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples in the target domain
    M : array-like, shape (ns, nt)
        loss matrix [(1-alpha)*特征相似度矩阵]
    reg : float
        Regularization term >0
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numItermaxEmd : int, optional
        Max number of iterations for emd
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport

    """
    a, b, M, G0 = list_to_array(a, b, M, G0)
    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M)

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        G = G0

    def cost(G):
        return nx.sum(M * G) + reg * f(G)

    f_val = cost(G)

    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        logger.info('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        logger.info('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg * df(G)  # Eq 7

        # set M positive
        Mi += nx.min(Mi)

        # solve linear program
        Gc, logemd = emd(a, b, Mi, numItermax=numItermaxEmd, log=True, numThreads='max')

        deltaG = Gc - G

        # line search
        alpha, fc, f_val = solve_linesearch(
            cost, G, deltaG, Mi, f_val, reg=reg, M=M,
            alpha_min=0., alpha_max=1., **kwargs
        )

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)

        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                logger.info('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            logger.info('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

    if log:
        log.update(logemd)
        return G, log
    else:
        return G


def solve_1d_linesearch_quad(a, b, c):
    r"""
    For any convex or non-convex 1d quadratic function `f`, solve the following problem:

    .. math::

        \mathop{\arg \min}_{0 \leq x \leq 1} \quad f(x) = ax^{2} + bx + c

    Parameters
    ----------
    a,b,c : float
        The coefficients of the quadratic function

    Returns
    -------
    x : float
        The optimal value which leads to the minimal cost
    """
    f0 = c
    df0 = b
    f1 = a + f0 + df0

    if a > 0:  # convex
        minimum = min(1, max(0, -b / (2.0 * a)))  # min(1, max(0, np.divide(-b, 2.0 * a)))
        return minimum
    else:  # non convex
        if f0 > f1:
            return 1
        else:
            return 0
