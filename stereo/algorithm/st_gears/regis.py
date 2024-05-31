import numpy as np
import ot
import scipy
import gc

from typing import Tuple, Optional
from anndata import AnnData

from stereo.log_manager import logger

from .helper import determine_using_gpu, filter_rows_cols, generate_fea_simi_mtx,  filter_pi_mtx, generate_stru_mtx
from .weight import cal_weight
from .optim_pot import cg

# def my_fused_gromov_wasserstein(M, C1, C2, p, q, G_init, loss_fun='square_loss', alpha=0.5, armijo=False,
#                                 log=False, numItermax=200, verbose=False, **kwargs):
#     """
#     准备超参数：初始化G，梯度函数，目标（损失）函数；传入到求解器里求解
#     Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping).
#     Also added capability of utilizing different POT backends to speed up computation.
#
#     For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
#     """
#     p, q = ot.utils.list_to_array(p, q)  # 权重矩阵
#
#     # p0, q0, C10, C20, M0 = p, q, C1, C2, M
#
#     constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
#
#     def f(G):
#         return ot.gromov.gwloss(constC, hC1, hC2, G)
#
#     def df(G):
#         return ot.gromov.gwggrad(constC, hC1, hC2, G)
#
#     if log:
#         # 对ot.gromov.cg不合适的部分做修改，放在cg
#         # res, log = ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G_init, numItermax=numItermax, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)
#
#         # 质量守恒的ot
#         res, log = cg(p, q, (1 - alpha) * M, alpha, f, df, G_init, numItermax=numItermax, armijo=armijo,
#                       C1=C1, C2=C2, constC=constC, verbose=verbose, log=True, **kwargs)
#
#         fgw_dist = log['loss'][-1]
#         log['fgw_dist'] = fgw_dist
#
#         log['u'] = log['u']
#         log['v'] = log['v']
#         return res, log
#
#     else:
#         # res = ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G_init, numItermax=numItermax, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
#
#         res = cg(p, q, (1 - alpha) * M, alpha, f, df, G_init, numItermax=numItermax, armijo=armijo,
#                  C1=C1, C2=C2, constC=constC, verbose=verbose, **kwargs)
#
#         return res


def cal_fused_gromov_wasserstein_param(
        sliceA: AnnData,
        sliceB: AnnData,
        label_col: str,
        dissimilarity_val: str,
        dissimilarity_weight_val: str,
        uniform_weight: bool,
        map_method_dis2wei: str,
        filter_by_label: bool = True,
        use_gpu: bool = False,
        norm: bool = False)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    calculate parameters of Fused-Gromov-Wasserstein problem
    """

    sliceA, sliceB = filter_rows_cols(sliceA, sliceB, filter_by_label, label_col)
    # print('sliceA shape', sliceA.n_obs, sliceA.n_vars)
    # print('sliceB shape', sliceB.n_obs, sliceB.n_vars)

    if use_gpu:
        nx = ot.backend.TorchBackend()
    else:
        nx = ot.backend.NumpyBackend()

    a, b, _, _ = cal_weight(uniform_weight, nx, sliceA, sliceB, label_col, map_method_dis2wei,
                            dissimilarity_weight_val)

    D_A, D_B = generate_stru_mtx(nx, sliceA, sliceB, norm)

    M = generate_fea_simi_mtx(nx, sliceA, sliceB, dissimilarity_val)

    G_init = a[:, None] * b[None, :]

    if use_gpu:
        G_init = G_init.cuda()

    if use_gpu:
        import torch

        M = M.type(torch.float64)
        D_A = D_A.type(torch.float64)
        D_B = D_B.type(torch.float64)
        a = a.type(torch.float64)
        b = b.type(torch.float64)
        G_init = G_init.type(torch.float64)
    else:
        M = M.astype(np.float64)
        D_A = D_A.astype(np.float64)
        D_B = D_B.astype(np.float64)
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        G_init = G_init.astype(np.float64)

    a = a / nx.sum(a)
    b = b / nx.sum(b)

    return M, D_A, D_B, a, b, G_init


def pairwise_align(
        M: np.ndarray,
        D_A: np.ndarray,
        D_B: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        alpha: float,
        numItermax: int,
        G_init=None,
        use_gpu: bool = False,
        verbose: bool = False,
        **kwargs) -> Tuple[np.ndarray, Optional[int]]:
    """
    Calculates and returns optimal alignment of two slices, using Conditional Gradient Descent optimization, towards
    Fused Gromov Wasserstein OT problem.
    """

    if use_gpu:
        nx = ot.backend.TorchBackend()
    else:
        nx = ot.backend.NumpyBackend()

    # 1. calculate parameters
    p, q = ot.utils.list_to_array(a, b)  # 权重矩阵

    constC, hC1, hC2 = ot.gromov.init_matrix(D_A, D_B, p, q, 'square_loss')

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1 / nx.sum(G_init)) * G_init

    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs)

    # 2. Conditional Gradient Descent optimization, to solve Fused Gromov Wasserstein OT problem

    # res, log = ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G_init, numItermax=numItermax, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)
    # res, log = cg(p, q, (1 - alpha) * M, alpha, f, df, G_init, numItermax=numItermax, armijo=False,
    #               C1=D_A, C2=D_B, constC=constC, verbose=verbose, log=True, **kwargs)

    tol_rel = 1e-9
    tol_abs = 1e-9
    res, log = ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, line_search, log=True, numItermax=numItermax,
                           stopThr=tol_rel, stopThr2=tol_abs, verbose=verbose, **kwargs)

    fgw_dist = log['loss'][-1]
    log['fgw_dist'] = fgw_dist

    # 3. format the result to numpy
    pi = nx.to_numpy(res)
    obj = nx.to_numpy(log['fgw_dist'])

    return pi, obj


def calculate_ctype_score(pi, regis_f, regis_m, anncell_cid, filter_by_label, label_col):
    regis_f, regis_m = filter_rows_cols(regis_f, regis_m, filter_by_label, label_col)

    type_f_li = [anncell_cid[regis_f.obs[label_col].iloc[i]] for i in range(pi.shape[0])]
    type_m_li = [anncell_cid[regis_m.obs[label_col].iloc[i]] for i in range(pi.shape[1])]

    type_f_arr = np.broadcast_to(np.array(type_f_li)[..., np.newaxis], pi.shape)
    type_m_arr = np.broadcast_to(np.array(type_m_li), pi.shape)

    mask = (type_f_arr == type_m_arr)
    ctype_score = pi[mask].sum()
    return ctype_score


def serial_align(slicesl,
                 anncell_cid,
                 label_col,
                 start_i,
                 end_i,
                 tune_alpha_li=[0.2, 0.1, 0.05, 0.025, 0.01, 0.005],
                 numItermax=200,
                 dissimilarity_val='kl',
                 uniform_weight=False,
                 dissimilarity_weight_val='kl',
                 map_method_dis2wei='logistic',
                 filter_by_label=True,
                 use_gpu=False,
                 verbose=False):
    """
    Compute numerical optimization for a serial of slices (AnnData), and return a serial of probabilistic transition matrices.
    In the end of computation, probabilities ranked in the last 10% in transition matrix were filtered.

    input:
    slicesl: [slice0, slice1, slice2, ...], every element is AnnData. adata.X stores normalized expression matrix, with rows
            indicate cells or binsets, while columnw indicate genes.
    anncell_cid: Dictionary that maps annotated cell types to id starting from 0.
    label_col: String of column name in .obs, where annotated cell types are stored.
    start_i: Index in slicesl of the first slice to be registered. Slices ranked before it will not be registered, or
            visualized the the following protocol.
    end_i: Index in slicesl of the last slice to be registered. Slices ranked after it will not be registered, or
            visualized the the following protocol.
    tune_alpha_li: List of regularization factor in Fused Gromov Wasserstin (FGW) OT problem formulation, to be
                   automatically tunned. Refer to this paper for the FGW formulation:
                   Optimal transport for structured data with application on graphs. T Vayer, L Chapel, R Flamary,
                   R Tavenard… - arXiv preprint arXiv …, 2018 - arxiv.org
    numItermax: Max number of iterations.
    dissimilarity_val: Matrix to calculate feature similarity. Choose between 'kl' for Kullback-Leibler Divergence,
                       and 'euc'/'euclidean' for euclidean distance.
    uniform_weight: Whether to assign same margin weights to every spots. Choose between True and False.
    dissimilarity_weight_val: Matrix to calculate cell types feature similarity when assigning weighted boundary conditions
                             for margin constrains. Refer to our paper for more details. Only assign when uniform_weight is False.
    map_method_dis2wei: Methood to map cell types feature similarity to margin weighhts. Choose between linear' and 'logistic'.
                        Only assign when uniform_weight is False.
    filter_by_label: Where to filter out spots not appearing in its registered slice, so it won't interfere with the ot
                     solving process.
    use_gpu: Whether to use GPU or not, in the parameter calculation process. OT solving process is only built on CPU.
    verbose: To print the OT solving process of each iteration.

    output:
    pili: List of probabilistic transition matrix solved. The order is the same as slices in slicesl. Number of element of
          pili is one less than slicesl.
    tyscoreli: List of LTARI score of each transition matrix. The order is the same as slices in slicesl. Number of
               element of pili is one less than slicesl.
    alphali: List of alpha value that was chosen and used for each pair of slices, among tune_alpha_li provided by users.
    regis_ilist: List of index of registered slices.
    ali: List of margin weights of the first slice of slice pairs. Each pair corresponds to an element of the list.
    bli: List of margin weights of the second slice of slice pairs. Each pair corresponds to an element of the list.
    """

    if not type(slicesl) == list:
        try:
            slicesl = list(slicesl)
            assert len(slicesl) >= 2, "Input slices less than 2"
        except:
            raise TypeError('Input list for slicesl')

    assert label_col in slicesl[0].obs.columns, "label_col not in adata.obs.columns"

    assert type(start_i) == int, "Type of start_i is not integer."
    assert start_i >= 0, "start_i less than 0"

    assert type(end_i) == int, "Type of end_i is not integer"
    assert end_i - start_i >= 1, "end_i is not at least 1 larger than start_i"

    assert type(tune_alpha_li) in [list, float], "Input tune_alpha_li is not a list or a float"
    if type(tune_alpha_li) == float:
        tune_alpha_li = [tune_alpha_li]
    tune_alpha_li = list(set([float(alpha) for alpha in tune_alpha_li]))

    assert dissimilarity_val in ['kl', 'euc', 'euclidean'], "dissimilarity_val not in ['kl', 'euc', 'euclidean']"

    assert type(uniform_weight) == bool, "Type of uniform_weight is not bool"
    if uniform_weight is False:
        assert dissimilarity_weight_val in ['kl', 'euc', 'euclidean'], "dissimilarity_weight_val not in ['kl', 'euc', 'euclidean']"
        assert map_method_dis2wei in ['linear', 'logistic'], "map_method_dis2wei not in ['linear', 'logistic']"

    assert type(filter_by_label) == bool, "Type of filter_by_label is not bool"
    assert type(use_gpu) == bool, "Type of use_gpu is not bool"
    assert type(verbose) == bool, "Type of verbose is not bool"

    # determine using gpu
    use_gpu = determine_using_gpu(use_gpu)
    if use_gpu:
        import torch

    pili = []
    tyscoreli = []
    alphali = []
    regis_ilist = []
    ali = []
    bli = []

    i = start_i
    while i + 1 <= end_i:

        slice_f = slicesl[i]
        slice_m = slicesl[i + 1]

        M, D_A, D_B, a, b, G_init = cal_fused_gromov_wasserstein_param(slice_f, slice_m, label_col,
                                                                       dissimilarity_val=dissimilarity_val, dissimilarity_weight_val=dissimilarity_weight_val,
                                                                       uniform_weight=uniform_weight, map_method_dis2wei=map_method_dis2wei,
                                                                       filter_by_label=filter_by_label, use_gpu=use_gpu)
        # print('range of a,', a.min(), a.max())
        # print('range of b,', b.min(), b.max())

        for j, alpha_val in enumerate(tune_alpha_li):
            logger.info(f"slice: {i}, tune_alpha[{j}]: {alpha_val}")

            # should not normalize baseed on raw M, D_A and D_B since the solver will use the data in different dimension

            pi, _ = pairwise_align(M, D_A, D_B, a, b, alpha=alpha_val, numItermax=numItermax, G_init=G_init, use_gpu=use_gpu, verbose=verbose)

            if use_gpu:
                torch.cuda.empty_cache()

            pi = filter_pi_mtx(pi, percentage=10, same_mem=True)
            type_score = calculate_ctype_score(pi, slice_f, slice_m, anncell_cid, filter_by_label=filter_by_label, label_col=label_col)

            logger.info(f"type score: {type_score}")

            if j == 0:
                type_score_best = type_score
                alpha_best = alpha_val

                pi_best = scipy.sparse.csr_matrix(pi)

                del pi
                gc.collect()

            elif type_score > type_score_best:
                pi_best = scipy.sparse.csr_matrix(pi)

                del pi
                gc.collect()

                type_score_best = type_score
                alpha_best = alpha_val

            else:
                del pi
                gc.collect()
        try:
            del M
        except:
            pass
        try:
            del A_X
        except:
            pass
        try:
            del B_X
        except:
            pass

        del D_A
        del D_B
        del G_init
        gc.collect()

        if use_gpu:
            torch.cuda.empty_cache()

        # a, b to cpu to numpy
        if use_gpu:
            nx = ot.backend.TorchBackend()
            a = nx.to_numpy(a.cpu())
            b = nx.to_numpy(b.cpu())
            gc.collect()

        ali.append(a)
        bli.append(b)

        del a
        del b
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        pili.append(pi_best)
        logger.info(f"type_score_best: {type_score_best}")

        tyscoreli.append(type_score_best)

        alphali.append(alpha_best)
        logger.info(f"alpha_best: {alpha_best}")

        regis_ilist.append(i)

        i += 1

    regis_ilist.append(i)

    if use_gpu:
        torch.cuda.empty_cache()

    return pili, tyscoreli, alphali, regis_ilist, ali, bli
