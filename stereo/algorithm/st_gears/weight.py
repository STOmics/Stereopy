"""module that calculates weighted margins for OT"""

import math
import numpy as np
import ot
import gc

from .helper import extract_data_matrix, kl_divergence_backend, gene_selection_slice, to_dense_array


def dis2weight(dis_x, map_method_dis2wei):
    """
    transform dissimilarity to weights
    """
    def ad_logistic_func(dis_x):
        """
        adjusted logistic function
        output: [0, 0.5)
        """
        cal_li = dis_x.tolist()
        re_li = [1 / (1 + math.exp(-1 * ele)) - 0.5 for ele in cal_li]
        return np.array(re_li)

    if map_method_dis2wei == 'logistic':
        dis_x = ad_logistic_func(dis_x)

    wei_x = -1 * dis_x  # 距离越小，相对来说权重越大

    wei_x_abs = wei_x - np.min(wei_x)  # 保证权重为正数: 最小值置为0
    wei_x = wei_x_abs / np.sum(wei_x_abs)  # 保证其和为1
    return wei_x


def cal_weight_un_uniform_per_group(sliceA, sliceB, label_col, dissimilarity, map_method_dis2wei):
    """
    calculate un-unform weight for margins based on cell-type similarity across slices
    """

    def find_dis_of_the_type(sliceA, sliceB, label_colname, ctype, dissimilarity):

        def logarithm_norm(arr_1d, min_val=0, max_val=10000):
            arr_1d = (arr_1d - arr_1d.min()) / (arr_1d.max() - arr_1d.min()) * (max_val - min_val) + min_val
            return np.log10(arr_1d + 1)

        anno_a_se = sliceA.obs[label_colname]
        anno_b_se = sliceB.obs[label_colname]

        M_A = to_dense_array(extract_data_matrix(sliceA, rep=None))
        M_B = to_dense_array(extract_data_matrix(sliceB, rep=None))

        # M_A = sc.tl.pca(M_A, n_comps=200, svd_solver='arpack')
        # M_B = sc.tl.pca(M_B, n_comps=200, svd_solver='arpack')

        ind_li = [int(i) for i, x in enumerate(anno_a_se.tolist()) if x == ctype]

        M_A_ctype_cell = M_A[np.array(ind_li), :]  # expression mtx of certain cell type by cell
        M_A_ctype = np.sum(M_A_ctype_cell, axis=0)  # expression mtx of certain cell type, 1d_array

        ind_li = [i for i, x in enumerate(anno_b_se.tolist()) if x == ctype]
        M_B_ctype_cell = M_B[np.array(ind_li), :]
        M_B_ctype = np.sum(M_B_ctype_cell, axis=0)  # expression mtx of certain cell type, 1d_array

        M_A_ctype = logarithm_norm(M_A_ctype)
        M_B_ctype = logarithm_norm(M_B_ctype)

        if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
            dis = np.linalg.norm(M_A_ctype - M_B_ctype)
        elif dissimilarity in ['kl', 'js']:
            M_A_ctype = np.expand_dims(M_A_ctype, axis=0) + 0.01
            M_B_ctype = np.expand_dims(M_B_ctype, axis=0) + 0.01

            if dissimilarity == 'kl':
                M = kl_divergence_backend(M_A_ctype, M_B_ctype)
                dis = M.item()
            elif dissimilarity == 'js':
                M = kl_divergence_backend(M_A_ctype, M_B_ctype)
                dis1 = M.item()
                M = kl_divergence_backend(M_B_ctype, M_A_ctype)
                dis2 = M.item()
                dis = (dis1 + dis2) / 2
        return dis

    sliceA = gene_selection_slice(sliceA, n_top_genes_val=2000)
    sliceB = gene_selection_slice(sliceB, n_top_genes_val=2000)

    # if 'anno_cell' in sliceA.obs.columns:
    #     label_colname = 'anno_cell'
    # elif 'annotation' in sliceA.obs.columns:
    #     label_colname = 'annotation'

    ctype_wei = {}  # 用于存储sliceA中的每种标签的距离

    for ctype in set(sliceA.obs[label_col]):
        dis = find_dis_of_the_type(sliceA, sliceB, label_col, ctype=ctype, dissimilarity=dissimilarity)
        ctype_wei[ctype] = dis

    a_dis = np.array([ctype_wei[ctype] for ctype in sliceA.obs[label_col].tolist()])
    b_dis = np.array([ctype_wei[ctype] for ctype in sliceB.obs[label_col].tolist()])

    a_wei = dis2weight(a_dis, map_method_dis2wei)
    b_wei = dis2weight(b_dis, map_method_dis2wei)

    return a_wei, b_wei, -1 * a_dis, -1 * b_dis


def cal_weight(uniform_weight, nx, sliceA, sliceB, label_col, map_method_dis2wei, dissimilarity_weight):
    """
    calculate margin lists, weighted by cell type similarity, or with uniform weight
    """
    if uniform_weight:
        a = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
        b = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]
        a_wei_abs = None
        b_wei_abs = None
    else:
        # a, b, a_wei_abs, b_wei_abs = cal_weight_un_uniform_per_spot(M, nx, norm_method)
        a, b, a_wei_abs, b_wei_abs = cal_weight_un_uniform_per_group(sliceA, sliceB, label_col, dissimilarity=dissimilarity_weight, map_method_dis2wei=map_method_dis2wei)  # 需要改写输入，并减少内存占用
        a = nx.from_numpy(a)
        b = nx.from_numpy(b)

    if isinstance(nx, ot.backend.TorchBackend):
        a = a.float()
        b = b.float()

        a = a.cuda()
        b = b.cuda()

        gc.collect()
    return a, b, a_wei_abs, b_wei_abs
