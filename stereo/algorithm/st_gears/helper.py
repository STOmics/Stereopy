import ot
import scipy
import numpy as np
import scanpy as sc
import gc
import collections

from stereo.log_manager import logger

to_dense_array = lambda X: X.toarray() if isinstance(X, scipy.sparse.spmatrix) else np.array(X)  # scipy.sparse.csr.spmatrix


def extract_data_matrix(adata, rep):
    """
    extract expression matrix
    """
    if rep is None:
        return adata.X
    elif rep.split('.')[0] == 'obsm':
        return adata.obsm[rep.split('.')[1]]
    elif rep.split('.')[0] == 'layers':
        return adata.layers[rep.split('.')[1]]


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def determine_using_gpu(use_gpu):
    """
    Determine using gpu, or using cpu, or the program needs to be ended, for the sake of users to intall necessary packages.
    """
    assert type(use_gpu) == bool, "Type of use_gpu is not bool"

    if not use_gpu:
        use_gpu = False
        return use_gpu

    else:
        try:
            import torch
        except:
            logger.warning("We currently only have GPU support for torch. Please install torch before assigning GPU using, or assign use_gpu as False to use CPU")
            quit()

        if not torch.cuda.is_available():
            logger.warning("torch.cuda is not available. Please make sure cuda is installed, or assign use_gpu as False to use CPU")
            quit()
        else:
            use_gpu = True
            return use_gpu


def filter_rows_cols(sliceA, sliceB, filter_by_label, label_col):
    """
    filter both genes and spot cell-types that is not on either one of the two slices.
    """

    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    if filter_by_label:
        common_ctype = intersect(set(sliceA.obs[label_col].tolist()), set(sliceB.obs[label_col].tolist()))
        sliceA = sliceA[sliceA.obs[label_col].isin(common_ctype)]
        sliceB = sliceB[sliceB.obs[label_col].isin(common_ctype)]
    else:
        pass

    sliceA = sliceA.copy()
    sliceB = sliceB.copy()

    return sliceA, sliceB


def gene_selection_slice(slice, n_top_genes_val):
    """
    get highly variable genes
    """
    sc.experimental.pp.highly_variable_genes(slice, flavor='pearson_residuals', n_top_genes=n_top_genes_val)
    slice = slice[:, slice.var['highly_variable']]
    return slice


def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X, Y)

    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)

    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i', X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X, log_Y.T)
    return nx.to_numpy(D)


def cal_dissimilarity(A_X, B_X, dissimilarity, nx):
    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
        # 此处需要补充归一化，如果效果不好就去掉欧式距离
        M = ot.dist(A_X, B_X)  # distance between features using sqeuclidean by default
    else:
        if isinstance(nx, ot.backend.TorchBackend):
            import torch
            offset = nx.min(torch.Tensor([nx.min(A_X), nx.min(B_X)]))
        elif isinstance(nx, ot.backend.NumpyBackend):
            offset = nx.min(np.array([nx.min(A_X), nx.min(B_X)]))
        s_A = A_X - offset + 0.01  # 以保证s_A和s_B都是正的
        s_B = B_X - offset + 0.01

        M = kl_divergence_backend(s_A, s_B)
        M = nx.from_numpy(M)
    return M


def generate_fea_simi_mtx(nx, sliceA, sliceB, dissimilarity):
    """
    generate feature similarity matrix
    """

    # 不降维
    A_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, None)))
    B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, None)))

    gc.collect()

    # reduce dimension
    # exp_dim_re_A = sc.tl.pca(sliceA.X, n_comps=200, svd_solver='arpack')  # 降维的表达矩阵
    # exp_dim_re_B = sc.tl.pca(sliceB.X, n_comps=200, svd_solver='arpack')

    # exp_dim_re_A = TSNE(n_components=20, random_state=33, method='exact').fit_transform(sliceA.X)
    # exp_dim_re_B = TSNE(n_components=20, random_state=33, method='exact').fit_transform(sliceB.X)

    # A_X, B_X = nx.from_numpy(to_dense_array(exp_dim_re_A)), nx.from_numpy(to_dense_array(exp_dim_re_B))

    if isinstance(nx, ot.backend.TorchBackend):
        # 送入到GPU
        A_X = A_X.float()
        B_X = B_X.float()

        A_X = A_X.cuda()
        B_X = B_X.cuda()

        gc.collect()

    M = cal_dissimilarity(A_X, B_X, dissimilarity, nx)

    if isinstance(nx, ot.backend.TorchBackend):
        M = M.cuda()
        gc.collect()
    return M


def generate_stru_mtx(nx, sliceA, sliceB, norm):
    """
    calculate spatial graph structures of sliceA and sliceB.
    """
    # Calculate spatial distances
    coordinatesA = sliceA.obsm['spatial'].copy()
    coordinatesB = sliceB.obsm['spatial'].copy()

    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = nx.from_numpy(coordinatesB)

    if isinstance(nx, ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()  # ? 尝试去掉，看区别
        coordinatesB = coordinatesB.float()

        coordinatesA = coordinatesA.cuda()
        coordinatesB = coordinatesB.cuda()

        gc.collect()

    D_A = ot.dist(coordinatesA, coordinatesA, metric='euclidean')  # 图中节点之间的结构相似性
    D_B = ot.dist(coordinatesB, coordinatesB, metric='euclidean')

    # if isinstance(nx, ot.backend.TorchBackend):
    #     D_A = D_A.cuda()
    #     D_B = D_B.cuda()

    if norm:
        D_A /= nx.max(D_A[D_A > 0])  # 源代码的min可能写错了，改为目前的版本
        D_B /= nx.max(D_B[D_B > 0])

    return D_A, D_B


def filter_pi_mtx(pi_mtx, percentage, same_mem):
    thresh = pi_mtx[pi_mtx > 0].min() + (pi_mtx.max() - pi_mtx[pi_mtx > 0].min()) * percentage / 100

    if same_mem is True:
        pi_mtx[pi_mtx <= thresh] = 0
        pi_mtx /= np.sum(pi_mtx)
        return pi_mtx

    else:
        pi_mtx_cp = pi_mtx.copy()
        pi_mtx_cp[pi_mtx_cp <= thresh] = 0
        pi_mtx_cp /= np.sum(pi_mtx_cp)
        return pi_mtx_cp


def gen_anncell_cid_from_all(slicesl, ctype_name):
    """
    generate dictionary from type to number, which ranges from 0 and increments at 1
    """
    anncell_set = []
    for slice in slicesl:
        anno_cell_li = list(collections.OrderedDict.fromkeys(slice.obs[ctype_name]).keys())
        anncell_add = anncell_set + anno_cell_li
        anncell_set = list(collections.OrderedDict.fromkeys(anncell_add).keys())
    return dict(zip(anncell_set, range(len(anncell_set))))

