import numpy as np
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from ..log_manager import logger
from ..utils.time_consume import TimeConsume

def _gaussan_c(dis, gs=0.95, a=1, b=0):
    return np.sqrt(-(dis - b)**2 / 2 / np.log(gs / a))

def _gaussan_weight(dis, a=1, b=0, c=12500):
    gs = a * np.exp(-(dis - b)**2 / 2 / (c**2))
    return gs


def _sp_graph_weight(arr, a=1, b=0, c=12500):
    out = arr.copy()
    row, col = np.nonzero(arr)
    for ro, co in zip(row, col):
        out[ro, co] = _gaussan_weight(arr[ro, co], a=a, b=b, c=c)
    return out

def gaussian_smooth(pca_exp_matrix: np.ndarray,
                    raw_exp_matrix: np.ndarray,
                    cells_position: np.ndarray,
                    n_neighbors: int = 10,
                    smooth_threshold: float = 90,
                    a: float = 1,
                    b: float = 0,
                    n_jobs: int = 10):
    if sp.issparse(pca_exp_matrix):
        pca_exp_matrix = pca_exp_matrix.toarray()
    if sp.issparse(raw_exp_matrix):
        raw_exp_matrix = raw_exp_matrix.toarray()
    tc = TimeConsume()
    tk = tc.start()
    Euc_distance = distance.cdist(cells_position, cells_position).astype(np.float32)
    logger.info(f'distance.cdist: {tc.get_time_consumed(tk)}')

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=n_jobs).fit(pca_exp_matrix)
    logger.info(f'NearestNeighbors.fit: {tc.get_time_consumed(tk)}')

    adjecent_matrice = nbrs.kneighbors_graph(pca_exp_matrix).astype(np.float32).toarray()
    logger.info(f'kneighbors_graph: {tc.get_time_consumed(tk, restart=False)}')
    aa = np.multiply(adjecent_matrice, Euc_distance)  ## 自己和自己的距离为0
    aa_nonzero = aa[np.nonzero(aa)]
    # aa = aa.tocsr()
    dist_threshold = np.percentile(aa_nonzero, smooth_threshold)
    c = _gaussan_c(dist_threshold)
    ##### smoothing
    gauss_weight = _sp_graph_weight(aa, a, b, c)
    temp_nor_para = np.squeeze(np.asarray(np.sum(gauss_weight, axis=1)))

    new_adata = np.asarray(((gauss_weight.dot(raw_exp_matrix)).T / temp_nor_para).T)
    return new_adata
