import numba as nb
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from ..log_manager import logger
from ..utils.time_consume import TimeConsume


def _gaussian_c(dis, gs=0.95, a=1, b=0):
    return np.sqrt(-(dis - b) ** 2 / 2 / np.log(gs / a))


def _gaussian_weight(dis, a=1, b=0, c=12500):
    gs = a * np.exp(-(dis - b) ** 2 / 2 / (c ** 2))
    return gs


def _sp_graph_weight(m, a=1, b=0, c=12500):
    m.data = a * np.exp(-(m.data - b) ** 2 / 2 / (c ** 2))
    return m


@nb.njit(cache=True, nogil=True, parallel=True)
def _calculate_points_distance(cells_position: np.ndarray):
    counts = int((cells_position.shape[0] ** 2 - cells_position.shape[0]) / 2)
    cells_distance = np.zeros(counts, dtype=np.float32)
    for i in nb.prange(cells_position.shape[0]):
        count = cells_position.shape[0] - (i + 1)
        start = int(((count + 1) + (count + i)) * i / 2)
        end = start + count
        cells_distance[start:end] = np.sqrt(np.sum((cells_position[i] - cells_position[(i + 1):]) ** 2, axis=1))
    return cells_distance


def _create_nnd_matrix(
        points_count: int,
        n_neighbors: int,
        points_distance: np.ndarray,
        knn_graph_matrix: csr_matrix
):
    @nb.njit(cache=True, nogil=True)
    def __get_dist_idx(points_count, row_idx, cols_idx):
        idx = np.zeros(len(cols_idx), dtype=cols_idx.dtype)
        for m, col_idx in enumerate(cols_idx):
            if row_idx < col_idx:
                i = row_idx
                j = col_idx
            else:
                i = col_idx
                j = row_idx
            count = points_count - (i + 1)
            start = int(((count + 1) + (count + i)) * i / 2)
            idx[m] = start + (j - (i + 1))
        return idx

    @nb.njit(cache=True, nogil=True, parallel=True)
    def __creator(
            points_count: int,
            n_neighbors: int,
            points_distance: np.ndarray,
            knn_gm_indptr: np.ndarray,
            knn_gm_indices: np.ndarray,
    ):
        new_indptr = np.zeros(len(knn_gm_indptr), dtype=knn_gm_indptr.dtype)
        new_indices = np.zeros(len(knn_gm_indices) - points_count, dtype=knn_gm_indices.dtype)
        new_data = np.zeros(len(knn_gm_indices) - points_count, dtype=points_distance.dtype)

        for i in nb.prange(points_count):
            s, e = knn_gm_indptr[i], knn_gm_indptr[i + 1]
            ind = knn_gm_indices[s:e]
            new_s, new_e = i * (n_neighbors - 1), (i + 1) * (n_neighbors - 1)
            new_indptr[i], new_indptr[i + 1] = new_s, new_e
            new_indices[new_s:new_e] = ind[ind != i]
            dist_idx = __get_dist_idx(points_count, i, new_indices[new_s:new_e])
            new_data[new_s:new_e] = points_distance[dist_idx]
        return new_indptr, new_indices, new_data

    new_indptr, new_indices, new_data = __creator(
        points_count,
        n_neighbors,
        points_distance,
        knn_graph_matrix.indptr,
        knn_graph_matrix.indices
    )
    return csr_matrix((new_data, new_indices, new_indptr), shape=(points_count, points_count))


def _update_express_matrix(
        exp_matrix: csr_matrix,
        nnd_matrix: csr_matrix,
        a: float = 1,
        b: float = 0,
        c: float = None
):
    nnd_matrix = _sp_graph_weight(nnd_matrix, a, b, c)
    temp_nor_para = np.squeeze(np.asarray(np.sum(nnd_matrix, axis=1)))
    temp_matrix: csc_matrix = (nnd_matrix * exp_matrix).tocsc()
    temp_matrix.data /= temp_nor_para[temp_matrix.indices]
    return temp_matrix.tocsr()


def gaussian_smooth(
        pca_exp_matrix: np.ndarray,
        raw_exp_matrix: csr_matrix,
        cells_position: np.ndarray,
        n_neighbors: int = 10,
        smooth_threshold: float = 90,
        a: float = 1,
        b: float = 0,
        n_jobs: int = -1
):
    orig_jobs = nb.get_num_threads()
    nb.set_num_threads(n_jobs)
    try:
        tc = TimeConsume()
        tk = tc.start()

        logger.info(f'Calulate the distance between each cell in {cells_position.shape[0]} cells')
        cells_distance = _calculate_points_distance(cells_position)
        logger.debug(f'_calculate_points_distance: {tc.get_time_consumed(tk)}')

        logger.info(f'Calculate {n_neighbors} nearest neighbors for each cell')
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=n_jobs).fit(pca_exp_matrix)
        logger.debug(f'NearestNeighbors.fit: {tc.get_time_consumed(tk)}')

        knn_graph_matrix = nbrs.kneighbors_graph(pca_exp_matrix)
        logger.debug(f'kneighbors_graph: {tc.get_time_consumed(tk)}')

        nnd_matrix = _create_nnd_matrix(
            cells_position.shape[0],
            n_neighbors,
            cells_distance,
            knn_graph_matrix
        )
        logger.debug(f'_create_nnd_matrix: {tc.get_time_consumed(tk)}')

        dist_threshold = np.percentile(nnd_matrix.data, smooth_threshold)
        c = _gaussian_c(dist_threshold)
        logger.debug(f'_gaussian_c: {tc.get_time_consumed(tk)}')

        # smoothing
        logger.info('Update express matrix.')
        new_expression_matrix = _update_express_matrix(raw_exp_matrix, nnd_matrix, a, b, c)
        logger.debug(f'_update_expression_matrix: {tc.get_time_consumed(tk, restart=False)}')
        return new_expression_matrix
    finally:
        nb.set_num_threads(orig_jobs)
