import numba
import numpy as np


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def corr_spearman(ranked_mat_ref: np.ndarray, ranked_mat_qry: np.ndarray):
    n, k1, k2 = ranked_mat_ref.shape[0], ranked_mat_ref.shape[1], ranked_mat_qry.shape[1]
    mean = (n + 1) / 2.
    result = np.empty((k1, k2), dtype=np.float32)
    for xi in numba.prange(k1):
        for yi in numba.prange(k2):
            sum_x = sum_xx = sum_yy = 0
            for i in numba.prange(n):
                vx = ranked_mat_ref[i, xi] - mean
                vy = ranked_mat_qry[i, yi] - mean

                sum_x += vx * vy
                sum_xx += vx * vx
                sum_yy += vy * vy
            divisor = np.sqrt(sum_xx * sum_yy)
            if divisor != 0:
                result[xi, yi] = sum_x / divisor
            else:
                result[xi, yi] = np.NaN
    return result


@numba.njit(cache=True, fastmath=True, nogil=True)
def rankdata1d(a: np.ndarray) -> np.ndarray:
    arr = np.ravel(a)

    contains_nan = np.isnan(np.sum(a))
    if contains_nan:
        return np.full_like(arr, np.nan, dtype=np.float32)

    sorter = np.argsort(arr, kind='quicksort')
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(0, sorter.size, 1, np.intp)

    arr = arr[sorter]
    arr_a, arr_b = arr[1:], arr[:-1]
    obs = np.empty(arr_a.shape[0] + 1, dtype=np.bool_)
    obs[0] = True
    for idx in numba.prange(len(arr_a)):
        obs[idx + 1] = arr_a[idx] != arr_b[idx]
    dense = obs.cumsum()[inv]

    count_nonzero = np.nonzero(obs)[0]
    count = np.empty(len(count_nonzero) + 1, dtype=np.float32)
    for idx in numba.prange(len(count_nonzero)):
        count[idx] = count_nonzero[idx]
    count[len(count) - 1] = len(obs)
    return np.float32(.5) * (count[dense] + count[dense - 1] + np.float32(1))


@numba.njit(cache=True, parallel=True, nogil=True)
def apply_along_axis(arr: np.ndarray):
    res = np.empty(arr.shape, dtype=np.float32)
    for i in numba.prange(res.shape[1]):
        res[:, i] = rankdata1d(arr[:, i])
    return res
