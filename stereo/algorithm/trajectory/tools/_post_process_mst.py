import numpy as np


def post_process_mst(arr):
    """
    post process arr output by scipy.sparse.csgraph._min_spanning_tree

    Parameters
    ----------
    arr: asymmetrical square matrix storing un-directed topology

    Returns
    -------

    """
    arr_t = arr.transpose()

    mask0 = (arr == 0) & (arr_t == 0)  # both are zero
    mask_sin_po = ((arr > 0) & (arr_t == 0)) | ((arr == 0) & (arr_t > 0))  # one is positive, the other is negative
    mask_both_po = (arr > 0) & (arr_t > 0)  # both are positive

    arr_re = np.zeros(shape=arr.shape)
    arr_re[mask0] = 0
    arr_re[mask_sin_po] = np.maximum(arr[mask_sin_po], arr_t[mask_sin_po])
    arr_re[mask_both_po] = np.minimum(arr[mask_both_po], arr_t[mask_both_po])
    return arr_re