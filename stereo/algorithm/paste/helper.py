from typing import (
    List,
    Optional,
    Tuple
)

# from anndata import AnnData
import numpy as np
import ot
import scipy

from stereo.core.stereo_exp_data import StereoExpData, AnnBasedStereoExpData


def filter_for_common_genes(slices: List[StereoExpData]) -> None:
    """
    Filters for the intersection of genes between all slices.

    Args:
        slices: List of slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].genes.gene_name
    for s in slices:
        common_genes = np.intersect1d(common_genes, s.genes.gene_name)
    for i in range(len(slices)):
        slices[i].tl.filter_genes(gene_list=common_genes)
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')


def match_spots_using_spatial_heuristic(
        X,
        Y,
        use_ot: bool = True
) -> np.ndarray:
    """
    Calculates and returns a mapping of spots using a spatial heuristic.

    Args:
        X (array-like, optional): Coordinates for spots X.
        Y (array-like, optional): Coordinates for spots Y.
        use_ot: If ``True``, use optimal transport ``ot.emd()`` to calculate mapping. Otherwise, use Scipy's ``min_weight_full_bipartite_matching()`` algorithm. # noqa

    Returns:
        Mapping of spots using a spatial heuristic.
    """
    n1, n2 = len(X), len(Y)
    X, Y = norm_and_center_coordinates(X), norm_and_center_coordinates(Y)
    dist = scipy.spatial.distance_matrix(X, Y)
    if use_ot:
        pi = ot.emd(np.ones(n1) / n1, np.ones(n2) / n2, dist)
    else:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(scipy.sparse.csr_matrix(dist))
        pi = np.zeros((n1, n2))
        pi[row_ind, col_ind] = 1 / max(n1, n2)
        if n1 < n2:
            pi[:, [(j not in col_ind) for j in range(n2)]] = 1 / (n1 * n2)
        elif n2 < n1:
            pi[[(i not in row_ind) for i in range(n1)], :] = 1 / (n1 * n2)
    return pi


def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X / X.sum(axis=1, keepdims=True)
    Y = Y / Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i], log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X, log_Y.T)
    return np.asarray(D)


def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
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
    # return np.intersect1d(lst1, lst2)


def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X - X.mean(axis=0)) / min(scipy.spatial.distance.pdist(X))


def apply_trsf(
        M: np.ndarray,
        translation: List[float],
        points: np.ndarray) -> np.ndarray:
    """
    Apply a rotation from a 2x2 rotation matrix `M` together with
    a translation from a translation vector of length 2 `translation` to a list of
    `points`.

    Args:
        M (nd.array): A 2x2 rotation matrix.
        translation (nd.array): A translation vector of length 2.
        points (nd.array): A nx2 array of `n` points 2D positions.

    Returns:
        (nd.array) A nx2 matrix of the `n` points transformed.
    """
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    trsf = np.identity(3)
    trsf[:-1, :-1] = M
    tr = np.identity(3)
    tr[:-1, -1] = -translation
    trsf = trsf @ tr

    flo = points.T
    flo_pad = np.pad(flo, ((0, 1), (0, 0)), constant_values=1)
    return ((trsf @ flo_pad)[:-1]).T


def to_sparse_matrix(m, to_type=scipy.sparse.csr_matrix):
    if isinstance(to_type, str):
        if to_type.lower() == 'csr' or to_type.lower() == 'csr_matrix':
            to_type = scipy.sparse.csr_matrix
        elif to_type.lower() == 'csc' or to_type.lower() == 'csc_matrix':
            to_type = scipy.sparse.csc_matrix
        else:
            raise ValueError('Just only can convert to csr matrix or csc matrix.')
    if to_type not in [scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]:
        raise ValueError('Just only can convert to csr matrix or csc matrix.')
    if not isinstance(m, to_type):
        m = to_type(m)
    return m


# Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X, scipy.sparse.csr.spmatrix) else np.array(X)  # noqa

# Returns the data matrix or representation
extract_data_matrix = lambda data, rep: data.exp_matrix if rep is None else data.tl.result[rep].to_numpy()  # noqa

"""
    Functions to plot slices and align spatial coordinates after obtaining a mapping from PASTE.
"""


def stack_slices_pairwise(
        slices: List[StereoExpData],
        pis: List[np.ndarray],
        output_params: bool = False,
        matrix: bool = False
) -> Tuple[List[StereoExpData], Optional[List[float]], Optional[List[np.ndarray]]]:
    """
    Align spatial coordinates of sequential pairwise slices.

    In other words, align:

        slices[0] --> slices[1] --> slices[2] --> ...

    Args:
        slices: List of slices.
        pis: List of pi (``pairwise_align()`` output) between consecutive slices.
        output_params: If ``True``, addtionally return angles of rotation (theta) and translations for each slice.
        matrix: If ``True`` and output_params is also ``True``, the rotation is
            return as a matrix instead of an angle for each slice.

    Returns:
        - List of slices with aligned spatial coordinates.

        If ``output_params = True``, additionally return:

        - List of angles of rotation (theta) for each slice.
        - List of translations [x_translation, y_translation] for each slice.
    """
    assert len(slices) == len(pis) + 1, "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."
    new_coor = []
    thetas = []
    translations = []
    if not output_params:
        S1, S2 = generalized_procrustes_analysis(slices[0].position, slices[1].position, pis[0])
    else:
        S1, S2, theta, tX, tY = generalized_procrustes_analysis(slices[0].position, slices[1].position, pis[0],
                                                                output_params=output_params, matrix=matrix)
        thetas.append(theta)
        translations.append(tX)
        translations.append(tY)
    new_coor.append(S1)
    new_coor.append(S2)
    for i in range(1, len(slices) - 1):
        if not output_params:
            x, y = generalized_procrustes_analysis(new_coor[i], slices[i + 1].position, pis[i])
        else:
            x, y, theta, tX, tY = generalized_procrustes_analysis(new_coor[i], slices[i + 1].position, pis[i],
                                                                  output_params=output_params, matrix=matrix)
            thetas.append(theta)
            translations.append(tY)
        new_coor.append(y)

    # new_slices = []
    for i in range(len(slices)):
        if isinstance(slices[i], AnnBasedStereoExpData):
            if slices[i].position_z is not None:
                slices[i].adata.obsm['spatial_paste_pairwise'] = np.concatenate((new_coor[i], slices[i].position_z), axis=1)
            else:
                slices[i].adata.obsm['spatial_paste_pairwise'] = new_coor[i]
            slices[i].spatial_key = 'spatial_paste_pairwise'
        else:
            slices[i].raw_position = slices[i].position
            slices[i].position = new_coor[i]

    if not output_params:
        return slices
    else:
        return slices, thetas, translations


def stack_slices_center(
        center_slice: StereoExpData,
        slices: List[StereoExpData],
        pis: List[np.ndarray],
        matrix: bool = False,
        output_params: bool = False
) -> Tuple[StereoExpData, List[StereoExpData], Optional[List[float]], Optional[List[np.ndarray]]]:
    """
    Align spatial coordinates of a list of slices to a center_slice.

    In other words, align:

        slices[0] --> center_slice

        slices[1] --> center_slice

        slices[2] --> center_slice

        ...

    Args:
        center_slice: Inferred center slice.
        slices: List of original slices to be aligned.
        pis: List of pi (``center_align()`` output) between center_slice and slices.
        output_params: If ``True``, additionally return angles of rotation (theta) and translations for each slice.
        matrix: If ``True`` and output_params is also ``True``, the rotation is
            return as a matrix instead of an angle for each slice.

    Returns:
        - Center slice with aligned spatial coordinates.
        - List of other slices with aligned spatial coordinates.

        If ``output_params = True``, additionally return:

        - List of angles of rotation (theta) for each slice.
        - List of translations [x_translation, y_translation] for each slice.
    """
    assert len(slices) == len(pis), "'slices' should have the same length 'pis'. Please double check."
    new_coor = []
    thetas = []
    translations = []

    for i in range(len(slices)):
        if not output_params:
            c, y = generalized_procrustes_analysis(center_slice.position, slices[i].position, pis[i])
        else:
            c, y, theta, tX, tY = generalized_procrustes_analysis(center_slice.position, slices[i].position, pis[i],
                                                                  output_params=output_params, matrix=matrix)
            thetas.append(theta)
            translations.append(tY)
        new_coor.append(y)

    for i in range(len(slices)):
        if isinstance(slices[i], AnnBasedStereoExpData):
            if slices[i].position_z is not None:
                slices[i].adata.obsm['spatial_paste_center'] = np.concatenate((new_coor[i], slices[i].position_z), axis=1)
            else:
                slices[i].adata.obsm['spatial_paste_center'] = new_coor[i]
            slices[i].spatial_key = 'spatial_paste_center'
        else:
            slices[i].raw_position = slices[i].position
            slices[i].position = new_coor[i]

    if isinstance(center_slice, AnnBasedStereoExpData):
        if center_slice.position_z is not None:
            center_slice.adata.obsm['spatial_paste_center'] = np.concatenate((center_slice.position, center_slice.position_z), axis=1)
        else:
            center_slice.adata.obsm['spatial_paste_center'] = c
        center_slice.spatial_key = 'spatial_paste_center'
    else:
        center_slice.raw_position = center_slice.position
        center_slice.position = c
    if not output_params:
        return center_slice, slices
    else:
        return center_slice, slices, thetas, translations


def generalized_procrustes_analysis(X, Y, pi, output_params=False, matrix=False):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).

    Args:
        X: np array of spatial coordinates (ex: sliceA.obs['spatial'])
        Y: np array of spatial coordinates (ex: sliceB.obs['spatial'])
        pi: mapping between the two layers output by PASTE
        output_params: Boolean of whether to return rotation angle and translations along with spatial coordiantes.
        matrix: Boolean of whether to return the rotation as a matrix or an angle.


    Returns:
        Aligned spatial coordinates of X, Y, rotation angle, translation of X, translation of Y.
    """
    assert X.shape[1] == 2 and Y.shape[1] == 2

    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T
    if output_params and not matrix:
        M = np.array([[0, -1], [1, 0]])
        theta = np.arctan(np.trace(M.dot(H)) / np.trace(H))
        return X, Y, theta, tX, tY
    elif output_params and matrix:
        return X, Y, R, tX, tY
    else:
        return X, Y
