# from multiprocessing import Pool
# import multiprocessing as mp
import sys
from typing import Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from ipywidgets import VBox
from numpy.random import RandomState
from pysal.explore import esda
from pysal.lib import weights
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.stats import fisher_exact, norm
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import holoviews as hv
import panel as pn

from stereo.core.stereo_exp_data import StereoExpData
from stereo.stereo_config import stereo_conf
from stereo.log_manager import logger

from .utils import kmeans_centers, nearest_neighbors
from ..utils import get_cell_coordinates


def get_ot_matrix(
    data: StereoExpData,
    data_type: str,
    alpha1: int = 0.5,
    alpha2: int = 0.5,
    basis: str = 'spatial',
    # random_state: Union[None, int, RandomState] = 0,
    pattern: Literal["run", "test", "test2"] = "run",
    n_pcs: int = 50,
    pca_res_key: str = 'pca'
) -> np.ndarray:
    """
    Calculate transfer probabilities between cells.

    Using optimal transport theory based on gene expression and/or spatial location information.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    data_type
        The type of sequencing data.

        - ``'spatial'``: for the spatial transcriptome data.
        - ``'single-cell'``: for the single-cell sequencing data.

    alpha1
        The proportion of spatial location information.
        (Default: 0.5)
    alpha2
        The proportion of gene expression information.
        (Default: 0.5)
    random_state
        Different initial states for the pca.
        (Default: 0)
    n_pcs
        The number of used pcs.
        (Default: 50)

    Returns
    -------
    :class:`~numpy.ndarray`
        Cell transition probability matrix.
    """

    position = get_cell_coordinates(data, basis=basis)

    if pca_res_key is None:
        pca_res_key = 'pca'

    if pca_res_key not in data.tl.result:
        logger.info("Can not find PCA result, to calculate automatically using highly variable genes.")
        logger.info(f"n_pcs: {n_pcs}")
        if 'highly_variable_genes' not in data.tl.result:
            use_highly_genes = False
        else:
            use_highly_genes = True
        data.tl.pca(use_highly_genes=use_highly_genes, hvg_res_key='highly_variable_genes', n_pcs=n_pcs, svd_solver="arpack", res_key=pca_res_key)
    
    pca_res = data.tl.result[pca_res_key].to_numpy()

    def getM(alpha1, alpha2):
        if data_type == "spatial":
            # calculate physical distance
            ed_coor = euclidean_distances(position, position, squared=True)
            m1 = ed_coor / sum(sum(ed_coor))
            # calculate gene expression PCA space distance
            ed_gene = euclidean_distances(pca_res, pca_res, squared=True)
            m2 = ed_gene / sum(sum(ed_gene))

            # M = alpha1 * m1 + alpha2 * m
            M = alpha2 * m1 + alpha1 * m2
            M /= M.max()

        elif data_type == "single-cell":
            ed = euclidean_distances(pca_res, pca_res, squared=True)
            M = ed / sum(sum(ed))
            M /= M.max()

        else:
            raise ValueError(
                "Please give the right data type, choose from 'spatial' or 'single-cell'."
            )

        return M

    if pattern == "run":
        logger.info(
            f"alpha1(gene expression): {alpha1}   alpha2(spatial information): {alpha2}"
        )
        M = getM(alpha1, alpha2)
    elif pattern == "test":
        alpha1 = autoAlpha1 = 0
        autoM = None
        minSumM = float("inf")

        while alpha1 <= 1.0 + 0.1 / 2:
            alpha1 = round(alpha1, 1)
            # print(alpha1)
            alpha2 = 1 - alpha1
            M = getM(alpha1, alpha2)

            sumM = np.sum(M)
            # logger.info(sumM)
            if sumM < minSumM:
                minSumM = sumM
                autoAlpha1 = alpha1
                autoM = M

            alpha1 += 0.1
        logger.info(
            f"auto alpha1(gene expression) = {autoAlpha1}   auto alpha2(spatial information) = {1-autoAlpha1}"
        )
        M = autoM
    elif pattern == "test2":
        alpha1 = autoAlpha1 = 0
        autoM = None
        maxSumM = 0

        while alpha1 <= 1.0 + 0.1 / 2:
            alpha1 = round(alpha1, 1)
            # print(alpha1)
            alpha2 = 1 - alpha1
            M = getM(alpha1, alpha2)

            sumM = np.sum(M)
            # print(sumM)
            if sumM > maxSumM:
                maxSumM = sumM
                autoAlpha1 = alpha1
                autoM = M

            alpha1 += 0.1
        logger.info(
            f"auto alpha1(gene expression) = {autoAlpha1}   auto alpha2(spatial information) = {1-autoAlpha1}"
        )
        M = autoM

    # Set the diagonal elements to large values
    row, col = np.diag_indices_from(M)
    M[row, col] = M.max() * 1000

    a, b = (
        np.ones((data.n_cells,)) / data.n_cells,
        np.ones((data.n_cells,)) / data.n_cells,
    )
    lambd = 1e-1
    Gs = np.array(ot.sinkhorn(a, b, M, lambd))

    return Gs


def auto_estimate_para(
    data: StereoExpData,
    basis: str = 'spatial',
    hvg_gene_number: int = 2000,
    # hvg_key: Optional[str] = None
):
    """
    Parameters
    ----------
    data
        An object of StereoExpData.
    hvg_gene_number
        highly variable expression gene number (Default: 2000)

    """
    ## 01 Average expression value of highly variable genes in each spot
    # if hvg_key is None:
    #     hvg_key = 'highly_variable'
    
    # if hvg_key == 'highly_variable' and hvg_key not in data.genes:
    #     logger.info("Can not find highly variable genes, to calculate automatically.")
    #     logger.info(f"hvg_gene_number: {hvg_gene_number}.")
    #     data.tl.highly_variable_genes(
    #         min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=hvg_gene_number, res_key='highly_variable_genes'
    #     )
    # elif hvg_key not in data.genes:
    #     raise ValueError(f"Can not find the key '{hvg_key}' in genes/var.")
    data.tl.highly_variable_genes(
        min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=hvg_gene_number, res_key='highly_variable_genes'
    )

    hvg_exp = data.exp_matrix[:, data.genes['highly_variable']]
    hvg_exp_mean = np.array(hvg_exp.mean(axis=1)).flatten()

    ## 02 measure alpah1  and alpah2 using moran I
    position = get_cell_coordinates(data, basis=basis)
    w_xy = weights.KNN(position, k=10)

    moran_res = esda.Moran(hvg_exp_mean, w_xy)
    a1 = 0.5
    a2 = 0.5 + (moran_res.I.round(3)/2)
    logger.info(f"Parameter estimation of alpah1 for gene expression is: {a1}")
    logger.info(f"Parameter estimation of alpah2 for spatial distance is: {a2.round(3)}")
    return a1, a2

def set_start_cells(
    data: StereoExpData,
    select_way: Literal["coordinates", "cell_type"],
    cell_type: Optional[str] = None,
    start_point: Optional[Tuple[int, int]] = None,
    basis: str = 'spatial',
    use_col: str = "cluster",
    split: bool = False,
    n_clusters: int = 2,
    n_neigh: int = 5,
) -> list:
    """
    Use coordinates or cell type to manually select starting cells.

    Parameters
    ----------
    data
        An object of StereoExpData.
    select_way
        Ways to select starting cells.

        (1) ``'cell_type'``: select by cell type.
        (2) ``'coordinates'``: select by coordinates.

    cell_type
        Restrict the cell type of starting cells.
        (Deafult: None)
    start_point
        The coordinates of the start point in 'coordinates' mode.
    basis
        The key to get position information of cells.
    use_col
        The cells/obs column name specifying the cell type.
    split
        Whether to split the specific type of cells into several small clusters according to cell density.
    n_clsuters
        The number of cluster centers after splitting.
    n_neigh
        The number of neighbors next to the start point/cluster center selected as the starting cell.
    add_col
        Name of the cells/obs column in which stores the symbol representing the start cells.

    Returns
    -------
    list
        The index number of selected starting cells.
    """
    position = get_cell_coordinates(data, basis=basis)

    if select_way == "coordinates":
        if start_point is None:
            raise ValueError(
                "`start_point` must be specified in the 'coordinates' mode."
            )

        start_cells = nearest_neighbors(start_point, position, n_neigh)[0]

        if cell_type is not None:
            type_cells = np.where(data.cells[use_col] == cell_type)[0]
            start_cells = np.intersect1d(start_cells, type_cells)

    elif select_way == "cell_type":
        if cell_type is None:
            raise ValueError("in 'cell_type' mode, `cell_type` cannot be None.")

        start_cells = np.where(data.cells[use_col] == cell_type)[0]

        if split:
            mask = data.cells[use_col] == cell_type
            cell_coords = position[mask]
            cluster_centers = kmeans_centers(cell_coords, n_clusters=n_clusters)

            select_cluster_coords = position.copy()
            select_cluster_coords[np.logical_not(mask)] = 1e10
            start_cells = nearest_neighbors(
                cluster_centers, select_cluster_coords, n_neigh
            ).flatten()
    else:
        raise ValueError("`select_way` must choose from 'coordinates' or 'cell_type'.")

    return start_cells

def calc_alpha_by_moransI(data: StereoExpData, basis: str = 'spatial'):

    def moransI(coords, bandwidth, x, w_type="Binary"):
        distances = np.linalg.norm(coords - coords[:, np.newaxis], axis=2)
        dij = distances.copy()

        obs = len(x)

        if bandwidth >= obs:
            bandwidth = obs - 1
            print(f"Bandwidth set to: {bandwidth}")

        moran_nom = 0.0
        moran_denom = 0.0
        mean_x = np.mean(x)

        wts = np.zeros((obs, obs))

        for i in range(obs):
            # Get the data and add the distances
            data_set = np.column_stack((x, dij[:, i]))

            # Sort by distance
            data_set_sorted = data_set[data_set[:, 1].argsort()]

            # Keep nearest neighbours
            sub_set1 = data_set_sorted[1 : (bandwidth + 1), :]

            # Find furthest neighbour
            kernel_h = np.max(sub_set1[:, 1])

            # Calculate weights
            for j in range(obs):
                if data_set[j, 1] > kernel_h:
                    wts[i, j] = 0
                else:
                    wts[i, j] = 1

                if j != i:
                    moran_nom += wts[i, j] * (x[i] - mean_x) * (x[j] - mean_x)

            moran_denom += (x[i] - mean_x) * (x[i] - mean_x)

        np.fill_diagonal(wts, 0)
        sum_w = np.sum(wts)

        nom = obs * moran_nom
        denom = sum_w * moran_denom

        morans_i = nom / denom
        if morans_i > 1:
            morans_i = 1
        if morans_i < -1:
            morans_i = -1
        return morans_i
    
    pca = PCA(n_components=10, random_state=10086)
    pca.fit(data.exp_matrix.toarray().T if data.issparse() else data.exp_matrix.T)
    pca2 = pca.components_

    mi2 = []
    coords = get_cell_coordinates(data=data, basis=basis)
    for i in range(10):# range num = Number of PCs
        mi = moransI(coords, 8, pca2[i])  # 计算每个pc的moran i
        mi2.append(mi)

    var_w = pca.explained_variance_ratio_
    result = np.sum(np.array(var_w) * np.array(mi2))

    alpha1 = 1 / (1 + abs(result))
    alpha2 = 1 - alpha1
    logger.info(f"Morans'I value is {result}, the estimated values of alpha1 and alpha2 are {alpha1} and {alpha2}.")
    return alpha1, alpha2


def get_ptime(
    data: StereoExpData,
    start_cells_key: str = 'start_cells',
    start_cells: Optional[Union[list, np.ndarray]] = None,
    ot_mtx_key: str = 'trans'
):
    """
    Get the cell pseudotime based on transition probabilities from initial cells.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    start_cells
        List of index numbers of starting cells.

    Returns
    -------
    :class:`~numpy.ndarray`
        Ptime correspongding to cells.
    """

    if start_cells is None:
        if start_cells_key not in data.cells:
            raise ValueError(f"Please set the start cells first, the key should be '{start_cells_key}'.")
        start_cells = data.cells[start_cells_key].to_numpy()
    select_trans = data.tl.result[ot_mtx_key][start_cells]
    cell_tran = np.sum(select_trans, axis=0)
    # data.cells["tran"] = cell_tran
    cell_tran_sort = np.argsort(cell_tran)[::-1]

    ptime = np.zeros(data.n_cells, dtype=np.float64)
    ptime[cell_tran_sort] = np.arange(data.n_cells) / (data.n_cells - 1)

    return ptime


def get_neigh_trans(
    data: StereoExpData,
    basis: str = 'spatial',
    n_neigh_pos: int = 10,
    n_neigh_gene: int = 0,
    ot_mtx_key: str = 'trans'
):
    """
    Get the transport neighbors from two ways, position and/or gene expression

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    basis
        The basis used in visualizing the cell position.
    n_neigh_pos
        Number of neighbors based on cell positions such as spatial or umap coordinates.
        (Default: 10)
    n_neigh_gene
        Number of neighbors based on gene expression (PCA).
        (Default: 0)

    Returns
    -------
    :class:`~scipy.sparse._csr.csr_matrix`
        A sparse matrix composed of transition probabilities of selected neighbor cells.
    """
    if n_neigh_pos == 0 and n_neigh_gene == 0:
        raise ValueError(
            "the number of position neighbors and gene neighbors cannot be zero at the same time."
        )

    if n_neigh_pos:
        position = get_cell_coordinates(data, basis=basis)
        nn = NearestNeighbors(n_neighbors=n_neigh_pos, n_jobs=-1)
        nn.fit(position)
        dist_pos, neigh_pos = nn.kneighbors(position)
        dist_pos = dist_pos[:, 1:]
        neigh_pos = neigh_pos[:, 1:]

        neigh_pos_list = []
        for i in range(data.n_cells):
            idx = neigh_pos[i]  # embedding上的邻居
            idx2 = neigh_pos[idx]  # embedding上邻居的邻居
            idx2 = np.setdiff1d(idx2, i)

            neigh_pos_list.append(np.unique(np.concatenate([idx, idx2])))
            # neigh_pos_list.append(idx)

    if n_neigh_gene:
        # if "X_pca" not in adata.obsm:
        #     print("X_pca is not in adata.obsm, automatically do PCA first.")
        #     sc.tl.pca(adata)
        # sc.pp.neighbors(
        #     adata, use_rep="X_pca", key_added="X_pca", n_neighbors=n_neigh_gene
        # )
        logger.info("Automatically calculate neighbors using the pca result spcified by key pca.")
        logger.info(f"The number of neighbors is: {n_neigh_gene}")
        logger.info(f"The neighbors are stored in 'nbrs_velocity'.")
        data.tl.neighbors(pca_res_key='pca', n_neighbors=n_neigh_gene, res_key='nbrs_velocity')

        neigh_gene = data.tl.result['nbrs_velocity']["nn_dist"].indices.reshape(
            -1, n_neigh_gene - 1
        )
        logger.info(neigh_gene.shape)
        # neigh_gene_indptr = data.tl.result['neighbors_spa_track']["nn_dist"].indptr
        # neigh_gene_indices = data.tl.result['neighbors_spa_track']["nn_dist"].indices

    indptr = [0]
    indices = []
    csr_data = []
    count = 0
    for i in range(data.n_cells):
        if n_neigh_pos == 0:
            n_all = neigh_gene[i]
            # n_all = neigh_gene_indices[neigh_gene_indptr[i]:neigh_gene_indptr[i+1]]
        elif n_neigh_gene == 0:
            n_all = neigh_pos_list[i]
        else:
            n_all = np.unique(np.concatenate([neigh_pos_list[i], neigh_gene[i]]))
            # n_all = np.unique(np.concatenate([neigh_pos_list[i], neigh_gene_indices[neigh_gene_indptr[i]:neigh_gene_indptr[i+1]]]))
        count += len(n_all)
        indptr.append(count)
        indices.extend(n_all)
        csr_data.extend(
            data.tl.result[ot_mtx_key][i][n_all]
            / (data.tl.result[ot_mtx_key][i][n_all].sum())  # normalize
        )

    trans_neigh_csr = csr_matrix(
        (csr_data, indices, indptr), shape=(data.n_cells, data.n_cells)
    )

    return trans_neigh_csr


def get_velocity(
    data: StereoExpData,
    basis: str = 'spatial',
    n_neigh_pos: int = 10,
    n_neigh_gene: int = 0,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
    pseudotime_key: str = 'ptime',
    ot_mtx_key: str = 'trans'
) -> tuple:
    """
    Get the velocity of each cell.

    The speed can be determined in terms of the cell location and/or gene expression.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    basis
        The key to get position information of cells.
    n_neigh_pos
        Number of neighbors based on cell positions such as spatial or umap coordinates.
        (Default: 10)
    n_neigh_gene
        Number of neighbors based on gene expression.
        (Default: 0)

    Returns
    -------
    tuple
        The grid coordinates and cell velocities on each grid to draw the streamplot figure.
    """
    trans_neigh_csr: csr_matrix = get_neigh_trans(
        data, basis, n_neigh_pos, n_neigh_gene, ot_mtx_key=ot_mtx_key
    )

    position = get_cell_coordinates(data, basis=basis, ndmin=2)
    V = np.zeros(position.shape)  # 速度为2维

    for cell in range(data.n_cells):  # 循环每个细胞
        cell_u = 0.0  # 初始化细胞速度
        cell_v = 0.0
        x1 = position[cell][0]  # 初始化细胞坐标
        y1 = position[cell][1]
        for neigh in trans_neigh_csr[cell].indices:  # 针对每个邻居
            p = trans_neigh_csr[cell, neigh]
            if (
                data.cells[pseudotime_key][neigh] < data.cells[pseudotime_key][cell]
            ):  # 若邻居的ptime小于当前的，则概率反向
                p = -p

            x2 = position[neigh][0]
            y2 = position[neigh][1]

            # 正交向量确定速度方向，乘上概率确定速度大小
            sub_u = p * (x2 - x1) / (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            sub_v = p * (y2 - y1) / (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            cell_u += sub_u
            cell_v += sub_v
        V[cell][0] = cell_u / trans_neigh_csr[cell].indptr[1]
        V[cell][1] = cell_v / trans_neigh_csr[cell].indptr[1]
    # adata.obsm["velocity_" + basis] = V
    # print(f"The velocity of cells store in 'velocity_{basis}'.")

    P_grid, V_grid = get_velocity_grid(
        data,
        P=position,
        V=V,
        grid_num=grid_num,
        smooth=smooth,
        density=density,
    )
    return P_grid, V_grid, trans_neigh_csr, V


def get_2_biggest_clusters(
    data: StereoExpData,
    use_col: str = 'cluster',
    ot_matx_key: str = 'trans'
):
    """Automatic selection of starting cells to determine the direction of cell trajectories

    Args:
        data: An object of StereoExpData

    Returns:
        tuple: Contains 2 cluster with maximum sum of transition probabilities
    """
    clusters = np.unique(data.cells[use_col])
    cluster_trans = pd.DataFrame(index=clusters, columns=clusters)
    for start_cluster in clusters:
        for end_cluster in clusters:
            if start_cluster == end_cluster:
                cluster_trans.loc[start_cluster, end_cluster] = 0
                continue
            starts = data.cells[use_col] == start_cluster
            ends = data.cells[use_col] == end_cluster
            cluster_trans.loc[start_cluster, end_cluster] = (
                np.sum(data.tl.result[ot_matx_key][starts][:, ends])
                / np.sum(starts)
                / np.sum(ends)
            )
    highest_2_clusters = cluster_trans.stack().astype(float).idxmax()
    return highest_2_clusters


def auto_get_start_cluster(
    data: StereoExpData,
    use_col: str = 'cluster',
    ot_mtx_key: str = 'trans',
    clusters: Optional[list] = None
):
    """
    Select the start cluster with the largest sum of transfer probability

    Parameters
    ----------
    data
        An object of StereoExpData
    clusters, list
        Give clusters to find, by default None, each cluster will be traversed and calculated

    Returns
    -------
    str
        One cluster with maximum sum of transition probabilities
    """
    if clusters is None:
        clusters = np.unique(data.cells[use_col])
    cluster_prob_sum = {}
    for cluster in clusters:
        start_cells = set_start_cells(data, select_way="cell_type", cell_type=cluster, use_col=use_col)
        ptime = get_ptime(data, start_cells=start_cells, ot_mtx_key=ot_mtx_key)
        cell_time_sort = ptime.argsort()

        prob_sum = 0
        for i in range(len(cell_time_sort) - 1):
            pre = cell_time_sort[i]
            next = cell_time_sort[i + 1]
            prob = data.tl.result[ot_mtx_key][pre, next]
            prob_sum += prob
        cluster_prob_sum[cluster] = prob_sum
    highest_cluster = max(cluster_prob_sum, key=cluster_prob_sum.get)

    print(
        "The auto selecting cluster is: '"
        + highest_cluster
        + "'. If there is a large discrepancy with the known biological knowledge, please manually select the starting cluster."
    )

    return highest_cluster


def get_velocity_grid(
    data: StereoExpData,
    P: np.ndarray,
    V: np.ndarray,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
) -> tuple:
    """
    Convert cell velocity to grid velocity for streamline display

    The visualization of vector field borrows idea from scTour: https://github.com/LiQian-XC/sctour/blob/main/sctour.

    Parameters
    ----------
    P
        The position of cells.
    V
        The velocity of cells.
    smooth
        The factor for scale in Gaussian pdf.
        (Default: 0.5)
    density
        grid density
        (Default: 1.0)
    Returns
    ----------
    tuple
        The embedding and unitary displacement vectors in grid level.
    """
    grids = []
    for dim in range(P.shape[1]):
        m, M = np.min(P[:, dim]), np.max(P[:, dim])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(grid_num * density))
        grids.append(gr)

    meshes = np.meshgrid(*grids)
    P_grid = np.vstack([i.flat for i in meshes]).T

    n_neighbors = int(P.shape[0] / grid_num)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(P)
    dists, neighs = nn.kneighbors(P_grid)

    scale = np.mean([grid[1] - grid[0] for grid in grids]) * smooth
    weight = norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]

    P_grid = np.stack(grids)
    ns = P_grid.shape[1]
    V_grid = V_grid.T.reshape(2, ns, ns)

    mass = np.sqrt((V_grid * V_grid).sum(0))
    min_mass = 1e-5
    min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
    cutoff = mass < min_mass

    V_grid[0][cutoff] = np.nan

    # adata.uns["P_grid"] = P_grid
    # adata.uns["V_grid"] = V_grid

    return P_grid, V_grid


# class Lasso:
#     """
#     Lasso an region of interest (ROI) based on spatial cluster.

#     Parameters
#     ----------
#     adata
#         An :class:`~anndata.AnnData` object.
#     """

#     __sub_index = []
#     sub_cells = []

#     def __init__(self, data: StereoExpData):
#         self.data = data

#     def vi_plot(
#         self,
#         use_col: str = 'cluster',
#         cell_type: Optional[str] = None,
#     ):
#         """
#         Plot figures.

#         Parameters
#         ----------
#         basis
#             The basis in `adata.obsm` to store position information.
#             (Deafult: 'spatial')
#         cell_type
#             Restrict the cell type of starting cells.
#             (Deafult: None)

#         Returns
#         -------
#             The container of cell scatter plot and table.
#         """
#         cell_types = self.data.cells[use_col].unique()
#         # colors = sns.color_palette(n_colors=len(cell_types)).as_hex()
#         # cluster_color = dict(zip(cell_types, colors))
#         # self.adata.uns["cluster_color"] = cluster_color
#         colors = stereo_conf.get_colors('stereo_30', n=len(cell_types))
#         cluster_colors = dict(zip(cell_types, colors))

#         df = pd.DataFrame()
#         df["group_ID"] = self.data.cell_names
#         df["labels"] = self.data.cells[use_col].to_numpy()
#         df["spatial_0"] = self.data.position[:, 0]
#         df["spatial_1"] = self.data.position[:, 1]
#         df["color"] = df["labels"].map(cluster_colors)

#         py.init_notebook_mode()

#         f = go.FigureWidget(
#             [
#                 go.Scatter(
#                     x=df["spatial_0"],
#                     y=df["spatial_1"],
#                     mode="markers",
#                     marker_color=df["color"],
#                 )
#             ]
#         )
#         scatter = f.data[0]
#         f.layout.plot_bgcolor = "rgb(255,255,255)"
#         f.layout.autosize = False

#         axis_dict = dict(
#             showticklabels=True,
#             autorange=True,
#         )
#         f.layout.yaxis = axis_dict
#         f.layout.xaxis = axis_dict
#         f.layout.width = 600
#         f.layout.height = 600

#         # Create a table FigureWidget that updates on selection from points in the scatter plot of f
#         t = go.FigureWidget(
#             [
#                 go.Table(
#                     header=dict(
#                         values=["group_ID", "labels", "spatial_0", "spatial_1"],
#                         fill=dict(color="#C2D4FF"),
#                         align=["left"] * 5,
#                     ),
#                     cells=dict(
#                         values=[
#                             df[col]
#                             for col in ["group_ID", "labels", "spatial_0", "spatial_1"]
#                         ],
#                         fill=dict(color="#F5F8FF"),
#                         align=["left"] * 5,
#                     ),
#                 )
#             ]
#         )

#         def selection_fn(trace, points, selector):

#             t.data[0].cells.values = [
#                 df.loc[points.point_inds][col]
#                 for col in ["group_ID", "labels", "spatial_0", "spatial_1"]
#             ]

#             Lasso.__sub_index = t.data[0].cells.values[0]
#             Lasso.sub_cells = np.where(np.isin(self.data.cell_names, Lasso.__sub_index))[0]

#             if cell_type is not None:
#                 type_cells = np.where(self.data.cells[use_col] == cell_type)[0]
#                 Lasso.sub_cells = sorted(
#                     set(Lasso.sub_cells).intersection(set(type_cells))
#                 )
#             if len(Lasso.sub_cells) > 0:
#                 res_key: str = 'start_cells'
#                 self.data.cells[res_key] = False
#                 self.data.cells.loc[self.data.cell_names[Lasso.sub_cells], res_key] = True
#                 self.data.cells[res_key] = self.data.cells[res_key].astype('category')

#         scatter.on_selection(selection_fn)

#         # Put everything together
#         return VBox((f, t))

def lasso_select(
    data: StereoExpData,
    use_col: str = 'cluster',
    cell_type: Optional[str] = None,
    basis: str = 'spatial',
    bg_color: str = '#2F2F4F',
    palette: str = 'stereo_30',
    marker: str = 'o',
    marker_size: int = 5,
    width: int = 500,
    height: int = 500,
    invert_y: bool = True
):
    hv.extension("bokeh")
    pn.extension()
    cell_types = data.cells[use_col].unique()
    colors = stereo_conf.get_colors(palette, n=len(cell_types))
    cluster_colors = dict(zip(cell_types, colors))

    show_data_df = pd.DataFrame()
    show_data_df["cell"] = data.cell_names
    show_data_df["cluster"] = data.cells[use_col].to_numpy()
    position = get_cell_coordinates(data, basis=basis, ndmin=2)
    show_data_df["x"] = position[:, 0]
    show_data_df["y"] = position[:, 1]

    scatter = hv.Scatter(show_data_df, kdims=['x'], vdims=['y', 'cell', 'cluster']).opts(
        bgcolor=bg_color,
        color='cluster',
        cmap=cluster_colors,
        size=marker_size,
        marker=marker,
        width=width,
        height=height,
        xaxis=None,
        yaxis=None,
        invert_yaxis=invert_y,
        active_tools=['wheel_zoom'],
        tools=['lasso_select', 'box_select'],
        show_legend=False
    )

    selection1d = hv.streams.Selection1D(source=scatter)

    selected_index = None

    def __selection1d_callback_1(index):
        nonlocal selected_index
        selected_index = index
        if index is not None and len(index) > 0:
            df = show_data_df.iloc[index]
            if cell_type is not None:
                df = df[df['cluster'] == cell_type]
            return hv.Table(df).opts(editable=False, fit_columns=True, sortable=False, width=width * 2 - 150)
        else:
            return hv.Table(pd.DataFrame(show_data_df)).opts(
                editable=False, fit_columns=True, sortable=False, width=width * 2 - 150)
        
    def __selection1d_callback_2(index):
        df = show_data_df.copy(deep=True)
        df['color'] = 'gray'
        if index is not None and len(index) > 0:
            sub_df = df.loc[index]
            if cell_type is not None:
                sub_df = sub_df[sub_df['cluster'] == cell_type]
            df.loc[sub_df.index, 'color'] = 'red'
        return hv.Scatter(df, kdims=['x'], vdims=['y', 'cell', 'cluster', 'color']).opts(
            bgcolor=bg_color,
            color='color',
            size=marker_size,
            marker=marker,
            width=width,
            height=height,
            xaxis=None,
            yaxis=None,
            invert_yaxis=invert_y,
            toolbar=None,
            show_legend=False
        )
    
    table_dmap = hv.DynamicMap(__selection1d_callback_1, streams=[selection1d])
    scatter_dmap = hv.DynamicMap(__selection1d_callback_2, streams=[selection1d])

    legend_df = pd.DataFrame({
        'x': [0] * len(cell_types),
        'y': [0] * len(cell_types),
        'cluster': cell_types,
    })

    legend_per_figure = 15
    legend_figures_count, left = divmod(len(cell_types), legend_per_figure)
    if left > 0:
        legend_figures_count += 1

    legend_figures = []
    for i in range(legend_figures_count):
        sub_df = legend_df[i*legend_per_figure:(i+1)*legend_per_figure]
        legfig = hv.Scatter(sub_df, kdims=['x'], vdims=['y', 'cluster']).opts(
            bgcolor='white',
            color='cluster',
            cmap=cluster_colors,
            size=0,
            marker='o',
            width=70 * sub_df.shape[0],
            height=80,
            xaxis=None,
            yaxis=None,
            toolbar=None,
            show_legend=True,
            legend_position='top',
            show_frame=False,
            legend_opts={'orientation': 'horizontal'},
            shared_axes=False
        )
        legend_figures.append(legfig)
    
    set_button = pn.widgets.Button(name='set start cells', button_type='primary', width=20)
    set_message = pn.widgets.StaticText(name='', value='')

    def __set_start_cells(_):
        set_button.loading = True
        set_message.value = ''
        try:
            if selected_index is not None and len(selected_index) > 0:
                data.cells['start_cells'] = False
                data.cells.loc[data.cell_names[selected_index], 'start_cells'] = True
                data.cells['start_cells'] = data.cells['start_cells'].astype('category')
                set_message.value = '<font color="red"><b>Set start cells successfully.</b></font>'
            else:
                pass
        finally:
            set_button.loading = False
    set_button.on_click(__set_start_cells)

    return pn.Column(
        pn.Column(*legend_figures),
        hv.Layout([scatter, scatter_dmap]),
        pn.Row(table_dmap, pn.Column(set_button, set_message))
    )
