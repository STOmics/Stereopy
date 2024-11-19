import anndata as ad
from typing import Callable, Union
import networkx as nx
import numpy as np
import scipy.sparse as sp
from warnings import warn
from scipy import interpolate
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# from .plot_least_action_path import *

from stereo.core.stereo_exp_data import StereoExpData
from stereo.log_manager import logger

from .utils import nearest_neighbors
from ..utils import get_cell_coordinates

# We caculated and visualized the LAP path along the trajectory using the corresponding functions implemented in the Dynamo.
# Ref: Qiu X, Zhang Y, Martin-Rufino JD, Weng C, Hosseinzadeh S, Yang D, et al. Mapping transcriptomic vector fields of single cells. Cell. 2022 Feb 17;185(4):690-711.e45. doi: 10.1016/j.cell.2021.12.045. 
# dynamo: https://github.com/aristoteleo/dynamo-release.

def distance_point_to_segment(point, segment_start, segment_end):
    segment_vector = segment_end - segment_start
    point_vector = point - segment_start
    projection = np.dot(point_vector, segment_vector) / np.dot(
        segment_vector, segment_vector
    )

    if projection < 0:
        distance = np.linalg.norm(point - segment_start)
        intersection = segment_start
    elif projection > 1:
        distance = np.linalg.norm(point - segment_end)
        intersection = segment_end
    else:
        distance = np.linalg.norm(
            np.cross(segment_vector, point_vector)
        ) / np.linalg.norm(segment_vector)
        intersection = segment_start + projection * segment_vector

    return distance, intersection


def map_cell_to_LAP(data: StereoExpData, basis='spatial', cell_neighbors=150):
    """
    Assign a new pseudotime value to each of these cells based on their position along the LAP.

    Parameters
    ----------
    data
        An object of StereoExpData.
    basis
        The embedding data.
        (Default: 'spatial')
    cell_neighbors
        The number of cell neighbors.
        (Default: 150)

    Returns
    -----------
    tuple
        The ptime of cells map to the LAP and selected neighbor cells. 
    """
    LAP_points = data.tl.result["LAP_"+basis]["prediction"][0]
    coords = get_cell_coordinates(data, basis=basis, ndmin=2)
    LAP_neighbor_cells = nearest_neighbors(LAP_points, coords, n_neighbors=cell_neighbors)
    LAP_neighbor_cells = np.unique(LAP_neighbor_cells.flatten())

    n_segments = len(LAP_points) - 1

    total_length_dict = {}
    total_length = 0
    for i in range(n_segments):  # do not traverse the last line segment
        segment_length = np.linalg.norm(LAP_points[i] - LAP_points[i + 1])
        total_length += segment_length
        total_length_dict[i] = total_length

    point_total_length_list = []
    for i in LAP_neighbor_cells:
        point = coords[i]
        min_distance = np.linalg.norm(point - LAP_points[0])
        min_intersection = LAP_points[0]
        min_segment = 0

        for j in range(n_segments):
            segment_start = LAP_points[j]
            segment_end = LAP_points[j + 1]
            distance, intersection = distance_point_to_segment(
                point, segment_start, segment_end
            )
            if distance < min_distance:
                min_distance = distance
                min_intersection = intersection
                min_segment = j
        point_total_length = total_length_dict[min_segment] - np.linalg.norm(
            min_intersection - LAP_points[min_segment + 1]
        )
        point_total_length_list.append(point_total_length)

    LAP_ptime = point_total_length_list / max(point_total_length_list)
    return LAP_ptime, LAP_neighbor_cells


def log1p_(adata, X_data):
    if "norm_method" not in adata.uns["pp"].keys():
        return X_data
    else:
        if adata.uns["pp"]["norm_method"] is None:
            if sp.issparse(X_data):
                X_data.data = np.log1p(X_data.data)
            else:
                X_data = np.log1p(X_data)

        return X_data

def fetch_states(data: StereoExpData, init_states, init_cells, basis, layer, average, t_end):
    if basis is not None:
        vf_key = "VecFld_" + basis
    else:
        vf_key = "VecFld"
    VecFld = data.tl.result[vf_key]
    X = VecFld["X"]
    valid_genes = None

    if init_states is None and init_cells is None:
        raise Exception("Either init_state or init_cells should be provided.")
    elif init_states is None and init_cells is not None:
        if isinstance(init_cells, (str, np.str_)):
            init_cells = [init_cells]
        intersect_cell_names = sorted(
            set(init_cells).intersection(data.cell_names),
            key=lambda x: list(init_cells).index(x),
        )
        _cell_names = init_cells if len(intersect_cell_names) == 0 else intersect_cell_names

        if basis is not None:
            # init_states = adata[_cell_names].obsm["X_" + basis].copy()
            coords = get_cell_coordinates(data, basis=basis)
            cells_index = data.cells.obs.index.get_indexer(_cell_names)
            init_states = coords[cells_index].copy()
            if len(_cell_names) == 1:
                init_states = init_states.reshape((1, -1))
            # VecFld = adata.uns["VecFld_" + basis]
            # X = adata.obsm["X_" + basis]
            X = coords

            valid_genes = [basis + "_" + str(i) for i in np.arange(init_states.shape[1])]

    if init_states.shape[0] > 1 and average in ["origin", "trajectory", True]:
        init_states = init_states.mean(0).reshape((1, -1))

    if t_end is None:
        t_end = getTend(X, VecFld["V"])

    if sp.issparse(init_states):
        init_states = init_states.A

    return init_states, VecFld, t_end, valid_genes

def getTend(X, V):
    xmin, xmax = X.min(0), X.max(0)
    V_abs = np.abs(V)
    t_end = np.max(xmax - xmin) / np.percentile(V_abs[V_abs > 0], 1)

    return t_end

def _nearest_neighbors(coord, coords, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    _, neighs = nbrs.kneighbors(np.atleast_2d(coord))
    return neighs

def arclength_sampling_n(X, num, t=None):
    arclen = np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))
    arclen = np.hstack((0, arclen))

    z = np.linspace(arclen[0], arclen[-1], num)
    X_ = interpolate.interp1d(arclen, X, axis=0)(z)
    if t is not None:
        t_ = interpolate.interp1d(arclen, t)(z)
        return X_, arclen[-1], t_
    else:
        return X_, arclen[-1]

def get_init_path(G, start, end, coords, interpolation_num=20):
    source_ind = _nearest_neighbors(start, coords, k=1)[0][0]
    target_ind = _nearest_neighbors(end, coords, k=1)[0][0]

    path = nx.shortest_path(G, source_ind, target_ind)
    init_path = coords[path, :]

    # _, arclen, _ = remove_redundant_points_trajectory(init_path, tol=1e-4, output_discard=True)
    # arc_stepsize = arclen / (interpolation_num - 1)
    # init_path_final, _, _ = arclength_sampling(init_path, step_length=arc_stepsize, t=np.arange(len(init_path)))
    init_path_final, _, _ = arclength_sampling_n(init_path, interpolation_num, t=np.arange(len(init_path)))

    # add the beginning and end point
    init_path_final = np.vstack((start, init_path_final, end))

    return init_path_final

def least_action_path(start, end, vf_func, jac_func, n_points=20, init_path=None, D=1, dt_0=1, EM_steps=2):
    if init_path is None:
        path = (
            np.tile(start, (n_points + 1, 1))
            + (np.linspace(0, 1, n_points + 1, endpoint=True) * np.tile(end - start, (n_points + 1, 1)).T).T
        )
    else:
        path = np.array(init_path, copy=True)

    # initial dt estimation:
    t_dict = minimize(lambda t: action(path, vf_func, D=D, dt=t), dt_0)
    dt = t_dict["x"][0]

    while EM_steps > 0:
        EM_steps -= 1
        path, dt, action_opt = lap_T(path, dt * len(path), vf_func, jac_func, D=D)

    return path, dt, action_opt

def action_aux(path_flatten, vf_func, dim, start=None, end=None, **kwargs):
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action(path, vf_func, **kwargs)

def action_grad_aux(path_flatten, vf_func, jac_func, dim, start=None, end=None, **kwargs):
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action_grad(path, vf_func, jac_func, **kwargs).flatten()

def reshape_path(path_flatten, dim, start=None, end=None):
    path = path_flatten.reshape(int(len(path_flatten) / dim), dim)
    if start is not None:
        path = np.vstack((start, path))
    if end is not None:
        path = np.vstack((path, end))
    return path

def action_grad(path, vf_func, jac_func, D=1, dt=1):
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt

    dv = v - vf_func(x)
    J = jac_func(x)
    z = np.zeros(dv.shape)
    for s in range(dv.shape[0]):
        z[s] = dv[s] @ J[:, :, s]
    grad = (dv[:-1] - dv[1:]) / D - dt / (2 * D) * (z[:-1] + z[1:])
    return grad

def lap_T(path_0, T, vf_func, jac_func, D=1):
    n = len(path_0)
    dt = T / (n - 1)
    dim = len(path_0[0])

    def fun(x):
        return action_aux(x, vf_func, dim, start=path_0[0], end=path_0[-1], D=D, dt=dt)

    def jac(x):
        return action_grad_aux(x, vf_func, jac_func, dim, start=path_0[0], end=path_0[-1], D=D, dt=dt)

    sol_dict = minimize(fun, path_0[1:-1], jac=jac)
    path_sol = reshape_path(sol_dict["x"], dim, start=path_0[0], end=path_0[-1])

    # further optimization by varying dt
    t_dict = minimize(lambda t: action(path_sol, vf_func, D=D, dt=t), dt)
    action_opt = t_dict["fun"]
    dt_sol = t_dict["x"][0]

    return path_sol, dt_sol, action_opt

def action(path, vf_func, D=1, dt=1):
    # centers
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt

    s = (v - vf_func(x)).flatten()
    s = 0.5 * s.dot(s) * dt / D

    return s

def minimize_lap_time(path_0, t0, t_min, vf_func, jac_func, D=1, num_t=20, elbow_method="hessian", hes_tol=3):
    T = np.linspace(t_min, t0, num_t)
    A = np.zeros(num_t)
    opt_T = np.zeros(num_t)
    laps = []

    for i, t in enumerate(T):
        path, dt, action = lap_T(path_0, t, vf_func, jac_func, D=D)
        A[i] = action
        opt_T[i] = dt * (len(path_0) - 1)
        laps.append(path)

    i_elbow = find_elbow(opt_T, A, method=elbow_method, order=-1, tol=hes_tol)

    return i_elbow, laps, A, opt_T

def normalize(x):
    x_min = np.min(x)
    return (x - x_min) / (np.max(x) - x_min)

def interp_second_derivative(t, f, num=5e2, interp_kind="cubic", **interp_kwargs):
    """
    interpolate f(t) and calculate the discrete second derivative using:
        d^2 f / dt^2 = (f(x+h1) - 2f(x) + f(x-h2)) / (h1 * h2)
    """
    t_ = np.linspace(t[0], t[-1], int(num))
    f_ = interpolate.interp1d(t, f, kind=interp_kind, **interp_kwargs)(t_)

    dt = np.diff(t_)
    df = np.diff(f_)
    t_ = t_[1:-1]

    d2fdt2 = np.zeros(len(t_))
    for i in range(len(t_)):
        d2fdt2[i] = (df[i + 1] - df[i]) / (dt[i + 1] * dt[i])

    return t_, d2fdt2

def interp_curvature(t, f, num=5e2, interp_kind="cubic", **interp_kwargs):
    """"""
    t_ = np.linspace(t[0], t[-1], int(num))
    f_ = interpolate.interp1d(t, f, kind=interp_kind, **interp_kwargs)(t_)

    dt = np.diff(t_)
    df = np.diff(f_)
    dfdt_ = df / dt

    t_ = t_[1:-1]
    d2fdt2 = np.zeros(len(t_))
    dfdt = np.zeros(len(t_))
    for i in range(len(t_)):
        dfdt[i] = (dfdt_[i] + dfdt_[i + 1]) / 2
        d2fdt2[i] = (df[i + 1] - df[i]) / (dt[i + 1] * dt[i])

    cur = d2fdt2 / (1 + dfdt * dfdt) ** 1.5

    return t_, cur

def kneedle_difference(t, f, type="decrease"):
    if type == "decrease":
        diag_line = lambda x: -x + 1
    elif type == "increase":
        diag_line = lambda x: x
    else:
        raise NotImplementedError(f"Unsupported function type {type}")

    t_ = normalize(t)
    f_ = normalize(f)
    res = np.abs(f_ - diag_line(t_))
    return res

def find_elbow(T, F, method="kneedle", order=1, **kwargs):
    i_elbow = None
    if method == "hessian":
        T_ = normalize(T)
        F_ = normalize(F)
        tol = kwargs.pop("tol", 2)
        t_, der = interp_second_derivative(T_, F_, **kwargs)

        found = False
        for i, t in enumerate(t_[::order]):
            if der[::order][i] > tol:
                i_elbow = np.argmin(np.abs(T_ - t))
                found = True
                break

        if not found:
            warn("The elbow was not found.")

    elif method == "curvature":
        T_ = normalize(T)
        F_ = normalize(F)
        t_, cur = interp_curvature(T_, F_, **kwargs)

        i_elbow = np.argmax(cur)

    elif method == "kneedle":
        type = "decrease" if order == -1 else "increase"
        res = kneedle_difference(T, F, type=type)
        i_elbow = np.argmax(res)
    else:
        raise NotImplementedError(f"The method {method} is not supported.")

    return i_elbow

def least_action(
    data: StereoExpData,
    init_cells: Union[str, list],
    target_cells: Union[str, list],
    basis: str = "spatial",
    vf_key: str = "VecFld",
    vecfld: Union[None, Callable] = None,
    adj_key: str = None,
    n_points: int = 25,
    n_neighbors: int =100,
    **kwargs,
):
    """
    Calculate the optimal paths between any two cell states.

    Parameters
    ----------
    data
        An object of StereoExpData.
    init_cells
        Cell name or indices of the initial cell states.
    target_cells
        Cell name or indices of the terminal cell states.
    basis
        The embedding data used to predict the least action path.
        (Default: "umap")
    vf_key
        A key to the vector field functions in adata.uns.
        (Default: "VecFld")
    vecfld
        The vector field function.
        (Default: None)
    adj_key
        The key to the adjacency matrix in adata.obsp.
        (Default: "pearson_transition_matrix")
    n_points
        The number of points on the least action path.
        (Default: 25)
    n_neighbors
        The number of neighbors.
        (Default: 100)

    Returns
    -----------
    LeastActionPath
        A trajectory class containing the least action paths information.
    """
    init_states,target_states = None,None
    paired = True
    min_lap_t = False
    elbow_method = "hessian"
    num_t = 20
    init_paths = None
    D = 10
    PCs = None 
    expr_func: callable = np.expm1
    add_key = None

    data.tl.neighbors(pca_res_key=basis, n_neighbors=n_neighbors, res_key=f'nbrs_lap_{basis}')

    # if vecfld is None:
    #     vf = SvcVectorField()
    #     vf.from_adata(adata, basis=basis, vf_key=vf_key)
    # else:
    #     vf = vecfld
    if vecfld is None:
        raise Exception("The vector field function is not provided.")
    vf = vecfld

    coords = get_cell_coordinates(data, basis=basis, ndmin=2)

    if adj_key is None:
        adj_key = f"nbrs_lap_{basis}_distances"
    assert adj_key in data.cells_pairwise, f"The key {adj_key} is not in cells_pairwise/obsp."
    T = data.cells_pairwise[adj_key]
    G = nx.from_scipy_sparse_array(T)

    init_states, _, _, _ = fetch_states(
        data,
        init_states,
        init_cells,
        basis,
        "X",
        False,
        None,
    )
    target_states, _, _, valid_genes = fetch_states(
        data,
        target_states,
        target_cells,
        basis,
        "X",
        False,
        None,
    )

    init_states = np.atleast_2d(init_states)
    target_states = np.atleast_2d(target_states)

    if paired:
        if init_states.shape[0] != target_states.shape[0]:
            logger.warning("The numbers of initial and target states are not equal. The longer one is trimmed")
            num = min(init_states.shape[0], target_states.shape[0])
            init_states = init_states[:num]
            target_states = target_states[:num]
        pairs = [(init_states[i], target_states[i]) for i in range(init_states.shape[0])]
    else:
        pairs = [(pi, pt) for pi in init_states for pt in target_states]
        logger.warning(
            f"A total of {len(pairs)} pairs of initial and target states will be calculated."
            "To reduce the number of LAP calculations, please use the `paired` mode."
        )

    t, prediction, action, exprs, mftp, trajectory = [], [], [], [], [], []
    if min_lap_t:
        i_elbow = []
        laps = []
        opt_T = []
        A = []

    path_ind = 0

    for (init_state, target_state) in pairs:
        if init_paths is None:
            init_path = get_init_path(G, init_state, target_state, coords, interpolation_num=n_points)
        else:
            init_path = init_paths if type(init_paths) == np.ndarray else init_paths[path_ind]

        path_ind += 1

        path_sol, dt_sol, action_opt = least_action_path(
            init_state, target_state, vf.func, vf.get_Jacobian(), n_points=n_points, init_path=init_path, D=D, **kwargs
        )

        n_points = len(path_sol)  # the actual #points due to arclength resampling

        if min_lap_t:
            t_sol = dt_sol * (n_points - 1)
            t_min = 0.3 * t_sol
            i_elbow_, laps_, A_, opt_T_ = minimize_lap_time(
                path_sol, t_sol, t_min, vf.func, vf.get_Jacobian(), D=D, num_t=num_t, elbow_method=elbow_method
            )
            if i_elbow_ is None:
                i_elbow_ = 0
            path_sol = laps_[i_elbow_]
            dt_sol = opt_T_[i_elbow_] / (n_points - 1)

            i_elbow.append(i_elbow_)
            laps.append(laps_)
            A.append(A_)
            opt_T.append(opt_T_)

        traj = LeastActionPath(X=path_sol, vf_func=vf.func, D=D, dt=dt_sol)
        trajectory.append(traj)
        t.append(np.arange(path_sol.shape[0]) * dt_sol)
        prediction.append(path_sol)
        action.append(traj.action())
        mftp.append(traj.mfpt())

        # if basis == "pca":
        #     pc_keys = "PCs" if PCs is None else PCs
        #     if pc_keys not in data.genes_matrix:
        #         logger.warning("Expressions along the trajectories cannot be retrieved, due to lack of `PCs` in .uns.")
        #     else:
        #         if "pca_mean" not in data.tl.result:
        #             pca_mean = 0
        #         else:
        #             pca_mean = data.tl.result["pca_mean"]
        #         exprs.append(pca_to_expr(traj.X, data.genes_matrix["PCs"], pca_mean, func=expr_func))

    if add_key is None:
        LAP_key = "LAP" if basis is None else "LAP_" + basis
    else:
        LAP_key = add_key

    data.tl.result[LAP_key] = {
        "init_states": init_states,
        "init_cells": init_cells,
        "target_states": target_states,
        "target_cells": target_cells,
        "t": t,
        "mftp": mftp,
        "prediction": prediction,
        "action": action,
        # "genes": adata.var_names[adata.var.use_for_pca],
        "exprs": exprs,
        "vf_key": vf_key,
        # 'path_sol': path_sol,
        # 'dt_sol': dt_sol
    }

    if min_lap_t:
        data.tl.result[LAP_key]["min_t"] = {"A": A, "T": opt_T, "i_elbow": i_elbow, "paths": laps, "method": elbow_method}

    return trajectory[0] if len(trajectory) == 1 else trajectory

def pca_to_expr(X, PCs, mean=0, func=None):
    # reverse project from PCA back to raw expression space
    if PCs.shape[1] == X.shape[1]:
        exprs = X @ PCs.T + mean
        if func is not None:
            exprs = func(exprs)
    else:
        raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[1], X.shape[1]))
    return exprs

class Trajectory:
    def __init__(self, X, t=None) -> None:
        """
        Base class for handling trajectory interpolation, resampling, etc.
        """
        self.X = X
        self.t = t

class LeastActionPath(Trajectory):
    def __init__(self, X, vf_func, D=1, dt=1) -> None:
        super().__init__(X, t=np.arange(X.shape[0]) * dt)
        self.func = vf_func
        self.D = D
        self._action = np.zeros(X.shape[0])
        for i in range(1, len(self._action)):
            self._action[i] = action(self.X[: i + 1], self.func, self.D, dt)

    def get_t(self):
        return self.t

    def get_dt(self):
        return np.mean(np.diff(self.t))

    def action(self, t=None, **interp_kwargs):
        if t is None:
            return self._action
        else:
            return interp1d(self.t, self._action, **interp_kwargs)(t)

    def mfpt(self, action=None):
        """Eqn. 7 of Epigenetics as a first exit problem."""
        action = self._action if action is None else action
        return 1 / np.exp(-action)

    def optimize_dt(self):
        dt_0 = self.get_dt()
        t_dict = minimize(lambda t: action(self.X, self.func, D=self.D, dt=t), dt_0)

        dt_sol = t_dict["x"][0]
        self.t = np.arange(self.X.shape[0]) * dt_sol
        return dt_sol

def plot_least_action_path(
        data: StereoExpData,
        basis='spatial',
        ax: Axes = None,
        linewidth=3,
        point_size=6,
        linestyle='solid',
        width=None,
        height=None,
    ):
    """
    Plot the LAP and selected subset of cells.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    basis
        The embedding data used to predict the least action path.
        (Default: 'spatial')
    ax
        Figure axes.
        (Default: None)
    linewidth
        Linewidth of the LAP.
        (Default: 3)
    point_size
        Point size of the LAP.
        (Default: 6)
    linestyle
        Linestyple of the LAP.
        (Default: 'solid')

    Returns
    -----------
    ax
        The plot of the LAP and cells.
    """
    lap_dict=data.tl.result['LAP_'+basis]
    id_array=np.arange(0,len(lap_dict['prediction'][0]),2)
    lap_point_pos = lap_dict['prediction'][0][id_array]
    lap_value = lap_dict['action'][0][id_array]

    minima=np.min(lap_value)
    maxima=np.max(lap_value)
    norm=matplotlib.colors.Normalize(vmin=minima,vmax=maxima,clip=True)
    mapper=cm.ScalarMappable(norm=norm,cmap=plt.get_cmap('hsv'))

    cols=[mapper.to_rgba(v) for v in lap_value]

    ax.plot(*lap_point_pos.T, c="k",linewidth= linewidth,linestyle=linestyle,zorder=3)
    ax.scatter(*lap_point_pos.T, c=cols,s=point_size,zorder=2)
    return ax
