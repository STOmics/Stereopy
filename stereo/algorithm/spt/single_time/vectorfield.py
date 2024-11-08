import anndata as ad
from typing import Union, Dict
import sys
import numpy as np
import numpy.matlib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.linalg.blas import dgemm

from stereo.core.stereo_exp_data import StereoExpData

from ..utils import get_cell_coordinates


def update_n_merge_dict(dict1, dict2):
    dict = {
        **dict1,
        **dict2,
    }

    return dict


def norm(X, V, T, fix_velocity=True):
    Y = X + V
    n, m = X.shape[0], V.shape[0]

    xm = np.mean(X, 0)
    ym = np.mean(Y, 0)

    x, y, t = (
        X - xm[None, :],
        Y - ym[None, :],
        T - (1 / 2 * (xm[None, :] + ym[None, :])) if T is not None else None,
    )

    xscale, yscale = (
        np.sqrt(np.sum(np.sum(x**2, 1)) / n),
        np.sqrt(np.sum(np.sum(y**2, 1)) / m),
    )

    X, Y, T = (
        x / xscale,
        y / yscale,
        t / (1 / 2 * (xscale + yscale)) if T is not None else None,
    )

    X, V, T = X, V if fix_velocity else Y - X, T
    norm_dict = {
        "xm": xm,
        "ym": ym,
        "xscale": xscale,
        "yscale": yscale,
        "fix_velocity": fix_velocity,
    }

    return X, V, T, norm_dict


def sample_by_velocity(V, n, seed=19491001):
    np.random.seed(seed)
    tmp_V = np.linalg.norm(V, axis=1)
    p = tmp_V / np.sum(tmp_V)
    idx = np.random.choice(np.arange(len(V)), size=n, p=p, replace=False)
    return idx


def bandwidth_selector(X):
    """
    This function computes an empirical bandwidth for a Gaussian kernel.
    """
    n, m = X.shape
    if n > 200000 and m > 2:
        from pynndescent import NNDescent

        nbrs = NNDescent(
            X,
            metric="euclidean",
            n_neighbors=max(2, int(0.2 * n)),
            n_jobs=-1,
            random_state=19491001,
        )
        _, distances = nbrs.query(X, k=max(2, int(0.2 * n)))
    else:
        alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
        nbrs = NearestNeighbors(
            n_neighbors=max(2, int(0.2 * n)), algorithm=alg, n_jobs=-1
        ).fit(X)
        distances, _ = nbrs.kneighbors(X)

    d = np.mean(distances[:, 1:]) / 1.5
    return np.sqrt(2) * d


def con_K(x, y, beta, method="cdist", return_d=False):
    if method == "cdist" and not return_d:
        K = cdist(x, y, "sqeuclidean")
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]

        D = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(
            np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0]
        )
        K = np.squeeze(np.sum(D**2, 1))
    K = -beta * K
    K = np.exp(K)

    if return_d:
        return K, D
    else:
        return K


def con_K_div_cur_free(x, y, sigma=0.8, eta=0.5):
    m, d = x.shape
    n, d = y.shape
    sigma2 = sigma**2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(
        np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0]
    )
    G_tmp = np.squeeze(np.sum(G_tmp**2, 1))
    G_tmp3 = -G_tmp / sigma2
    G_tmp = -G_tmp / (2 * sigma2)
    G_tmp = np.exp(G_tmp) / sigma2
    G_tmp = np.kron(G_tmp, np.ones((d, d)))

    x_tmp = np.matlib.tile(x, [n, 1])
    y_tmp = np.matlib.tile(y, [1, m]).T
    y_tmp = y_tmp.reshape((d, m * n), order="F").T
    xminusy = x_tmp - y_tmp
    G_tmp2 = np.zeros((d * m, d * n))

    tmp4_ = np.zeros((d, d))
    for i in tqdm(range(d), desc="Iterating each dimension in con_K_div_cur_free:"):
        for j in np.arange(i, d):
            tmp1 = xminusy[:, i].reshape((m, n), order="F")
            tmp2 = xminusy[:, j].reshape((m, n), order="F")
            tmp3 = tmp1 * tmp2
            tmp4 = tmp4_.copy()
            tmp4[i, j] = 1
            tmp4[j, i] = 1
            G_tmp2 = G_tmp2 + np.kron(tmp3, tmp4)

    G_tmp2 = G_tmp2 / sigma2
    G_tmp3 = np.kron((G_tmp3 + d - 1), np.eye(d))
    G_tmp4 = np.kron(np.ones((m, n)), np.eye(d)) - G_tmp2
    df_kernel, cf_kernel = (1 - eta) * G_tmp * (G_tmp2 + G_tmp3), eta * G_tmp * G_tmp4
    G = df_kernel + cf_kernel

    return G, df_kernel, cf_kernel


def get_P(Y, V, sigma2, gamma, a, div_cur_free_kernels=False):
    if div_cur_free_kernels:
        Y = Y.reshape((2, int(Y.shape[0] / 2)), order="F").T
        V = V.reshape((2, int(V.shape[0] / 2)), order="F").T

    D = Y.shape[1]
    temp1 = np.exp(-np.sum((Y - V) ** 2, 1) / (2 * sigma2))
    temp2 = (2 * np.pi * sigma2) ** (D / 2) * (1 - gamma) / (gamma * a)
    temp1[temp1 == 0] = np.min(temp1[temp1 != 0])
    P = temp1 / (temp1 + temp2)
    E = (
        P.T.dot(np.sum((Y - V) ** 2, 1)) / (2 * sigma2)
        + np.sum(P) * np.log(sigma2) * D / 2
    )

    return (P[:, None], E) if P.ndim == 1 else (P, E)


def lstsq_solver(lhs, rhs, method="drouin"):
    C = linear_least_squares(lhs, rhs)
    return C


def linear_least_squares(a, b, residuals=False):
    a = np.asarray(a, order="c")
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b))

    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x


def SparseVFC(
    X: np.ndarray,
    Y: np.ndarray,
    Grid: np.ndarray,
    M: int = 100,
    a: float = 5,
    beta: float = None,
    ecr: float = 1e-5,
    gamma: float = 0.9,
    lambda_: float = 3,
    minP: float = 1e-5,
    MaxIter: int = 500,
    theta: float = 0.75,
    div_cur_free_kernels: bool = False,
    velocity_based_sampling: bool = True,
    sigma: float = 0.8,
    eta: float = 0.5,
    seed=0,
    lstsq_method: str = "drouin",
    verbose: int = 1,
) -> dict:
    need_utility_time_measure = verbose > 1
    X_ori, Y_ori = X.copy(), Y.copy()
    valid_ind = np.where(np.isfinite(Y.sum(1)))[0]
    X, Y = X[valid_ind], Y[valid_ind]
    N, D = Y.shape
    grid_U = None

    # Construct kernel matrix K
    tmp_X, uid = np.unique(X, axis=0, return_index=True)  # return unique rows
    M = min(M, tmp_X.shape[0])
    if velocity_based_sampling:
        idx = sample_by_velocity(Y[uid], M, seed=seed)
    else:
        idx = np.random.RandomState(seed=seed).permutation(
            tmp_X.shape[0]
        )  # rand select some initial points
        idx = idx[range(M)]
    ctrl_pts = tmp_X[idx, :]

    if beta is None:
        h = bandwidth_selector(ctrl_pts)
        beta = 1 / h**2

    K = (
        con_K(ctrl_pts, ctrl_pts, beta)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(ctrl_pts, ctrl_pts, sigma, eta)[0]
    )
    U = (
        con_K(X, ctrl_pts, beta)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(X, ctrl_pts, sigma, eta)[0]
    )
    if Grid is not None:
        grid_U = (
            con_K(Grid, ctrl_pts, beta)
            if div_cur_free_kernels is False
            else con_K_div_cur_free(Grid, ctrl_pts, sigma, eta)[0]
        )
    M = ctrl_pts.shape[0] * D if div_cur_free_kernels else ctrl_pts.shape[0]

    if div_cur_free_kernels:
        X = X.flatten()[:, None]
        Y = Y.flatten()[:, None]

    # Initialization
    V = X.copy() if div_cur_free_kernels else np.zeros((N, D))
    C = np.zeros((M, 1)) if div_cur_free_kernels else np.zeros((M, D))
    i, tecr, E = 0, 1, 1
    # test this
    sigma2 = (
        sum(sum((Y - X) ** 2)) / (N * D)
        if div_cur_free_kernels
        else sum(sum((Y - V) ** 2)) / (N * D)
    )
    sigma2 = 1e-7 if sigma2 < 1e-8 else sigma2
    tecr_vec = np.ones(MaxIter) * np.nan
    E_vec = np.ones(MaxIter) * np.nan
    P = None
    while i < MaxIter and tecr > ecr and sigma2 > 1e-8:
        # E_step
        E_old = E
        P, E = get_P(Y, V, sigma2, gamma, a, div_cur_free_kernels)

        E = E + lambda_ / 2 * np.trace(C.T.dot(K).dot(C))
        E_vec[i] = E
        tecr = abs((E - E_old) / E)
        tecr_vec[i] = tecr

        P = np.maximum(P, minP)
        if div_cur_free_kernels:
            P = np.kron(
                P, np.ones((int(U.shape[0] / P.shape[0]), 1))
            )  # np.kron(P, np.ones((D, 1)))
            lhs = (U.T * np.matlib.tile(P.T, [M, 1])).dot(U) + lambda_ * sigma2 * K
            rhs = (U.T * np.matlib.tile(P.T, [M, 1])).dot(Y)
        else:
            UP = U.T * numpy.matlib.repmat(P.T, M, 1)
            lhs = UP.dot(U) + lambda_ * sigma2 * K
            rhs = UP.dot(Y)

        C = lstsq_solver(lhs, rhs, method=lstsq_method)

        # Update V and sigma**2
        V = U.dot(C)
        Sp = sum(P) / 2 if div_cur_free_kernels else sum(P)
        sigma2 = (sum(P.T.dot(np.sum((Y - V) ** 2, 1))) / np.dot(Sp, D))[0]

        # Update gamma
        numcorr = len(np.where(P > theta)[0])
        gamma = numcorr / X.shape[0]

        if gamma > 0.95:
            gamma = 0.95
        elif gamma < 0.05:
            gamma = 0.05

        i += 1
    if i == 0 and not (tecr > ecr and sigma2 > 1e-8):
        raise Exception(
            "please check your input parameters, "
            f"tecr: {tecr}, ecr {ecr} and sigma2 {sigma2},"
            f"tecr must larger than ecr and sigma2 must larger than 1e-8"
        )

    grid_V = None
    if Grid is not None:
        grid_V = np.dot(grid_U, C)

    VecFld = {
        "X": X_ori,
        "valid_ind": valid_ind,
        "X_ctrl": ctrl_pts,
        "ctrl_idx": idx,
        "Y": Y_ori,
        "beta": beta,
        "V": V.reshape((N, D)) if div_cur_free_kernels else V,
        "C": C,
        "P": P,
        "VFCIndex": np.where(P > theta)[0],
        "sigma2": sigma2,
        "grid": Grid,
        "grid_V": grid_V,
        "iteration": i - 1,
        "tecr_traj": tecr_vec[:i],
        "E_traj": E_vec[:i],
    }

    return VecFld


def get_vf_dict(data: StereoExpData, basis="", vf_key="VecFld"):
    if basis is not None and len(basis) > 0:
        vf_key = "%s_%s" % (vf_key, basis)

    if vf_key not in data.tl.result:
        raise ValueError(
            f"Vector field function {vf_key} is not included in the adata object! "
            f"Try firstly running dyn.vf.VectorField(adata, basis='{basis}')"
        )

    vf_dict = data.tl.result[vf_key]
    return vf_dict


def vecfld_from_adata(data: StereoExpData, basis="", vf_key="VecFld"):
    vf_dict = get_vf_dict(data, basis=basis, vf_key=vf_key)

    method = vf_dict["method"]
    func = lambda x: vector_field_function(x, vf_dict)

    return vf_dict, func


class BaseVectorField:
    def __init__(
        self,
        X=None,
        V=None,
        Grid=None,
        *args,
        **kwargs,
    ):
        self.data = {"X": X, "V": V, "Grid": Grid}
        self.vf_dict = kwargs.pop("vf_dict", {})
        self.func = kwargs.pop("func", None)
        self.fixed_points = kwargs.pop("fixed_points", None)
        super().__init__(**kwargs)

    def from_adata(self, adata, basis="", vf_key="VecFld"):
        vf_dict, func = vecfld_from_adata(adata, basis=basis, vf_key=vf_key)
        self.data["X"] = vf_dict["X"]
        self.data["V"] = vf_dict["Y"]  # use the raw velocity
        self.vf_dict = vf_dict
        self.func = func


class DifferentiableVectorField(BaseVectorField):
    def get_Jacobian(self, method=None):
        # subclasses must implement this function.
        pass


class SvcVectorField(DifferentiableVectorField):
    def __init__(self, X=None, V=None, Grid=None, *args, **kwargs):
        super().__init__(X, V, Grid)
        if X is not None and V is not None:
            self.parameters = kwargs
            self.parameters = update_n_merge_dict(
                self.parameters,
                {
                    "M": kwargs.pop("M", None)
                    or max(min([50, len(X)]), int(0.05 * len(X)) + 1),
                    # min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100)))),
                    "a": kwargs.pop("a", 5),
                    "beta": kwargs.pop("beta", None),
                    "ecr": kwargs.pop("ecr", 1e-5),
                    "gamma": kwargs.pop("gamma", 0.9),
                    "lambda_": kwargs.pop("lambda_", 3),
                    "minP": kwargs.pop("minP", 1e-5),
                    "MaxIter": kwargs.pop("MaxIter", 500),
                    "theta": kwargs.pop("theta", 0.75),
                    "div_cur_free_kernels": kwargs.pop("div_cur_free_kernels", False),
                    "velocity_based_sampling": kwargs.pop(
                        "velocity_based_sampling", True
                    ),
                    "sigma": kwargs.pop("sigma", 0.8),
                    "eta": kwargs.pop("eta", 0.5),
                    "seed": kwargs.pop("seed", 0),
                },
            )
        self.norm_dict = {}

    def train(self, normalize=False, **kwargs):
        if normalize:
            X_norm, V_norm, T_norm, norm_dict = norm(
                self.data["X"], self.data["V"], self.data["Grid"]
            )
            (self.data["X"], self.data["V"], self.data["Grid"], self.norm_dict,) = (
                X_norm,
                V_norm,
                T_norm,
                norm_dict,
            )

        verbose = kwargs.pop("verbose", 0)
        lstsq_method = kwargs.pop("lstsq_method", "drouin")
        VecFld = SparseVFC(
            self.data["X"],
            self.data["V"],
            self.data["Grid"],
            **self.parameters,
            verbose=verbose,
            lstsq_method=lstsq_method,
        )
        self.parameters = update_dict(self.parameters, VecFld)

        self.vf_dict = VecFld

        self.func = lambda x: vector_field_function(x, VecFld)
        self.vf_dict["V"] = self.func(self.data["X"])
        self.vf_dict["normalize"] = normalize

        return self.vf_dict

    def get_Jacobian(
        self, method="analytical", input_vector_convention="row", **kwargs
    ):
        return lambda x: Jacobian_rkhs_gaussian(x, self.vf_dict, **kwargs)


def Jacobian_rkhs_gaussian(x, vf_dict, vectorize=False):
    if x.ndim == 1:
        K, D = con_K(x[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        J = (vf_dict["C"].T * K) @ D[0].T
    elif not vectorize:
        n, d = x.shape
        J = np.zeros((d, d, n))
        for i, xi in enumerate(x):
            K, D = con_K(xi[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
            J[:, :, i] = (vf_dict["C"].T * K) @ D[0].T
    else:
        K, D = con_K(x, vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        if K.ndim == 1:
            K = K[None, :]
        J = np.einsum("nm, mi, njm -> ijn", K, vf_dict["C"], D)

    return -2 * vf_dict["beta"] * J


def vector_field_function(
    x, vf_dict, dim=None, kernel="full", X_ctrl_ind=None, **kernel_kwargs
):
    """vector field function constructed by sparseVFC.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
    """
    # x=np.array(x).reshape((1, -1))
    if "div_cur_free_kernels" in vf_dict.keys():
        has_div_cur_free_kernels = True
    else:
        has_div_cur_free_kernels = False

    # x = np.array(x)
    if x.ndim == 1:
        x = x[None, :]

    if has_div_cur_free_kernels:
        if kernel == "full":
            kernel_ind = 0
        elif kernel == "df_kernel":
            kernel_ind = 1
        elif kernel == "cf_kernel":
            kernel_ind = 2
        else:
            raise ValueError(
                f"the kernel can only be one of {'full', 'df_kernel', 'cf_kernel'}!"
            )

        K = con_K_div_cur_free(
            x,
            vf_dict["X_ctrl"],
            vf_dict["sigma"],
            vf_dict["eta"],
            **kernel_kwargs,
        )[kernel_ind]
    else:
        Xc = vf_dict["X_ctrl"]
        K = con_K(x, Xc, vf_dict["beta"], **kernel_kwargs)

    if X_ctrl_ind is not None:
        C = np.zeros_like(vf_dict["C"])
        C[X_ctrl_ind, :] = vf_dict["C"][X_ctrl_ind, :]
    else:
        C = vf_dict["C"]

    K = K.dot(C)

    if dim is not None and not has_div_cur_free_kernels:
        if np.isscalar(dim):
            K = K[:, :dim]
        elif dim is not None:
            K = K[:, dim]

    return K


def VectorField(
    data: StereoExpData,
    basis: Union[None, str] = 'spatial',
    normalize: bool = False,
    result_key: Union[str, None] = None,
    **kwargs,
):
    """
    Learn the function of vector filed.

    Parameters
    ----------
    data
        An object of StereoExpData.
    basis
        The label of cell coordinates, for example, `umap` or `spatial`.
        (Default: None)
    normalize
        Logic flag to determine whether to normalize the data to have zero means and unit covariance.
        (Default: False)

    Returns
    -----------
    BaseVectorfield
        A vector field class object.
    """
    method = "SparseVFC"
    result_key = None

    X = get_cell_coordinates(data, basis=basis, ndmin=2)
    V = data.tl.result[f"velocity_{basis}"]
    if isinstance(V, pd.DataFrame):
        V = V.to_numpy()

    Grid = None

    if X is None:
        raise Exception(
            f"X is None. Make sure you passed the correct X or {basis} dimension reduction method."
        )
    elif V is None:
        raise Exception("V is None. Make sure you passed the correct V.")

    if method.lower() == "sparsevfc":
        vf_kwargs = {
            "M": None,
            "a": 5,
            "beta": None,
            "ecr": 1e-5,
            "gamma": 0.9,
            "lambda_": 3,
            "minP": 1e-5,
            "MaxIter": 30,
            "theta": 0.75,
            "div_cur_free_kernels": False,
            "velocity_based_sampling": True,
            "sigma": 0.8,
            "eta": 0.5,
            "seed": 0,
        }
    else:
        raise ValueError("only support SparseVFC")

    vf_kwargs = update_dict(vf_kwargs, kwargs)

    if method.lower() == "sparsevfc":
        VecFld = SvcVectorField(X, V, Grid, **vf_kwargs)
        vf_dict: Dict[str, np.ndarray] = VecFld.train(normalize=normalize, **kwargs)

    if result_key is None:
        vf_key = "VecFld" if basis is None else "VecFld_" + basis
    else:
        vf_key = result_key if basis is None else result_key + "_" + basis

    vf_dict["method"] = method
    if basis is not None:
        key = "velocity_" + basis + "_" + method
        X_copy_key = "X_" + basis + "_" + method

        # data.tl.result[key] = vf_dict["V"]
        # data.tl.result[X_copy_key] = vf_dict["X"]
        data.tl.result.set_value(key, vf_dict["V"])
        data.tl.result.set_value(X_copy_key, vf_dict["X"])
        data.tl.result.set_value(vf_key, vf_dict)

    control_point, inlier_prob, valid_ids = (
        "control_point_" + basis if basis is not None else "control_point",
        "inlier_prob_" + basis if basis is not None else "inlier_prob",
        vf_dict["valid_ind"],
    )
    if method.lower() == "sparsevfc":
        data.cells[control_point], data.cells[inlier_prob] = False, np.nan
        data.cells.loc[data.cell_names[vf_dict["ctrl_idx"]], control_point] = True
        data.cells.loc[data.cell_names[valid_ids], inlier_prob] = vf_dict["P"].flatten()

    # angles between observed velocity and that predicted by vector field across cells:
    cell_angles = np.zeros(data.n_cells, dtype=float)
    for i, u, v in zip(valid_ids, V[valid_ids], vf_dict["V"]):
        # fix the u, v norm == 0 in angle function
        cell_angles[i] = angle(u.astype("float64"), v.astype("float64"))

    if basis is not None:
        temp_key = "obs_vf_angle_" + basis
        data.cells[temp_key] = cell_angles

    return VecFld


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())

    return dict1


def angle(vector1, vector2):
    """Returns the angle in radians between given vectors"""
    v1_norm, v1_u = unit_vector(vector1)
    v2_norm, v2_u = unit_vector(vector2)

    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    else:
        minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        return sign * np.arccos(dot_p)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    vec_norm = np.linalg.norm(vector)
    if vec_norm == 0:
        return vec_norm, vector
    else:
        return vec_norm, vector / vec_norm
