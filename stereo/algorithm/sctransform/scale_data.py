import numba
import numpy as np
import pandas as pd


@numba.jit(cache=True, forceobj=True)
def fast_row_scale(mat, center=True, scale=True, scale_max=10):
    # TODO: ignore `scale`
    rm = None
    if center:
        # the seurat SCT is `float32`
        rm = np.mat(mat.mean(1), dtype=np.float32).T
    if center:
        mat = mat - rm
    if scale_max != np.Inf:
        mat[mat > scale_max] = scale_max
    return mat


@numba.jit(cache=True, forceobj=True)
def fast_row_scale_sparse(mat, center=True, scale=True, scale_max=10):
    # TODO: ignore `scale`
    rm = None
    if center:
        # TODO: `float32` the seurat SCT result
        rm = np.mat(mat.mean(1), dtype=np.float32)
    if center:
        mat = mat - rm
    if scale_max != np.Inf:
        mat[mat > scale_max] = scale_max
    return mat


def ScaleData(
        scale_data: pd.DataFrame,
        features=None,
        vars_to_regress=None,
        latent_data=None,
        split_by=None,
        model_use="linear",
        use_umi=False,
        do_scale=True,
        do_center=True,
        scale_max=10,
        block_size=1000,
        min_cells_to_block=3000,
        verbose=True,
        **kwargs):
    # TODO: `ScaleData` only support one way to run, will fix in the future
    if features is not None:
        features = features
        scale_data = scale_data.loc[features]
    else:
        features = scale_data.index.values

    if vars_to_regress:
        raise NotImplementedError
    max_block = int(np.ceil(len(features) / block_size))
    scaled_data = pd.DataFrame(np.zeros(shape=scale_data.shape, dtype=np.double), index=scale_data.index.values,
                               columns=scale_data.columns.values)
    for i in range(1, max_block + 1):
        my_inds = np.array(range(block_size * (i - 1), block_size * i))
        my_inds = my_inds[my_inds < len(features)]
        arg_list = {
            "mat": scale_data.loc[features[my_inds]],
            "scale": do_scale,
            "center": do_center,
            "scale_max": scale_max,
        }
        scaled_data.loc[features[my_inds]] = fast_row_scale(**arg_list)
    return scaled_data
