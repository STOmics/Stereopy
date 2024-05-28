import scipy
import numpy as np
import anndata
import pandas as pd

from scipy import sparse


def _norm_sum_log(arr, min_val=0, max_val=10000):
    """

    :param arr: np.ndarray of 2d
    :param min_val:
    :param max_val:
    :return:
    """
    summed_gene = arr.sum(axis=1) + 0.01

    # if len(summed_gene.shape) == 1:
    arr = arr / np.repeat(np.expand_dims(summed_gene, axis=1), arr.shape[1], axis=1) * (max_val - min_val) + min_val  # minmax normalize
    # else:
    #     arr = arr / np.repeat(summed_gene, arr.shape[1], axis=1) * (max_val - min_val) + min_val  # minmax normalize
    return np.log10(arr + 1)


def binning(adata, ctype_col, pixsize):
    """
    Binning process of granularity adjusting, which grids the spatial range of a section by assigned step size, then sum
    up gene expression to a representative spot of each grid, and label the spot with most frequent annotation type or cluster.

    Previous spatial profile, and new location stored adata.obsm['spatial']

    :param adata: AnnData.anndata
    :param ctype_col: cluster or annotation type stored in adata.obs[ctype_col]
    :param pixsize: step size. each grid has size of pixsize x pixsize
    :return: adata
    """

    # 1 get data from shrinked adata
    exp = adata.X.todense()

    if adata.obsm['spatial'].shape[1] >= 3:
        spa_z = adata.obsm['spatial'][:, 2]
    else:
        spa_z = np.ones(shape=(adata.n_obs,)) * 1

    spa_xy = adata.obsm['spatial'][:, :2]
    ctype = adata.obs[ctype_col]

    # 2 make cell belonging to same pixel share the same coordinate
    offset_xy = spa_xy.min(axis=0)  # (2,)
    shift_xy = spa_xy - np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # starts from 0, and then increments at 50
    shift_xy = shift_xy / pixsize  # 1 equals to input bin size
    new_bin_xy = np.floor(shift_xy) * pixsize + np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # increments at pixsize

    # 3 utilize Pandas to aggregate cells from same pixel together
    df = pd.DataFrame(np.concatenate((exp,
                                      new_bin_xy,
                                      np.expand_dims(spa_z, axis=1),
                                      np.expand_dims(ctype, axis=1)), axis=1))  # each line still represent one cell
    col_X_name_li = df.columns.tolist()[:-4]
    col_x_name, col_y_name, col_z_name = df.columns.tolist()[-4:-1]
    col_ty_name = df.columns.tolist()[-1]
    df_agg_spa = df.groupby(by=[col_x_name, col_y_name])[col_x_name, col_y_name, col_z_name].agg('first').reset_index(drop=True)
    df_agg_exp = df.groupby(by=[col_x_name, col_y_name])[col_X_name_li].agg('sum').reset_index(drop=True)
    df_agg_ty = df.groupby(by=[col_x_name, col_y_name])[col_ty_name].agg(lambda x: x.value_counts().index[0]).to_frame().reset_index(drop=True)
    exp_norm = _norm_sum_log(df_agg_exp.to_numpy())  # sum to unity and perform log

    # 4. construct an adata
    slice_srk = anndata.AnnData(sparse.csr_matrix(exp_norm))  # shrink
    slice_srk.obsm['spatial'] = df_agg_spa.to_numpy()
    slice_srk.obs[ctype_col] = df_agg_ty.to_numpy()

    return slice_srk


def interpolate(com_slicesl, reuse_slicesl):
    """
    The interpolation step in granularity adjusting, where the original resolution data is interpolated into the
    pre-registered and registered coarse dataset, leading to registration result in original resolution

    :param com_slicesl: list of AnnData.anndata which stores information of coarse data on which ST-GEARS is implemented on
    :param reuse_slicesl: list of AnnData.anndata which stores information of original data on which ST-GEARS has not been implemented on
    :return: reuse_slicesl, registration result by interpolatioj is stored on adata.obsm['spatial_elas_reuse']
    """
    for i in range(len(reuse_slicesl)):
        slice_re = reuse_slicesl[i]
        slice_com = com_slicesl[i]
        dx_interp = scipy.interpolate.griddata(slice_com.obsm['spatial'][:, :2],
                                               slice_com.obsm['spatial_elas'][:, 0] - slice_com.obsm['spatial'][:, 0],
                                               slice_re.obsm['spatial'][:, :2],
                                               method='nearest')
        x_interp = slice_re.obsm['spatial'][:, 0] + dx_interp

        dy_interp = scipy.interpolate.griddata(slice_com.obsm['spatial'][:, :2],
                                               slice_com.obsm['spatial_elas'][:, 1] - slice_com.obsm['spatial'][:, 1],
                                               slice_re.obsm['spatial'][:, :2],
                                               method='nearest')
        y_interp = slice_re.obsm['spatial'][:, 1] + dy_interp

        xy_interp = np.concatenate([np.expand_dims(x_interp, axis=1), np.expand_dims(y_interp, axis=1)], axis=1)
        xy_interp_with_z = np.concatenate([xy_interp, np.expand_dims(slice_re.obsm['spatial'][:, 2], axis=1)], axis=1)
        slice_re.obsm['spatial_elas_reuse'] = xy_interp_with_z

        reuse_slicesl[i] = slice_re
    return reuse_slicesl