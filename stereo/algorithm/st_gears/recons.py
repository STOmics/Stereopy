import numpy as np
import gc
import scipy
import pandas as pd

from typing import List, Tuple
from anndata import AnnData

from stereo.log_manager import logger

from .helper import intersect, filter_pi_mtx, to_dense_array
# from . import tps


def filter_rows_cols_to_spa(sliceA, sliceB, filter_by_label, label_col, spaAtype, spaBtype):
    """
    filter both genes and spot cell-types that are not on either one of the two slices, then output the remained spots'
    spatial information
    """

    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    if filter_by_label:
        common_ctype = intersect(set(sliceA.obs[label_col].tolist()), set(sliceB.obs[label_col].tolist()))
        sliceA = sliceA[sliceA.obs[label_col].isin(common_ctype)]
        sliceB = sliceB[sliceB.obs[label_col].isin(common_ctype)]
    else:
        pass

    spaA = sliceA.obsm[spaAtype].copy()
    spaB = sliceB.obsm[spaBtype].copy()
    return spaA, spaB


def generalized_procrustes_analysis(spaA, spaB, pi):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).

    Args:
        spaA: filtered (spot, coor)
        spaB: filtered
        pi: mapping between the two layers (spots with indentical label types filtered) output by PASTE
        output_params:

    Returns:
        tA: center of filtered sliceA -- arr: (avg_x, avg_y)
        tB: center of filtered sliceB
        R: After transforming (filtered or unfiltered) sliceA and (filtered or unfiltered) sliceB respectively to their own center,
           use R as rotational matix (Y = R.dot(Y.T).T) to reach an optimum Euclidean tranformation
    """

    assert spaA.shape[1] in [2, 3] and spaB.shape[1] in [2, 3]  # (index, coor)

    tA = np.mean(spaA, axis=0)  # arr: (avg_x, avg_y)
    tB = np.mean(spaB, axis=0)

    # 移动到中心
    spaA = spaA - np.broadcast_to(tA, spaA.shape)
    spaB = spaB - np.broadcast_to(tB, spaB.shape)

    H = spaB.T.dot(pi.T.dot(spaA))

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T.dot(U.T)  # 2*2 rotational matrix

    return tA, tB, R


def stack_slices_pairwise_rigid(
        slicesl: List[AnnData],
        pis: List[np.ndarray],
        label_col: str,
        fil_pc = 20,
        filter_by_label: bool = True,
        spatial_key: str = 'spatial'
    ) -> Tuple[List[AnnData]]:
    """
    Stack slices by Proscrutes Analysis, in accordance to the order of slices.
    Align 2nd to 1st slice, then 3rd to the transformed 2nd slice, etc.

    Args:
        slicesl: list of raw slices
        pis: list of pi(s)
        label_col: column name of adata.obs that stores annotated celltypes
        fil_pc: percentage of ranked probabilities in transition matrix, after which probabilities will be filtered.
        filter_by_label: if spots were filtered acordding to annotation, in the ot solving process, or not

    Returns:
        rigid registration result saved in adata.obsm['spatial_rigid']

    """
    assert type(slicesl) == list, "Type of slicesl is not list"
    assert len(slicesl) >= 2, "Length of slicesl less than 2"

    assert type(pis) == list, "Type of pis is not list"
    assert len(pis) == len(slicesl) - 1, "'slices' should have length one more than 'pis'. Please double check."

    assert type(filter_by_label) == bool, "Type of filter_by_label is not bool"

    spaA, spaB = filter_rows_cols_to_spa(slicesl[0], slicesl[1], filter_by_label, label_col, spaAtype=spatial_key, spaBtype=spatial_key)

    # the first and the second slice
    try:
        pi_rigid_0 = filter_pi_mtx(to_dense_array(pis[0]), percentage=fil_pc, same_mem=False)
        tA, tB, R = generalized_procrustes_analysis(spaA, spaB, pi_rigid_0)  # output: tA: (2,), R: (2,2)
        del pi_rigid_0
        del spaA
        del spaB
        gc.collect()

        slicesl[0].obsm['spatial_rigid'] = slicesl[0].obsm[spatial_key] - np.broadcast_to(tA, slicesl[0].obsm[spatial_key].shape)
        slicesl[1].obsm['spatial_rigid'] = slicesl[1].obsm[spatial_key] - np.broadcast_to(tB, slicesl[1].obsm[spatial_key].shape)
        slicesl[1].obsm['spatial_rigid'] = R.dot(slicesl[1].obsm['spatial_rigid'].T).T
        gc.collect()
    except:  # in case that pi matrices doesn't work
        tA = np.mean(spaA, axis=0)
        tB = np.mean(spaB, axis=0)
        del spaA
        del spaB
        gc.collect()

        slicesl[0].obsm['spatial_rigid'] = slicesl[0].obsm[spatial_key] - np.broadcast_to(tA, slicesl[0].obsm[spatial_key].shape)
        slicesl[1].obsm['spatial_rigid'] = slicesl[1].obsm[spatial_key] - np.broadcast_to(tB, slicesl[1].obsm[spatial_key].shape)
        gc.collect()

    # start from the second slice
    for i in range(1, len(slicesl) - 1):
        # Tentatively take the filtered spots have the same spatial center as all spots before filtering
        spaA, spaB = filter_rows_cols_to_spa(slicesl[i], slicesl[i+1], filter_by_label, label_col, spaAtype='spatial_rigid', spaBtype=spatial_key)

        try:
            pi_rigid_i = filter_pi_mtx(to_dense_array(pis[i]), percentage=fil_pc, same_mem=False)
            tA, tB, R = generalized_procrustes_analysis(spaA, spaB, pi_rigid_i)
            del pi_rigid_i
            del spaA
            del spaB
            gc.collect()

            slicesl[i+1].obsm['spatial_rigid'] = slicesl[i+1].obsm[spatial_key] - np.broadcast_to(tB, slicesl[i+1].obsm[spatial_key].shape)
            slicesl[i+1].obsm['spatial_rigid'] = R.dot(slicesl[i+1].obsm['spatial_rigid'].T).T
            gc.collect()
        except:  # in case that pi matrices doesn't work
            tB = np.mean(spaB, axis=0)
            del spaA
            del spaB
            gc.collect()
            slicesl[i + 1].obsm['spatial_rigid'] = slicesl[i + 1].obsm[spatial_key] - np.broadcast_to(tB, slicesl[i + 1].obsm[spatial_key].shape)
            gc.collect()

    for i, slice in enumerate(slicesl):
        slice.obsm['spatial_rigid'][:, 2] = slice.obsm[spatial_key][:, 2]

    return slicesl


def cal_fil_offset_by_pi_to_pre(pi, spaA, spaB):
    # find the spot index with anchors from sliceB
    max_prob_val = np.max(pi, axis=0)
    indexBwithAnchor = np.where(max_prob_val > 0)
    indexBwithAnchor = indexBwithAnchor[0]

    # find out the pointed spots on sliceA from sliceB anchors, and their coordinates
    pi_BwithAnchor = pi[:, indexBwithAnchor]
    anchAindex = np.argmax(pi_BwithAnchor, axis=0)

    spaAraw_fil = spaA[anchAindex, :][:, :2]
    spaBraw_fil = spaB[indexBwithAnchor, :][:, :2]

    offset_xy = (spaA[anchAindex, :2] - spaB[indexBwithAnchor, :2])[:, :2]  # (n, 2)

    return spaAraw_fil, spaBraw_fil, offset_xy


def cal_fil_offset_by_pi_to_next(pi, spaA, spaB):
    # find the spot index with anchors from sliceA
    max_prob_val = np.max(pi, axis=1)
    indexAwithAnchor = np.where(max_prob_val > 0)
    indexAwithAnchor = indexAwithAnchor[0]

    # find out the pointed spots on sliceB from sliceA anchors, and their coordinates
    pi_AwithAnchor = pi[indexAwithAnchor, :]
    anchBindex = np.argmax(pi_AwithAnchor, axis=1)

    spaAraw_fil = spaA[indexAwithAnchor, :][:, :2]
    spaBraw_fil = spaB[anchBindex, :][:, :2]

    offset_xy = (spaB[anchBindex, :2] - spaA[indexAwithAnchor, :2])[:, :2]  # (n, 2)
    return spaAraw_fil, spaBraw_fil, offset_xy


def stack_slices_pairwise_elas(
        slicesl: List[AnnData],
        pis: List[np.ndarray],
        label_col: str,
        fil_pc: float = 20,
        filter_by_label: bool = True,
        warp_type: str='linear',
        lambda_val: float = None) -> Tuple[List[AnnData]]:

    """
    Register slices in elastic way, in accordance to the order of slices.

    Each slice is elastically registered according to its transition matrix with previous and with next slice; while the
    first and last slice are registered only according to its transition matrix with its adjacent slice.

    Args:
        slicesl: list of rigid registered slices
        pis: list of probabilistic transition matrix
        label_col: column name of adata.obs that stores annotated celltypes
        fil_pc: percentage of ranked probabilities in transition matrix, after which probabilities will be filtered.
        filter_by_label: if spots were filtered acordding to annotation, in the ot solving process, or not
        warp_type: 'linear', 'cubic', 'tps'
        lambda_val: regularization parameter if performing tps warping
    Returns:
        elastic registration result saved in adata.obsm['spatial_elas']

    """

    def cal_strict_fit(spa_raw_fil, spa_new_fil, spa_raw_unfil, warp_type):
        """
        the interpolation is based on bspline, and the extrapolation is based on tps
        """
        # interpolation
        spa_new_unfil_x = scipy.interpolate.griddata(spa_raw_fil, spa_new_fil[:, 0], spa_raw_unfil[:, :2],
                                                     method=warp_type)  # (n,)
        spa_new_unfil_y = scipy.interpolate.griddata(spa_raw_fil, spa_new_fil[:, 1], spa_raw_unfil[:, :2],
                                                     method=warp_type)  # (n,)

        # extrapolation. 'lstsq' is applied for solver to cover processing of singular value matrix
        tps_trans = tps.TPS(spa_raw_fil, spa_new_fil, lambda_=0, solver='lstsq')
        tps_interp_re = tps_trans(spa_raw_unfil[:, :2][np.where(np.isnan(spa_new_unfil_x))[0],
                                  :])  # tps_interp_result (n, 2) 指外插才能计算出的点，在tps插值下的结果
        spa_new_unfil_x[np.where(np.isnan(spa_new_unfil_x))[0]] = tps_interp_re[:, 0]
        spa_new_unfil_y[np.where(np.isnan(spa_new_unfil_y))[0]] = tps_interp_re[:, 1]

        spa_new_unfil = np.concatenate((np.expand_dims(spa_new_unfil_x, axis=1),
                                        np.expand_dims(spa_new_unfil_y, axis=1),
                                        np.zeros(shape=(spa_new_unfil_x.shape[0], 1))), axis=1)
        return spa_new_unfil

    def cal_tps_fit(spa_raw_fil, spa_new_fil, spa_raw_unfil, lambda_val):
        """
        tps fit
        """

        # 为了容纳奇异矩阵的处理，solver取‘lstsq’
        tps_trans = tps.TPS(spa_raw_fil, spa_new_fil, lambda_=lambda_val, solver='lstsq')
        tps_interp_re = tps_trans(spa_raw_unfil[:, :2])

        spa_new_unfil = np.concatenate((tps_interp_re, np.zeros(shape=(tps_interp_re.shape[0], 1))), axis=1)

        return spa_new_unfil

    for i in range(len(slicesl)):
        if not 'spatial_rigid' in list(slicesl[i].obsm):
            raise NameError("'spatial_rigid' not in anndata.obsm sub-columns, elastic transformation not allowed.")

    assert type(pis) == list, "Type of pis is not list"
    assert len(pis) == len(slicesl) - 1, "'slicesl' should have length one more than 'pis'."

    assert type(filter_by_label) == bool, "Type of filter_by_label is not bool"
    assert warp_type in ['linear', 'cubic', 'tps'], "warp_type is not among 'linear', 'cubic' and 'tps'."

    if warp_type == 'tps':
        try:
            lambda_val = float(lambda_val)
        except Exception as e:
            raise TypeError("lambda_val cannot be transformed to float while choosing tps warping") from e
        assert lambda_val > 0, "lambda_val is not greater than 0 while choosing tps_warping"

    # preprocessing of anchors
    slicesl[0].obsm['spatial_elas'] = slicesl[0].obsm['spatial_rigid'].copy()
    for i in range(len(slicesl) - 1):
        if i == 0:
            spaA, spaB = filter_rows_cols_to_spa(slicesl[i], slicesl[i + 1], filter_by_label, label_col,
                                                 spaAtype='spatial_rigid',
                                                 spaBtype='spatial_rigid')
        else:
            spaA, spaB = filter_rows_cols_to_spa(slicesl[i], slicesl[i + 1], filter_by_label, label_col,
                                                 spaAtype='spatial_elas',
                                                 spaBtype='spatial_rigid')

        # all points before filtering
        spaAraw_unfil = slicesl[i].obsm['spatial_rigid'].copy()
        spaBraw_unfil = slicesl[i + 1].obsm['spatial_rigid'].copy()

        try:
            pi_elas = filter_pi_mtx(to_dense_array(pis[i]), percentage=fil_pc, same_mem=False)
        except:  # in case that the pi matrix doesn't work
            logger.warning('transition matrix didn\'t work for the interpolating elastic method')
            slicesl[i].obsm['spatial_elas'] = slicesl[i].obsm['spatial_rigid']
            slicesl[i+1].obsm['spatial_elas'] = slicesl[i+1].obsm['spatial_rigid']
            continue

        # 计算得到A，B的过滤点的新位置
        _, spaBraw_fil, offset_xy2pre = cal_fil_offset_by_pi_to_pre(pi_elas, spaA, spaB)  # offset to be applied to sliceB, using sliceA as reference
        spaAraw_fil, _, offset_xy2next = cal_fil_offset_by_pi_to_next(pi_elas, spaA, spaB)  # offset to be applied to sliceA, using sliceB as reference

        if warp_type in ['linear', 'cubic', 'tps']:
            spaBnew_fil = spaBraw_fil + offset_xy2pre / 2
            spaAnew_fil = spaAraw_fil + offset_xy2next / 2
            # print('B', spaBraw_fil.shape, spaBnew_fil.shape, spaBraw_unfil.shape)
            # print('A', spaAraw_fil.shape, spaAnew_fil.shape, spaAraw_unfil.shape)

            if warp_type in ['linear', 'cubic']:
                spaBnew_unfil = cal_strict_fit(spaBraw_fil, spaBnew_fil, spaBraw_unfil, warp_type)
                spaAnew_unfil = cal_strict_fit(spaAraw_fil, spaAnew_fil, spaAraw_unfil, warp_type)
            else:
                spaBnew_unfil = cal_tps_fit(spaBraw_fil, spaBnew_fil, spaBraw_unfil, lambda_val)
                spaAnew_unfil = cal_tps_fit(spaAraw_fil, spaAnew_fil, spaAraw_unfil, lambda_val)

        slicesl[i].obsm['spatial_elas'] = spaAnew_unfil
        slicesl[i+1].obsm['spatial_elas'] = spaBnew_unfil

    for i, slice in enumerate(slicesl):
        slice.obsm['spatial_elas'][:, 2] = slice.obsm['spatial'][:, 2]

    return slicesl


def stack_slices_pairwise_elas_field(
        slicesl: List[AnnData],
        pis: List[np.ndarray],
        label_col: str,
        pixel_size: float,
        fil_pc: float = 20,
        filter_by_label: bool = True,
        sigma: float = 1,
        spatial_key: str = 'spatial'
    ) -> Tuple[List[AnnData]]:
    """

    Register slices in elastic way, in accordance to the order of slices.

    Each slice is elastically registered according to its transition matrix with previous and with next slice; while the
    first and last slice are registered only according to its transition matrix with its adjacent slice.

    Args:
        slicesl: list of rigid registered slices
        pis: list of probabilistic transition matrix
        label_col:  column name of adata.obs that stores annotated celltypes
        pixel_size: edge length of single pixel, when generating elastic field. Input a rough average of spots distance here
        fil_pc: percentage of ranked probabilities in transition matrix, after which probabilities will be filtered.
        filter_by_label: if spots were filtered acordding to annotation, in the ot solving process, or not
        sigma: sigma value of gaussina kernel, when filtering noises in elastic registration field, with a higher value
            indicating a smoother elastic field. Refer to this website to decide sigma according to your desired range of convolution.
            http://demofox.org/gauss.html
    Returns:
        elastic registration result saved in adata.obsm['spatial_elas']
    """

    def generate_fields_by_offset(spa_raw, spa_raw_fil, offset_xy, field_offset, pixel_size, sigma):
        """
        param spa_raw: spatial coordinates of all points of the slice to be registered
        param spa_raw_fil: spatial coordinates of the points with filtered anchors, of the slice to be registered
        param offset_xy: offset to be applied on the filtered spatial coordinates, indicated by anchors
        param: field_offset: minimum of x and y value of spatial coordinates of all points of the slice to be registered, in NdArray of shape (2,)
        param pixel_size: the representing size of a single pixel in the field matrix, in the same unit of spatial coordinates

        returned field x: field to be applied onto slice spots, with the first row aligning to the edge of spot(s) with lowest x coordinate,
                          and the first column aligning to the edge of spot(s) with lowest y coordinate. Each pixel has side
                          length of pixel_size, in the same unit of spatial coordinates.
        """
        # get pixel indices and values of points from filtered pi matrix
        spa_shifted_fil = spa_raw_fil - np.repeat(np.expand_dims(field_offset, axis=0), spa_raw_fil.shape[0],
                                                  axis=0)  # (n,2) # now the spatial coordinate starts at 0
        pixel_ind_fil = np.floor(spa_shifted_fil / pixel_size)  # get shifted spatial coordinates, in unit of pixel
        pixel_offset_x_fil = offset_xy[:, 0] / 2
        pixel_offset_y_fil = offset_xy[:, 1] / 2
        field_known_df = pd.DataFrame(
            {'indx': pixel_ind_fil[:, 0], 'indy': pixel_ind_fil[:, 1], 'offx': pixel_offset_x_fil,
             'offy': pixel_offset_y_fil})
        field_known_df = field_known_df.groupby(by=['indx', 'indy'])[['indx', 'indy', 'offx', 'offy']].agg(
            {'indx': 'first', 'indy': 'first', 'offx': 'mean', 'offy': 'mean'}).reset_index(drop=True)

        # get pixel indices of all points
        spa_shifted_raw = spa_raw - np.repeat(np.expand_dims(field_offset, axis=0), spa_raw.shape[0], axis=0)
        pixel_ind_raw = np.floor(spa_shifted_raw / pixel_size)  # (n, 2)

        # initiate two empty arrays to save field data
        field_x = np.empty(shape=(pixel_ind_raw.max(axis=0) + 1).astype(
            int))  # i index corresponds to x coordinate, and j index corresponds to y coordinates
        field_y = np.empty(shape=(pixel_ind_raw.max(axis=0) + 1).astype(int))

        # # visualize result of field generated by
        # field_x[:] = np.nan
        # field_x[field_known_df['indx'].to_numpy().astype(int), field_known_df['indy'].to_numpy().astype(int)] = field_known_df['offx'].to_numpy()
        # plt.figure()
        # plt.imshow(field_x, cmap='rainbow')
        # plt.colorbar()
        # plt.show()

        # generate the field value by interpolating, then reform the result to field
        xind = np.indices(field_x.shape)[0].flatten()
        yind = np.indices(field_y.shape)[1].flatten()
        xy_ind = np.concatenate((np.expand_dims(xind, axis=1), np.expand_dims(yind, axis=1)), axis=1)  # (n, 2)
        pixel_offset_x_interp = scipy.interpolate.griddata(field_known_df[['indx', 'indy']].to_numpy(),
                                                           field_known_df['offx'].to_numpy(), xy_ind, method='nearest')  # 不同methods？
        pixel_offset_y_interp = scipy.interpolate.griddata(field_known_df[['indx', 'indy']].to_numpy(),
                                                           field_known_df['offy'].to_numpy(), xy_ind, method='nearest')

        field_x[xind, yind] = pixel_offset_x_interp
        field_y[xind, yind] = pixel_offset_y_interp

        # # visualize result of interpolated field
        # plt.figure()
        # plt.imshow(field_x, cmap='rainbow')
        # plt.colorbar()
        # plt.show()

        # filter out noises by gaussian kernel
        field_x = scipy.ndimage.gaussian_filter(field_x, sigma=sigma, mode='mirror')  # todo: 可以修改为自适应的
        field_y = scipy.ndimage.gaussian_filter(field_y, sigma=sigma, mode='mirror')

        # # visualize result of interpolated field
        # plt.figure()
        # plt.imshow(field_x, cmap='rainbow')
        # plt.colorbar()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
        return field_x, field_y

    def generate_fields2next(sliceA, sliceB, filter_by_label, label_col, pi, fil_pc, pixel_size, sigma):
        spaA, spaB = filter_rows_cols_to_spa(sliceA, sliceB, filter_by_label, label_col, spaAtype='spatial_rigid',
                                             spaBtype='spatial_rigid')  # A is the first slice # (n, 3)
        pi_next = filter_pi_mtx(pi, percentage=fil_pc, same_mem=False)
        spaAraw_fil, _, offset_xy2next = cal_fil_offset_by_pi_to_next(pi_next, spaA, spaB)  # offset that should be applied onto filtered points, directly given by respective filtered anchors

        spaA_raw = sliceA.obsm['spatial_rigid'][:, :2]
        field_x, field_y = generate_fields_by_offset(spaA_raw, spaAraw_fil, offset_xy2next, spaA_raw.min(axis=0), pixel_size, sigma)  # fields to be applied onto this slice
        return field_x, field_y

    def generate_fields2pre(sliceA, sliceB, filter_by_label, label_col, pi, fil_pc, pixel_size, sigma):
        spaA, spaB = filter_rows_cols_to_spa(sliceA, sliceB, filter_by_label, label_col, spaAtype='spatial_rigid', spaBtype='spatial_rigid')  # B is the last slice # (n, 3)
        pi_pre = filter_pi_mtx(pi, percentage=fil_pc, same_mem=False)
        _, spaBraw_fil, offset_xy2pre = cal_fil_offset_by_pi_to_pre(pi_pre, spaA, spaB)

        spaB_raw = sliceB.obsm['spatial_rigid'][:, :2]
        field_x, field_y = generate_fields_by_offset(spaB_raw, spaBraw_fil, offset_xy2pre, spaB_raw.min(axis=0), pixel_size, sigma)
        return field_x, field_y

    def apply_field(field_x, field_y, pixel_size, margin, spa_arr):
        xind = np.indices(field_x.shape)[0].flatten()
        yind = np.indices(field_x.shape)[1].flatten()
        x_cor = xind * pixel_size + margin[0]
        y_cor = yind * pixel_size + margin[1]

        field_x_val = field_x[xind, yind]
        field_y_val = field_y[xind, yind]

        field_x_interp = scipy.interpolate.griddata(
            np.concatenate([np.expand_dims(x_cor, axis=1), np.expand_dims(y_cor, axis=1)], axis=1),
            field_x_val, spa_arr, method='linear')
        field_y_interp = scipy.interpolate.griddata(
            np.concatenate([np.expand_dims(x_cor, axis=1), np.expand_dims(y_cor, axis=1)], axis=1),
            field_y_val, spa_arr, method='linear')

        x_elas = spa_arr[:, 0] + field_x_interp
        y_elas = spa_arr[:, 1] + field_y_interp
        return np.concatenate([np.expand_dims(x_elas, axis=1), np.expand_dims(y_elas, axis=1)], axis=1)

    assert type(slicesl) == list, "Type of slicesl is not list"
    assert len(slicesl) >= 2, "Length of slicesl less than 2"

    for i in range(len(slicesl)):
        if not 'spatial_rigid' in list(slicesl[i].obsm):
            raise NameError("'spatial_rigid' not in anndata.obsm sub-columns, elastic transformation not allowed.")

    assert type(pis) == list, "Type of pis is not list"
    assert len(pis) == len(slicesl) - 1, "'slicesl' should have length one more than 'pis'."

    assert type(filter_by_label) == bool, "Type of filter_by_label is not bool"

    # preprocessing of anchors
    for i, slice in enumerate(slicesl):
        slice.obsm['spatial_elas'] = slice.obsm['spatial_rigid'].copy()

    # pixel_size = round(bin_size / unit_size_in_bin1, 2)
    for i in range(len(slicesl)):
        # 1. calculate field
        if i == 0:  # 1.1 first slice
            field_x2next, field_y2next = generate_fields2next(slicesl[i], slicesl[i + 1], filter_by_label, label_col, to_dense_array(pis[i]), fil_pc, pixel_size, sigma)
            field_x = field_x2next / 2
            field_y = field_y2next / 2
        elif i == len(slicesl) - 1:  # 1.2 last slice
            field_x2pre, field_y2pre = generate_fields2pre(slicesl[i-1], slicesl[i], filter_by_label, label_col, to_dense_array(pis[i-1]), fil_pc, pixel_size, sigma)
            field_x = field_x2pre / 2
            field_y = field_y2pre / 2
        else:  # 1.3 neither the first nor the last slice
            field_x2pre, field_y2pre = generate_fields2pre(slicesl[i - 1], slicesl[i], filter_by_label, label_col, to_dense_array(pis[i - 1]), fil_pc, pixel_size, sigma)
            field_x2next, field_y2next = generate_fields2next(slicesl[i], slicesl[i + 1], filter_by_label, label_col, to_dense_array(pis[i]), fil_pc, pixel_size, sigma)
            field_x = (field_x2pre + field_x2next) / 2
            field_y = (field_y2pre + field_y2next) / 2

            # # visualize the result
            # plt.figure()
            # plt.imshow(field_x, cmap='rainbow')
            # plt.colorbar()
            # plt.show()

        field_x = np.concatenate([field_x, np.expand_dims(field_x[-1, :], axis=0)], axis=0)  # to guarantee every spatial_rigid coordinate can interpolate by this field
        field_x = np.concatenate([field_x, np.expand_dims(field_x[:, -1], axis=1)], axis=1)
        field_y = np.concatenate([field_y, np.expand_dims(field_y[-1, :], axis=0)], axis=0)
        field_y = np.concatenate([field_y, np.expand_dims(field_y[:, -1], axis=1)], axis=1)

        # write the field to adata
        if not 'field' in slicesl[i].uns.keys():
            slicesl[i].uns['field'] = {}
        slicesl[i].uns['field']['pixel_size'] = pixel_size
        margin = slicesl[i].obsm['spatial_rigid'][:, :2].min(axis=0)  # margin when calculating field
        slicesl[i].uns['field']['margin'] = margin
        slicesl[i].uns['field']['field_x'] = field_x
        slicesl[i].uns['field']['field_y'] = field_y

        # 2. apply the field
        xy_elas = apply_field(field_x, field_y, pixel_size, margin, slicesl[i].obsm['spatial_rigid'][:, :2])

        # 3. write result to adata
        slicesl[i].obsm['spatial_elas'][:, :2] = xy_elas

    for i, slice in enumerate(slicesl):
        slice.obsm['spatial_elas'][:, 2] = slice.obsm[spatial_key][:, 2]

    return slicesl
