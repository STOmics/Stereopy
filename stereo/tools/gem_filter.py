# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:42:58 2022

@author: ywu28328
"""
import cv2
import numba as nb
import numpy as np
import tifffile as tif
from pandas import DataFrame

from stereo.log_manager import logger
from stereo.utils.time_consume import TimeConsume
from stereo.utils.time_consume import log_consumed_time

tc = TimeConsume()


def transfer_16bit_to_8bit(image_16bit):
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)

    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

    return image_8bit


@log_consumed_time
def gem_filter(tissue_mask_path: str, data: DataFrame):
    '''
    Filter gem.gz with tissue mask at low cost of memory.

    Parameters
    ----------
    tissue_mask_path : tissue mask path
    ----------
    Returns: None
    '''
    # print(tissue_mask)
    tk = tc.start()
    logger.info("start to filter data based on tissue mask")
    tissue_mask = tif.imread(tissue_mask_path)
    logger.info(f"tissue mask shape: {tissue_mask.shape}, type: {tissue_mask.dtype}")
    contours, hierarchy = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    logger.info(f"reading and findContours: {tc.get_time_consumed(key=tk)}")
    logger.info(f"count of contours: {len(contours)}")

    logger.info("get min and max coordinate")
    min_and_max = lambda c: [np.min(c[:, :, 0]), np.min(c[:, :, 1]), -np.max(c[:, :, 0]), -np.max(c[:, :, 1])] # noqa
    a = np.array([min_and_max(c) for c in contours], dtype=np.int32)
    min_x, min_y, max_x, max_y = np.abs(a.min(axis=0))
    logger.info(f"get min and max coordinate: {tc.get_time_consumed(key=tk)}")
    logger.info(f"tissue area: (min_x, max_x, min_y, max_y) = ({min_x}, {max_x}, {min_y}, {max_y})")

    logger.info("slice to coordinates")
    data_coordinates = data[['x', 'y']].values
    logger.info(f"slice to coordinates: {tc.get_time_consumed(key=tk)}")
    logger.info(f"start to filter data, data shape befor filtering: {data.shape}")
    filter_flag = filter_data(data_coordinates, min_x, min_y, max_x, max_y, tissue_mask)
    logger.info(f"filter data: {tc.get_time_consumed(key=tk)}")
    data_filtered = data[filter_flag]
    logger.info(f"data shape after filtering: {data_filtered.shape}")

    return data_filtered


# def filter_data(data_coordinates, min_x, min_y, max_x, max_y, tissue_mask):
#     x, y = data_coordinates[:, 0], data_coordinates[:, 1]
#     x_flag_1 = (x >= min_x)
#     x_flag_2 = (x <= max_x)
#     y_flag_1 = (y >= min_y)
#     y_flag_2 = (y <= max_y)
#     value_flag = (tissue_mask[y, x] > 0)

#     return x_flag_1 & x_flag_2 & y_flag_1 & y_flag_2 & value_flag

@nb.njit(cache=True, nogil=True, parallel=True)
def filter_data(data_coordinates, min_x, min_y, max_x, max_y, tissue_mask):
    data_count = data_coordinates.shape[0]
    filter_flag = np.zeros((data_count,), dtype=np.uint8).astype(np.bool8)
    for i in nb.prange(data_count):
        x = data_coordinates[i][0]
        y = data_coordinates[i][1]
        tissue_value = tissue_mask[y, x]
        if (x >= min_x) and (x <= max_x) and (y >= min_y) and (y <= max_y) and (tissue_value > 0):
            filter_flag[i] = True
    return filter_flag
