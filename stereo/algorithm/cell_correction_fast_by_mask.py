# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:04:50 2023

@author: ywu28328
"""
import os

import cv2
import numba as nb
import numpy as np
import tifffile as tif
from joblib import (
    Parallel,
    delayed,
    cpu_count
)
from scipy import ndimage
from tqdm import tqdm

from stereo.log_manager import logger
from stereo.utils.time_consume import TimeConsume
from stereo.utils.time_consume import log_consumed_time

tc = TimeConsume()

MASK_BLOCK_OVERLAP_DEFAULT = 100
MASK_ROW_BLOCK_SIZE_DEFAULT = 2000
MASK_COL_BLOCK_SIZE_DEFAULT = 2000


@log_consumed_time
def read_mask(mask_path):
    mask = tif.imread(mask_path)
    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)
    return mask


# @log_consumed_time
def create_edm_labels(mask):
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    mask[mask > 0] = 255
    mask = cv2.bitwise_not(mask)
    edm = ndimage.distance_transform_edt(mask)
    edm[edm > 255] = 255
    edm = edm.astype(np.uint8)
    return edm, labels


@nb.njit(cache=True, nogil=True)
def correct(edm, labels, distance):
    # print(f"Entering into main correction threading.")
    height, width = edm.shape
    queue = np.zeros((height * width, 2), dtype=np.int64)
    queued = np.where((edm > distance) | (labels != 0), np.uint8(1), np.uint8(0))
    ind = np.where((edm <= distance) & (labels == 0))
    queue_tail = 0
    for i, j in zip(*ind):
        if i > 0 and i < height - 1:
            if labels[i - 1, j] != 0 or labels[i + 1, j] != 0:
                queued[i, j] = 1
                queue[queue_tail] = i, j
                queue_tail += 1
                continue
        if j > 0 and j < width - 1:
            if labels[i, j - 1] != 0 or labels[i, j + 1] != 0:
                queued[i, j] = 1
                queue[queue_tail] = i, j
                queue_tail += 1
    labels = process_queue(queued, queue, queue_tail, labels, width, height)
    labels = np.where(labels > 0, np.uint8(1), np.uint8(0))
    return labels


# @log_consumed_time
@nb.njit(cache=True)
def getNeighborLabels8(labels, x, y, width, height):
    x_start, x_end = max(x - 1, 0), min(x + 2, height)
    y_start, y_end = max(y - 1, 0), min(y + 2, width)
    area = labels[x_start:x_end, y_start:y_end].copy().reshape(-1)
    nonzero_idx = np.nonzero(area)
    area_nonzero = area[nonzero_idx]
    if area_nonzero.size <= 0:
        return None
    if np.all(area_nonzero == area_nonzero[0]):
        return area_nonzero[0]
    return None


# @log_consumed_time
@nb.njit(cache=True)
def addNeighboursToQueue8(
        queued: np.ndarray,
        queue: np.ndarray,
        queue_tail: int,
        x: int,
        y: int,
        width: int,
        height: int
):
    if x > 0 and queued[x - 1, y] == 0:
        queued[x - 1, y] = 1
        queue[queue_tail] = x - 1, y
        queue_tail += 1
    if x < (height - 1) and queued[x + 1, y] == 0:
        queued[x + 1, y] = 1
        queue[queue_tail] = x + 1, y
        queue_tail += 1
    if y > 0 and queued[x, y - 1] == 0:
        queued[x, y - 1] = 1
        queue[queue_tail] = x, y - 1
        queue_tail += 1
    if y < (width - 1) and queued[x, y + 1] == 0:
        queued[x, y + 1] = 1
        queue[queue_tail] = x, y + 1
        queue_tail += 1
    if x > 0 and y > 0 and queued[x - 1, y - 1] == 0:
        queued[x - 1, y - 1] = 1
        queue[queue_tail] = x - 1, y - 1
        queue_tail += 1
    if x > 0 and y < (width - 1) and queued[x - 1, y + 1] == 0:
        queued[x - 1, y + 1] = 1
        queue[queue_tail] = x - 1, y + 1
        queue_tail += 1
    if x < (height - 1) and y > 0 and queued[x + 1, y - 1] == 0:
        queued[x + 1, y - 1] = 1
        queue[queue_tail] = x + 1, y - 1
        queue_tail += 1
    if x < (height - 1) and y < (width - 1) and queued[x + 1, y + 1] == 0:
        queued[x + 1, y + 1] = 1
        queue[queue_tail] = x + 1, y + 1
        queue_tail += 1

    return queue_tail


@nb.njit(cache=True)
def process_queue(queued, queue, queue_tail, labels, width, height):
    queue_head = 0
    while queue_head < queue_tail:
        x, y = queue[queue_head]
        queue_head += 1
        l = getNeighborLabels8(labels, x, y, width, height)  # noqa
        if l is None:
            continue
        labels[x, y] = l
        queue_tail = addNeighboursToQueue8(queued, queue, queue_tail, x, y, width, height)
    return labels


@log_consumed_time
@nb.njit(cache=True)
def crop_mask(mask):
    x, y = np.where(mask > 0)
    start_x, start_y = max(np.min(x) - 100, 0), max(np.min(y) - 100, 0)
    end_x, end_y = min(np.max(x) + 100, mask.shape[0]), min(np.max(y) + 100, mask.shape[1])
    start = (start_x, start_y)
    end = (end_x, end_y)
    cropmask = mask[start_x:end_x, start_y:end_y]
    return start, end, cropmask


def generate_block(
        image: np.ndarray,
        row_block_size: int = MASK_ROW_BLOCK_SIZE_DEFAULT,
        col_block_size: int = MASK_COL_BLOCK_SIZE_DEFAULT,
        overlap: int = MASK_BLOCK_OVERLAP_DEFAULT,
        bpr: int = None,
        bpc: int = None
):
    '''
    Divides array a into subarrays of size p-by-q
    p: block row size
    q: block column size
    '''
    height = image.shape[0]  # image row size
    width = image.shape[1]  # image column size
    for row_block_idx in range(bpr):
        for column_block_idx in range(bpc):
            row_start = row_block_idx * (row_block_size - overlap)
            col_start = column_block_idx * (col_block_size - overlap)
            row_end = min(row_start + row_block_size, height)
            col_end = min(col_start + col_block_size, width)
            block = image[row_start:row_end, col_start:col_end]
            yield block


@log_consumed_time
def array_to_block(
        image: np.ndarray,
        row_block_size: int = MASK_ROW_BLOCK_SIZE_DEFAULT,
        col_block_size: int = MASK_COL_BLOCK_SIZE_DEFAULT,
        overlap: int = MASK_BLOCK_OVERLAP_DEFAULT,
        n_split_data_jobs: int = -1
):
    '''
    Divides array a into subarrays of size p-by-q
    p: block row size
    q: block column size
    '''
    height = image.shape[0]  # image row size
    width = image.shape[1]  # image column size
    bpr = (height - overlap) // (row_block_size - overlap)  # blocks per row
    bpc = (width - overlap) // (col_block_size - overlap)  # blocks per column
    if ((height - overlap) % (row_block_size - overlap)) > 0:
        bpr += 1
    if ((width - overlap) % (col_block_size - overlap)) > 0:
        bpc += 1
    if n_split_data_jobs in (0, 1):
        block_list = [create_edm_labels(block) for block in
                      generate_block(image, row_block_size, col_block_size, overlap, bpr, bpc)]
    else:
        if n_split_data_jobs < 0 or n_split_data_jobs > cpu_count():
            n_split_data_jobs = cpu_count()
        if n_split_data_jobs > (bpr * bpc):
            n_split_data_jobs = bpr * bpc
        block_list = Parallel(n_jobs=n_split_data_jobs, backend='threading')(
            [delayed(create_edm_labels)(block) for block in
             generate_block(image, row_block_size, col_block_size, overlap, bpr, bpc)]
        )

    return block_list, bpr, bpc


@log_consumed_time
def merge_to_mask(final_result, bpr, bpc, mask, start, end, overlap=MASK_BLOCK_OVERLAP_DEFAULT):
    half_step = overlap // 2
    full_img_list = []
    for rr in range(bpr):
        row_img_list = []
        for cc in range(bpc):
            if bpc == 1:
                img = final_result[rr * bpc]
            elif cc == 0:
                img = final_result[rr * bpc][:, :-half_step]
            elif cc == bpc - 1:
                img = final_result[rr * bpc + cc][:, half_step:]
            else:
                img = final_result[rr * bpc + cc][:, half_step:-half_step]
            row_img_list.append(img)
        row_img = np.concatenate(row_img_list, axis=1)

        if bpr == 1:
            img = row_img
        elif rr == 0:
            img = row_img[:-half_step]
        elif rr == bpr - 1:
            img = row_img[half_step:]
        else:
            img = row_img[half_step:-half_step]
        full_img_list.append(img)
    full_img = np.concatenate(full_img_list, axis=0)

    mask[start[0]:end[0], start[1]:end[1]] = full_img

    return mask

# @log_consumed_time
# def merge_to_mask(final_result, bpr, bpc, mask, start, end, overlap=MASK_BLOCK_OVERLAP_DEFAULT):
#     full_img_shape = (end[0] - start[0], end[1] - start[1])
#     full_img = np.zeros(full_img_shape, dtype=mask.dtype)
#     for rr in range(bpr):
#         for cc in range(bpc):
#             row_start = rr * (MASK_ROW_BLOCK_SIZE_DEFAULT - overlap)
#             col_start = cc * (MASK_COL_BLOCK_SIZE_DEFAULT - overlap)
#             row_end = min(row_start + MASK_ROW_BLOCK_SIZE_DEFAULT, full_img_shape[0])
#             col_end = min(col_start + MASK_COL_BLOCK_SIZE_DEFAULT, full_img_shape[1])
#             full_img[row_start:row_end, col_start:col_end] = final_result[rr * bpc + cc]

#     mask[start[0]:end[0], start[1]:end[1]] = full_img

#     return mask


@log_consumed_time
def est_para(mask):
    _, maskImg = cv2.connectedComponents(mask, connectivity=8)
    cell_avg_area = np.count_nonzero(mask) / np.max(maskImg)
    if cell_avg_area >= 350:
        logger.info(f'cell average size is {cell_avg_area}, d recommend 5 or 10')
    else:
        radius = int(np.sqrt(400 / np.pi) - np.sqrt(cell_avg_area / np.pi))
        logger.info(f'd recommend at least {radius}')
    import psutil
    logger.info(f'processes perfer set to {int(psutil.cpu_count(logical=False) * 0.7)}')


@log_consumed_time
def main(
        mask_path: str = None,
        n_jobs: int = 10,
        n_split_data_jobs: int = -1,
        distance: int = 10,
        out_path: str = './',
        return_data: bool = False,
        save_data: bool = True
):
    logger.info("Enter into the function of correcting cells by cell mask.")

    mask = read_mask(mask_path)
    start, end, cropmask = crop_mask(mask)
    block_list, bpr, bpc = array_to_block(cropmask, n_split_data_jobs=n_split_data_jobs)

    logger.info(f"Number of blocks: {len(block_list)}")

    tk = tc.start()
    if n_jobs < 0 or n_jobs > cpu_count():
        n_jobs = cpu_count()
    if n_jobs > len(block_list):
        n_jobs = len(block_list)
    if n_jobs > 1:
        logger.info(f'Correcting starts on {n_jobs} threading')
        final_result = Parallel(n_jobs=n_jobs, backend='threading')(
            [delayed(correct)(edm, labels, distance) for edm, labels in block_list])
    else:
        logger.info('Correcting starts on single threading')
        final_result = [correct(edm, labels, distance) for edm, labels in
                        tqdm(block_list, desc='correcting', ncols=100)]
    logger.info(f'Correcting finished, time consumed: {tc.get_time_consumed(tk, restart=False)}')

    mask = merge_to_mask(final_result, bpr, bpc, mask, start, end)

    if not save_data:
        return_data = True

    if save_data:
        tk = tc.start()
        file_name = os.path.splitext(os.path.basename(mask_path))[0]
        file_path = os.path.join(out_path, f'{file_name}_edm_dis_{distance}.tif')
        logger.info(f'Saving new mask to {file_path}.')
        tif.imwrite(file_path, mask)
        logger.info(f'Saving new mask finished, time consumed: {tc.get_time_consumed(tk, restart=False)}')
        if return_data:
            return file_path, mask
        else:
            return file_path

    return mask
