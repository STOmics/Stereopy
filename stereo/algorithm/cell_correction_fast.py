# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:13:48 2022

@author: ywu28328
"""
import numba
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm

from ..log_manager import logger
from ..utils.time_consume import TimeConsume
from ..utils.time_consume import log_consumed_time

tc = TimeConsume()


def parse_head(gem):
    if gem.endswith('.gz'):
        import gzip
        f = gzip.open(gem, 'rb')
    else:
        f = open(gem, 'rb')

    header = ''
    num_of_header_lines = 0
    eoh = 0
    for i, l in enumerate(f):
        l = l.decode("utf-8")  # read in as binary, decode first # noqa
        if l.startswith('#'):  # header lines always start with '#'
            header += l
            num_of_header_lines += 1
            eoh = f.tell()  # get end-of-header position
        else:
            break
    # find start of expression matrix
    f.seek(eoh)

    return num_of_header_lines


def creat_cell_gxp(maskFile, geneFile, transposition=False):
    import cv2
    import tifffile as tifi
    print("Loading mask file...")
    mask = tifi.imread(maskFile)

    if transposition:
        mask = mask.T

    _, maskImg = cv2.connectedComponents(mask)

    print("Reading data..")
    typeColumn = {
        "geneID": 'str',
        "x": np.uint32,
        "y": np.uint32,
        "values": np.uint32,
        "UMICount": np.uint32,
        "MIDCount": np.uint32,
        "MIDCounts": np.uint32
    }

    header = parse_head(geneFile)
    genedf = pd.read_csv(geneFile, header=header, sep='\t', dtype=typeColumn)
    if "UMICount" in genedf.columns:
        genedf = genedf.rename(columns={'UMICount': 'MIDCount'})
    if "MIDCounts" in genedf.columns:
        genedf = genedf.rename(columns={'MIDCounts': 'MIDCount'})

    tissuedf = pd.DataFrame()
    dst = np.nonzero(maskImg)

    print("Dumping results...")
    tissuedf['x'] = dst[1] + genedf['x'].min()
    tissuedf['y'] = dst[0] + genedf['y'].min()
    tissuedf['label'] = maskImg[dst]

    res = pd.merge(genedf, tissuedf, on=['x', 'y'], how='left').fillna(0)  # keep background data
    return res


@numba.njit(cache=True, nogil=True)
def find_nearest(array, value):
    return np.searchsorted(array, value, side="left")


@numba.njit(cache=True, nogil=True)
def calDis(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))


@log_consumed_time
def allocate_free_pts(data: np.ndarray, cell_points, cells_index, free_points, free_points_index):
    x_axis, _ = free_points.T
    for cell in tqdm(cells_index, desc='correcting cells'):
        cell_id, cell_start, cell_end = cell
        current_cell_points = cell_points[cell_start:cell_end]
        min_x = np.min(current_cell_points, axis=0)[0] - 30
        max_x = np.max(current_cell_points, axis=0)[0] + 30

        idx_x_low, idx_x_upper = find_nearest(x_axis, min_x), find_nearest(x_axis, max_x)
        sub_free_pts_np = free_points[idx_x_low:idx_x_upper]
        sub_idx_in_data = free_points_index[idx_x_low:idx_x_upper]

        x, y = np.sum(current_cell_points, axis=0)
        centroid = np.array([int(x / len(current_cell_points)), int(y / len(current_cell_points))])
        length = 5
        try:
            hull = ConvexHull(current_cell_points)
            pts_in_hull = current_cell_points[hull.vertices]
            hull_dis_matrix = calDis(pts_in_hull, centroid)
            hull_min_dis = np.min(hull_dis_matrix)
            hull_max_dis = np.max(hull_dis_matrix)
            if hull_max_dis - hull_min_dis > 5:
                length = (2 * hull_min_dis + hull_max_dis) // 2
            else:
                length = hull_max_dis + 5
        except Exception:
            length = 5
        dis_matrix = calDis(sub_free_pts_np, centroid)
        idx = np.where(dis_matrix < length)
        sub_idx_in_data_allocated = sub_idx_in_data[idx]
        data[sub_idx_in_data_allocated, 4] = cell_id
        data[sub_idx_in_data_allocated, 6] = 'adjust'
    return data


@log_consumed_time
def save_to_result(data_np):
    data_np = data_np[data_np[:, 4] > 0]
    return pd.DataFrame(data_np, columns=['geneID', 'x', 'y', 'UMICount', 'label', 'geneid', 'tag'])


@log_consumed_time
def load_data(gem_file, mask_file):
    if isinstance(gem_file, pd.DataFrame):
        data = gem_file
    else:
        data = creat_cell_gxp(mask_file, gem_file, transposition=False)

    data.sort_values(by='label', ignore_index=True, inplace=True)
    data['tag'] = 'raw'
    positions = data[['x', 'y']].to_numpy(dtype=np.uint32)
    labels = data['label'].values.astype(dtype=np.uint32)

    return data.to_numpy(), positions, labels


@log_consumed_time
@numba.njit(cache=True, parallel=True, nogil=True)
def preprocess(positions, labels):
    free_points_count = (labels == 0).sum()
    cell_points_count = labels.shape[0] - free_points_count
    cell_points = np.zeros(shape=(cell_points_count, 2), dtype=np.int32)
    free_points = np.zeros(shape=(free_points_count, 2), dtype=np.uint32)
    free_points_x_axis = positions[0:free_points_count, 0]
    free_points_index = np.argsort(free_points_x_axis)
    free_points[:] = positions[0:free_points_count][free_points_index]
    cell_points[:] = positions[free_points_count:]
    cell_ids = np.unique(labels)[1:]
    cells_count = cell_ids.shape[0]
    cells_index = np.zeros(shape=(cells_count, 3), dtype=np.uint32)
    last_cell_id = 0
    cells_index_idx = 0
    for i in range(cell_points.shape[0]):
        j = i + free_points_count
        current_cell_id = labels[j]
        if current_cell_id != last_cell_id:
            cells_index[cells_index_idx][0] = current_cell_id
            cells_index[cells_index_idx][1] = i
            if cells_index_idx > 0:
                cells_index[cells_index_idx - 1][2] = i
            cells_index_idx += 1
            last_cell_id = current_cell_id
    cells_index[-1, 2] = cell_points.shape[0]

    return cell_points, cells_index, free_points, free_points_index


@log_consumed_time
def cell_correct(gem_file, mask_file):
    logger.info("start to correct cells!!!")
    data, positions, labels = load_data(gem_file, mask_file)
    cell_points, cells_index, free_points, free_points_index = preprocess(positions, labels)
    del positions, labels
    logger.info("load data end, start to allocate free RNA")
    data = allocate_free_pts(data, cell_points, cells_index, free_points, free_points_index)
    logger.info(f'allocate free RNA end, type of allocated RNA is {len(cells_index)}')
    return save_to_result(data)
