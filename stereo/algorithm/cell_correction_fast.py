# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:13:48 2022

@author: ywu28328
"""
import os
import gc
import datetime
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tifffile as tifi
import gzip
from scipy.spatial import ConvexHull
from tqdm import tqdm
from collections import defaultdict
from ..log_manager import logger


def parse_head(gem):
    if gem.endswith('.gz'):
        f = gzip.open(gem, 'rb')
    else:
        f = open(gem, 'rb')

    header = ''
    num_of_header_lines = 0
    eoh = 0
    for i, l in enumerate(f):
        l = l.decode("utf-8")  # read in as binary, decode first
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


def find_nearest(array, value):
    return np.searchsorted(array, value, side="left")


def calDis(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2, axis=1))


def allocate_free_pts(cell_list, cell_points, free_points_np):
    allocation_dict = {}
    x_axis, _ = free_points_np.T
    for cell in tqdm(cell_list, desc='correcting cells'):
        min_x = min(cell_points[cell], key=lambda x: x[0])[0] - 30
        max_x = max(cell_points[cell], key=lambda x: x[0])[0] + 30
        # pts_np = cell_points[cell_points[:, 2] == cell][:, [0, 1]]
        # min_x = np.min(pts_np, axis=0)[0]
        # max_x = np.max(pts_np, axis=0)[0]

        idx_x_low, idx_x_upper = find_nearest(x_axis, min_x), find_nearest(x_axis, max_x)
        sub_free_pts_np = free_points_np[idx_x_low:idx_x_upper]

        pts_np = np.array(cell_points[cell])
        x, y = np.sum(pts_np, axis=0)
        centroid = [int(x / len(pts_np)), int(y / len(pts_np))]
        length = 5
        try:
            # hull = ConvexHull(cell_points[cell])
            # pts_in_hull = np.array(cell_points[cell])[hull.vertices]
            hull = ConvexHull(pts_np)
            pts_in_hull = pts_np[hull.vertices]
            hull_dis_matrix = calDis(pts_in_hull, np.array(centroid))
            hull_min_dis = np.min(hull_dis_matrix)
            hull_max_dis = np.max(hull_dis_matrix)
            if hull_max_dis - hull_min_dis > 5:
                length = (2 * hull_min_dis + hull_max_dis) // 2
            else:
                length = hull_max_dis + 5
        except:
            length = 5
        dis_matrix = calDis(sub_free_pts_np, np.array(centroid))
        idx = np.where(dis_matrix < length)
        allocation_dict[cell] = sub_free_pts_np[idx].tolist()
    return allocation_dict


def save_to_csv(allocation_dict, cell_points, data_np, out_path):
    data = pd.DataFrame(data_np, columns=['geneID', 'x', 'y', 'UMICount', 'label'])
    free_pts_allocation = {}
    for label in allocation_dict:
        for pt in allocation_dict[label]:
            if pt not in cell_points[label]:
                free_pts_allocation[tuple(pt)] = label

    # data['tag'] = ['raw' for i in range(len(data['label']))]
    data['tag'] = 'raw'
    data_np = data.to_numpy()
    for i in range(len(data_np)):
        if (data_np[i][1], data_np[i][2]) in free_pts_allocation:
            data_np[i][4] = free_pts_allocation[(data_np[i][1], data_np[i][2])]
            data_np[i][5] = 'adjust'

    data_np = [i for i in data_np if i[5] == 'adjust' or i[4] != 0]

    test_df = pd.DataFrame(data_np, columns=['geneID', 'x', 'y', 'UMICount', 'label', 'tag'])
    test_df.to_csv(os.path.join(out_path, 'data_adjust.txt'), sep='\t', index=False)

    test_df = pd.DataFrame(
        data_np, columns=['geneID', 'x', 'y', 'UMICount', 'label', 'tag'])
    test_df.to_csv(os.path.join(out_path, 'data_adjust.csv'), sep=',', index=False)

def save_to_result(allocation_dict, cell_points, data_np):
    data = pd.DataFrame(data_np, columns=['geneID', 'x', 'y', 'UMICount', 'label'])
    free_pts_allocation = {}
    for label in allocation_dict:
        for pt in allocation_dict[label]:
            if pt not in cell_points[label]:
                free_pts_allocation[tuple(pt)] = label

    data['tag'] = ['raw' for i in range(len(data['label']))]
    data_np = data.to_numpy()
    for i in range(len(data_np)):
        if (data_np[i][1], data_np[i][2]) in free_pts_allocation:
            data_np[i][4] = free_pts_allocation[(data_np[i][1], data_np[i][2])]
            data_np[i][5] = 'adjust'

    data_np = [i for i in data_np if i[5] == 'adjust' or i[4] != 0]
    return pd.DataFrame(data_np, columns=['geneID', 'x', 'y', 'UMICount', 'label', 'tag'])


def draw_allocated_pts(allocation_dict, data_np):
    img = np.zeros([max(data_np, key=lambda x: x[2])[2] + 1, max(data_np, key=lambda x: x[1])[1] + 1, 3], dtype=np.uint8)
    for i in data_np:
        if i[4] != 0:
            img[i[2], i[1], 0] = 255
        else:
            img[i[2], i[1], 1] = 255
    for cell_label in allocation_dict:
        for pt in allocation_dict[cell_label]:
            img[pt[1], pt[0], 1] = 0
            img[pt[1], pt[0], 2] = 255
    im = Image.fromarray(img)
    return im
    # im.save(os.path.join(out_path, 'all_quick.tif'))


# def load_data(gem_file, mask_file):
#     if isinstance(gem_file, pd.DataFrame):
#         data_np = gem_file
#     else:
#         data_np = creat_cell_gxp(mask_file, gem_file, transposition=False)

#     data_np['x'] = data_np['x'] - data_np['x'].min()
#     data_np['y'] = data_np['y'] - data_np['y'].min()
#     data_np['label'] = data_np['label'].astype(int)
#     data_np = data_np.drop_duplicates(['x', 'y', 'label'], keep='last')
#     data_np = data_np.to_numpy()
#     free_points = data_np[data_np[:, 4] == 0][:, [1, 2]].tolist()
#     free_points = sorted(free_points, key=lambda x: (x[0], x[1])) 
#     free_points_np = np.array(free_points)
#     cell_points = data_np[data_np[:, 4] == 0][:, [1, 2, 4]]
#     cell_label_list = np.unique(data_np[:, 4])
#     return data_np, cell_label_list, cell_points, free_points_np

def load_data(gem_file, mask_file):
    if isinstance(gem_file, pd.DataFrame):
        data_np = gem_file
    else:
        data_np = creat_cell_gxp(mask_file, gem_file, transposition=False)

    data_np['label'] = data_np['label'].astype(int)
    data_np = data_np.to_numpy()

    cell_points = {}
    free_points = []
    for row in data_np:
        if row[4] != 0:
            if row[4] not in cell_points:
                cell_points[row[4]] = [[row[1], row[2]]]
            else:
                cell_points[row[4]].append([row[1], row[2]])
        else:
            free_points.append([row[1], row[2]])
    free_points = sorted(free_points, key=lambda x: (x[0], x[1])) 
    free_points_np = np.array(free_points)
    cell_label_list = sorted(list(cell_points.keys()))
    return data_np, cell_label_list, cell_points, free_points_np


def cell_correct(gem_file, mask_file):
    logger.info("start to correct cells!!!")
    data_np, cell_label_list, cell_points, free_points_np = load_data(gem_file, mask_file)
    logger.info("load data end, start to allocate free RNA")
    allocation_dict = allocate_free_pts(cell_label_list, cell_points, free_points_np)
    logger.info(f'allocate free RNA end, type of allocated RNA is {len(allocation_dict)}')
    return save_to_result(allocation_dict, cell_points, data_np)