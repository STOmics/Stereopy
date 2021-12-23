#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_segment.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/07/13  create file.
"""
from stereo import image as im
import time


def test_segment():
    print('start')
    t = time.time()
    # 170s
    # img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/imgs/fov_stitched_auto_tile_regist.tif'
    img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/imgs/fov_stitched_regist.tif'  # 5.5h
    out_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/imgs/result'
    model_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/seg_utils/models/seg_model_20211210.pth'
    im.cell_seg(model_path, img_path, out_path, flag=True, depp_cro_size=20000, overlap=100, gpu='-1')
    print(time.time()-t)
    print('end')


def test_segment_deepcell():
    print('start')
    t = time.time()
    # 170s
    img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/imgs/fov_stitched_auto_tile_regist.tif'
    # img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/imgs/fov_stitched_regist.tif'  # 1.6h
    out_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/imgs/result/deepcell'
    model_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/deepcell/'
    im.cell_seg_deepcell(model_path, img_path, out_path, flag=True, depp_cro_size=20000, overlap=100, gpu='-1')
    print(time.time()-t)
    print('end')


if __name__ == '__main__':
    test_segment_deepcell()
