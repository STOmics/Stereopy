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
    img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/segmentation_1229/img/20210426-T173-Z3-L-M019-01_regist_21635_18385_9064_13184.tif'
    out_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/segmentation_1229/img/stereopy'
    model_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/segmentation_1229/seg_utils/models/seg_model_20211210.pth'
    im.cell_seg(model_path, img_path, out_path, depp_cro_size=20000, overlap=100, gpu='-1')
    print(time.time()-t)
    print('end')


def test_segment_deepcell():
    print('start')
    t = time.time()
    # 170s
    img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/segmentation_1229/img/20210426-T173-Z3-L-M019-01_regist_21635_18385_9064_13184.tif'
    # img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/segmentation/imgs/fov_stitched_regist.tif'  # 1.6h
    out_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/segmentation_1229/img/stereopy/deepcell'
    model_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/deepcell/'
    im.cell_seg_deepcell(model_path, img_path, out_path, depp_cro_size=20000, overlap=100, gpu='-1')
    print(time.time()-t)
    print('end')


if __name__ == '__main__':
    test_segment()
    test_segment_deepcell()
