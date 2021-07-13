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


def test_segment():
    image_path = '/home/qiuping/workspace/st/data/segment/20210426-T196-Z4-L-M019-01_regist.tif'
    out_path = '/home/qiuping/workspace/st/data/segment/cell_segment.tif'
    im.cell_seg(image_path, out_path, flag=True)


if __name__ == '__main__':
    test_segment()
