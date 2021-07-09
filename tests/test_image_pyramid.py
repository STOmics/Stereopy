#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_image_pyramid.py
@description: test for pyramid.py
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/11  create file.
"""
from stereo import image as im


def test_merge_image():
    h5_path = '/home/qiuping/workspace/st/data/pyramid/T90_fullsize_pyramid.h5'
    bin_size = 50
    out_path = '/home/qiuping/workspace/st/data/pyramid/T90'
    im.merge_pyramid(h5_path, bin_size, out_path)


if __name__ == '__main__':
    test_merge_image()
