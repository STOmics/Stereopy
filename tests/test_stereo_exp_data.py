#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_stereo_exp_data.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/17  create file.
"""

from stereo.core.data import Data
from stereo.core.stereo_exp_data import StereoExpData


filename = '/home/qiuping/workspace/st/stereopy_data/mouse/DP8400013846TR_F5.gem'
binsize = 100


def test_data():
    data = Data(file_path=filename, file_format='txt')
    data.check()
    print(data.file)


def test_read_bins():
    stereo_data = StereoExpData(file_path=filename, file_format='txt', bin_type='bins')
    stereo_data.read()
    print(stereo_data.genes)
    print(stereo_data.cells)
    print(stereo_data.exp_matrix)


def test_read_cell_bins():
    path = '/home/qiuping//workspace/st/stereopy_data/cell_bin/ssdna_Cell_GetExp_gene.txt'
    stereo_data = StereoExpData(file_path=path, file_format='txt', bin_type='cell_bins')
    stereo_data.read()
    print(stereo_data.genes)
    print(stereo_data.cells)
    print(stereo_data.exp_matrix)


test_read_cell_bins()
