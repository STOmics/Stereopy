#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:test_cell_type_anno.py
@time:2021/03/12
"""
from stereo.tools.cluster import Cluster
from stereo.io.reader import read_stereo
import matplotlib.pyplot as plt


def get_data(path):
    data = read_stereo(path, bin_type='bins', bin_size=100)
    return data


def run_cluster(data):
    ct = Cluster(data, method='phenograph', normalize_method='zscore_disksmooth', dim_reduce_method='pca', n_neighbors=30)
    ct.fit()
    ct.plot_scatter(plot_dim_reduce=False)
    plt.show()
    return ct


if __name__ == '__main__':
    in_path = '/home/qiuping/workspace/st/stereopy_data/mouse/DP8400013846TR_F5.gem'
    data = get_data(in_path)
    run_cluster(data)

