#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:test_find_markers.py
@time:2021/03/16
"""
from stereo.tools.find_markers import FindMarker
from stereo.tools.clustering import Clustering
from stereo.io.reader import read_stereo
from stereo.plots.marker_genes import plot_marker_genes_text, plot_marker_genes_heatmap
import matplotlib.pyplot as plt
import pickle


def get_data(path):
    data = read_stereo(path, bin_type='bins', bin_size=100)
    return data


def run_cluster(data):
    ct = Clustering(data, normalization=True)
    ct.fit()
    return ct


def run_find_marker(data, group):
    ft = FindMarker(data, group)
    ft.fit()
    return ft


def test_heatmap(data, ct_res, ft_res):
    plot_marker_genes_heatmap(data, ct_res, ft_res)
    plt.savefig('./heatmap.jpg')


def test_text(ft_res, group='all'):
    plot_marker_genes_text(ft_res, group)
    plt.savefig('./text.jpg')


def pickle_res(in_path):
    data = get_data(in_path)
    ct = run_cluster(data)
    ft = run_find_marker(data, ct.result.matrix)
    pickle.dump(ct.result, open('./ct.pk', 'wb'))
    pickle.dump(ft.result, open('./ft.pk', 'wb'))


def test_heatmap_gene_list(data, ct_res, ft_res, gene_list, min_value, max_value):
    plot_marker_genes_heatmap(data, ct_res, ft_res, gene_list=gene_list, min_value=min_value, max_value=max_value)
    plt.savefig('./heatmap1.jpg')


if __name__ == '__main__':
    in_path = '/home/qiuping/workspace/st/stereopy_data/mouse/DP8400013846TR_F5.gem'
    pickle_res(in_path)
    data = get_data(in_path)
    ct_result = pickle.load(open('./ct.pk', 'rb'))
    ft_result = pickle.load(open('./ft.pk', 'rb'))
    test_heatmap_gene_list(data, ct_result, ft_result, None, 300, 800)
    # test_heatmap_gene_list(data, ct_result, ft_result, ['Fga', 'Apoe'], 1, 50)
    # test_heatmap(data, ct_result, ft_result)
    test_text(ft_result)

