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
from stereo.io.reader import read_gem
from stereo.plots.marker_genes import marker_genes_text, marker_genes_heatmap
import matplotlib.pyplot as plt
import pickle
import numpy as np


def get_data(path):
    data = read_gem(path, bin_type='bins', bin_size=100)
    return data


def run_cluster(data):
    ct = Clustering(data, normalization=True)
    ct.fit()
    return ct


def run_find_marker(data, group):
    ft = FindMarker(data, group, method='wilcoxon_test', case_groups=np.array(['0', '1', '2']), control_groups='2')
    ft.plot_heatmap()
    plt.savefig('./heatmap.jpg')
    ft.plot_marker_text()
    plt.savefig('./text.jpg')
    return ft


def test_heatmap(data, ct_res, ft_res):
    marker_genes_heatmap(data, ct_res, ft_res)
    plt.savefig('./heatmap.jpg')


def test_text(ft_res, group='all'):
    marker_genes_text(ft_res, group)
    plt.savefig('./text.jpg')


def pickle_res(in_path):
    data = get_data(in_path)
    ct = run_cluster(data)
    ft = run_find_marker(data, ct.result.matrix)
    pickle.dump(ct.result, open('./ct.pk', 'wb'))
    pickle.dump(ft.result, open('./ft.pk', 'wb'))


def test_heatmap_gene_list(data, ct_res, ft_res, gene_list, min_value, max_value):
    marker_genes_heatmap(data, ct_res, ft_res, gene_list=gene_list, min_value=min_value, max_value=max_value)
    plt.savefig('./heatmap1.jpg')


def test_logres():
    import pickle
    import scanpy as sc
    from anndata import AnnData

    data = pickle.load(open('/home/qiuping/workspace/st/data/test_data.pickle', 'rb'))

    adata = AnnData(data.to_df())
    print(1)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    # sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)
    print(2)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    print(3)
    sc.tl.rank_genes_groups(adata, 'leiden', method='logreg', groups=['2', '4', '0'], reference='1')
    from matplotlib import pyplot as plt

    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    plt.show()


if __name__ == '__main__':
    in_path = '/home/qiuping/workspace/st/stereopy_data/mouse/DP8400013846TR_F5.gem'
    data = get_data(in_path)
    ct_result = pickle.load(open('./ct.pk', 'rb'))
    ft_result = pickle.load(open('./ft.pk', 'rb'))
    ft = run_find_marker(data, ct_result.matrix)
    # test_heatmap_gene_list(data, ct_result, ft_result, None, 300, 800)
    # test_heatmap_gene_list(data, ct_result, ft_result, ['Fga', 'Apoe'], 1, 50)
    # test_heatmap(data, ct_result, ft_result)
    # test_text(ft_result)
    # test_logres()
