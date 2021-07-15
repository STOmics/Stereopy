#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_filter.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/07/14  create file.
"""
from stereo.preprocess.filter import filter_cells, filter_genes, filter_coordinates
from stereo.preprocess.qc import cal_qc
import numpy as np
from stereo.core.stereo_exp_data import StereoExpData
from scipy import sparse
import pandas as pd


def make_data():
    np.random.seed(1)
    a = np.random.randint(0, 2, size=(10, 8))
    cell = ['c' + str(i) for i in range(10)]
    gene = ['g' + str(i) for i in range(5)] + ['mt-1', 'mt-2', 'mt-3']
    exp_matrix = sparse.csr_matrix(a)
    position = np.random.randint(0, 10, (len(cell), 2))
    data = StereoExpData(bin_type='cell_bins', exp_matrix=exp_matrix, genes=np.array(gene),
                         cells=np.array(cell), position=position)
    data.init()
    return data


def data2df(data):
    df = pd.DataFrame(data.exp_matrix.toarray(), index=data.cell_names, columns=data.gene_names)
    if data.cells.total_counts is not None:
        df['total_counts'] = data.cells.total_counts
    if data.cells.n_genes_by_counts is not None:
        df['n_genes_by_counts'] = data.cells.n_genes_by_counts
    if data.cells.pct_counts_mt is not None:
        df['pct_counts_mt'] = data.cells.pct_counts_mt
    return df


if __name__ == '__main__':
    data = make_data()
    data = cal_qc(data)
    print(data2df(data))
    print(data.position)
    print('test for min_gene and max_gene:')
    data1 = filter_cells(data, min_gene=3, max_gene=4, inplace=False)
    print(data2df(data1))
    print('test for n_genes_by_counts:')
    data1 = filter_cells(data, n_genes_by_counts=6, inplace=False)
    print(data2df(data1))
    print('test for pct_counts_mt:')
    data1 = filter_cells(data, pct_counts_mt=30, inplace=False)
    print(data2df(data1))
    print('test for min_cell:')
    data1 = filter_genes(data, min_cell=5, inplace=False)
    print(data2df(data1))
    print('test for max_cell:')
    data1 = filter_genes(data, max_cell=6, inplace=False)
    print(data2df(data1))
    print('test for gene_list:')
    data1 = filter_genes(data, gene_list=['g1', 'g3'], inplace=False)
    print(data2df(data1))
    print('test for filter by position:')
    data1 = filter_coordinates(data, min_x=2, max_x=9, min_y=3, max_y=8, inplace=False)
    print(data2df(data1))
    print(data1.position)

