#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_qc.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/07/05  create file.
"""
from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.qc import cal_qc
from stereo.io.reader import read_h5ad


def make_data():
    import numpy as np
    from scipy import sparse
    genes = ['g1', 'mt-111', 'g3']
    rows = [0, 1, 0, 1, 2, 0, 1, 2]
    cols = [0, 0, 1, 1, 1, 2, 2, 2]
    cells = ['c1', 'c2', 'c3']
    v = [2, 3, 4, 5, 6, 3, 4, 5]
    exp_matrix = sparse.csr_matrix((v, (rows, cols)))
    # exp_matrix = sparse.csr_matrix((v, (rows, cols))).toarray()
    position = np.random.randint(0, 10, (len(cells), 2))
    out_path = '/home/qiuping//workspace/st/stereopy_data/test.h5ad'
    data = StereoExpData(bin_type='cell_bins', exp_matrix=exp_matrix, genes=np.array(genes),
                         cells=np.array(cells), position=position, output=out_path)
    return data


def quick_test():
    data = make_data()
    data = cal_qc(data)
    print(data.gene_names.dtype)
    print(data.cells.total_counts)
    print(data.cells.n_genes_by_counts)
    print(data.cells.pct_counts_mt)
    return data


def test_file():
    path = '/home/qiuping//workspace/st/stereopy_data/mource_bin100.h5ad'
    data = read_h5ad(path)
    print(data.gene_names.dtype)

    data = cal_qc(data)
    print(data.cells.total_counts)
    print(data.cells.n_genes_by_counts)
    print(data.cells.pct_counts_mt)


if __name__ == '__main__':
    quick_test()
    test_file()

