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
    data = Data(file_path=filename, file_format='gem')
    data.check()
    print(data.file)


def test_read_bins():
    stereo_data = StereoExpData(file_path=filename, file_format='gem', bin_type='bins')
    stereo_data.read()
    print(stereo_data.genes)
    print(stereo_data.cells)
    print(stereo_data.exp_matrix)
    return stereo_data


def test_read_cell_bins():
    path = '/home/qiuping//workspace/st/stereopy_data/cell_bin/ssdna_Cell_GetExp_gene.txt'
    stereo_data = StereoExpData(file_path=path, file_format='gem', bin_type='cell_bins')
    stereo_data.read()
    print(stereo_data.genes)
    print(stereo_data.cells)
    print(stereo_data.exp_matrix)


def make_data():
    import numpy as np
    from scipy import sparse
    genes = ['g1', 'g2', 'g3']
    rows = [0, 1, 0, 1, 2, 0, 1, 2]
    cols = [0, 0, 1, 1, 1, 2, 2, 2]
    cells = ['c1', 'c2', 'c3']
    v = [2, 3, 4, 5, 6, 3, 4, 5]
    exp_matrix = sparse.csr_matrix((v, (rows, cols)))
    position = np.random.randint(0, 10, (len(cells), 2))
    out_path = '/home/qiuping//workspace/st/stereopy_data/test.h5ad'
    data = StereoExpData(bin_type='cell_bins', exp_matrix=exp_matrix, genes=np.array(genes),
                         cells=np.array(cells), position=position, output=out_path)
    return data


def print_data(data):
    print('genes:')
    print(data.genes.gene_name)
    print('cells:')
    print(data.cells.cell_name)
    print('exp_matrix:')
    print(data.exp_matrix.shape)
    print(data.exp_matrix)
    print('pos:')
    print(data.position)
    print('bin_type:')
    print(data.bin_type)


def test_write_h5ad(data=None):
    data = make_data() if data is None else data
    data.write_h5ad()
    print_data(data)


def test_read_h5ad():
    file_path = '/home/qiuping//workspace/st/stereopy_data/test.h5ad'
    data = StereoExpData(file_path=file_path, file_format='h5ad')
    data.read()
    print_data(data)


if __name__ == '__main__':
    data = make_data()
    data.exp_matrix = data.exp_matrix.toarray()
    print(data.log1p(inplace=False))
    data.normalize_total(inplace=True)
    print(data.exp_matrix)
    # print('test write...')
    # data = test_read_bins()
    # data.output = '/home/qiuping//workspace/st/stereopy_data/mource_bin100.h5ad'
    # test_write_h5ad(data)
