#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_gsea.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/8/18 create file.
"""
import sys
from stereo.tools.gsea import ssgsea
from stereo.core.stereo_exp_data import StereoExpData

def make_data():
    import numpy as np
    from scipy import sparse
    genes = ['g1', 'MT-111', 'g3']
    rows = [0, 1, 0, 1, 2, 0, 1, 2]
    cols = [0, 0, 1, 1, 1, 2, 2, 2]
    cells = ['c1', 'c2', 'c3']
    v = [2, 3, 4, 5, 6, 3, 4, 5]
    exp_matrix = sparse.csr_matrix((v, (rows, cols)))
    # exp_matrix = sparse.csr_matrix((v, (rows, cols))).toarray()
    position = np.random.randint(0, 10, (len(cells), 2))
    out_path = sys.argv[0]+"/F5.h5ad"
    data = StereoExpData(bin_type='cell_bins', exp_matrix=exp_matrix, genes=np.array(genes),
                         cells=np.array(cells), position=position, output=out_path)
    return data


def test_ssgsea():
    ssgsea("")

#if __name__ == 'main' :
