#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:test_cell_type_anno.py
@time:2021/03/12
"""
#import sys
#sys.path.append('/data/workspace/st/stereopy-release')

from stereo.tools.cell_type_anno import CellTypeAnno
import scanpy as sc


andata = sc.read_h5ad('/data/workspace/st/data/monkey_brain/filter_100/raw_andata.bin100.h5ad')
ref_dir='/data/workspace/st/data/database/split_ref/monkey/GSE127898'
cores = 3
keep_zeros = True
use_rf = True
sample_rate = 0.8
n_estimators = 20
strategy = '2'
method = 'spearmanr'
split_num = 16
out_dir = '/data/workspace/st/data/monkey_brain/filter_100/anno_rf'
anno = CellTypeAnno(andata, ref_dir=None, cores=cores, keep_zeros=keep_zeros, use_rf=use_rf, sample_rate=sample_rate,
                    n_estimators=n_estimators, strategy=strategy, method=method, split_num=split_num, out_dir=out_dir, name='cell_type')
anno.fit()
sc.write(out_dir + '/anno_rf.h5ad', andata)
print(anno.data.uns['cell_type'])
