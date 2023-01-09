#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@file: regulatory_network.py
@time: 2023/Jan/08
@description: implement gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI

change log:
    2023/01/08 init
'''

# python core modules
import os
import time
import sys

# third party modules
import glob
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import scanpy as sc
import loompy as lp
from dask.diagnostics import ProgressBar
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies#, load_motifs
from pyscenic.aucell import aucell  #, derive_auc_threshold, create_rankings

# modules in self project
from ..log_manager import logger
from .algorithm_base import AlgorithmBase
from stereo.io.reader import read_gef


def csv2loom(fn:str):
    '''Convert single-cell csv file into a loom file'''
    basename = os.basename(fn).split('.')[0]
    x = sc.read_csv(fn)
    row_attrs = {'Gene':np.array(x.var_names),};
    col_attrs = {'CellID':np.array(x.obs_names),};
    lp.create(f'{basename}.loom', x.X.transpose(), row_attrs, col_attrs)



class RegulatoryNetwork(AlgorithmBase):
    '''
    A network object
    '''
    def __init__(self, matrix):
        self._expr_matrix = matrix
        self._pairs = [] #TF-gene pairs
        self._motif = [] #motif enrichment dataframe
        self._auc_matrix # TF activity level matrix


    @property
    def expr_matrix(self):
        return self._expr_matrix

    @expr_matrix.setter
    def expr_matrix(self, matrix):
        self._expr_matrix = matrix

    def main(self):
    '''
    A pipeline?
    '''
        pass





