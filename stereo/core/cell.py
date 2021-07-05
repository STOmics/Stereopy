#!/usr/bin/env python3
# coding: utf-8
"""
@file: cell.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/29  create file.
"""
from typing import Optional

import numpy as np


class Cell(object):
    def __init__(self, cell_name: Optional[np.ndarray] = None):
        self._cell_name = cell_name
        self.total_count = None
        self.pct_counts_mt = None
        self.n_gene_by_counts = None

    @property
    def cell_name(self):
        return self._cell_name

    @cell_name.setter
    def cell_name(self, name):
        if not isinstance(name, np.ndarray):
            raise TypeError('cell name must be a np.ndarray object.')
        self._cell_name = name
