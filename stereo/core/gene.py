#!/usr/bin/env python3
# coding: utf-8
"""
@file: gene.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/29  create file.
"""
from typing import Optional
import numpy as np


class Gene(object):
    def __init__(self, gene_name: Optional[np.ndarray]):
        self._gene_name = gene_name if gene_name is None else gene_name.astype('U')
        self.n_cells = None

    @property
    def gene_name(self):
        return self._gene_name

    @gene_name.setter
    def gene_name(self, name):
        if not isinstance(name, np.ndarray):
            raise TypeError('gene name must be a np.ndarray object.')
        self._gene_name = name.astype('U')

    def sub_set(self, index):
        if self.gene_name is not None:
            self.gene_name = self.gene_name[index]
        if self.n_cells is not None:
            self.n_cells = self.n_cells[index]
        return self
