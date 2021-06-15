#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:stereo_exp_data.py
@time:2021/03/22
"""
from .data import Data
import pandas as pd
import numpy as np


class StereoExpData(Data):
    def __init__(self, file_path, file_format, partitions=1):
        super(StereoExpData, self).__init__(file_path=file_path, file_format=file_format, partitions=partitions)
        self.exp_matrix = None
        self.genes = None
        self.bins = None
        self.position = None

    @property
    def genes(self):
        return self.genes

    @genes.setter
    def genes(self, g_array):
        self.genes = g_array

    def read(self):
        if self.file_format == 'bins':
            self.read_bins()
        else:
            self.read_cell_bins()

    def write(self):
        pass

    def read_bins(self):
        pass

    def read_cell_bins(self):
        pass

    def filter_genes(self):
        pass

    def filter_bins(self):
        pass

    def search(self):
        pass

    def combine_bins(self, bin_size, step):
        pass

    def select_by_genes(self, gene_list):
        pass

    def select_by_position(self, x_min, y_min, x_max, y_max, bin_size):
        pass

    def transform_matrix(self):
        pass

    def get_genes(self):
        pass

    def get_bins(self):
        pass

    def split_data(self):
        pass

    def sparse2array(self):
        pass
