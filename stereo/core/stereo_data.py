#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:stereo_data.py
@time:2021/03/22
"""


class StereoData(object):
    def __init__(self, raw_file=None, exp_matrix=None, genes=None, bins=None, position=None, partitions=1):
        self.index = None
        self.index_file = None
        self.exp_matrix = exp_matrix
        self.genes = genes
        self.bins = bins
        self.position = position
        self.raw_file = raw_file
        self.partitions = partitions

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
