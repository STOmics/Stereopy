# -*- coding: utf-8 -*-
# distutils: language=c++
# cython: language_level=3, boundscheck=False
# cython: c_string_type=unicode, c_string_encoding=utf8
# Created by huangzhibo on 2021/10/22

import numpy as np
cimport numpy as np
cimport cython
import h5py

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "H5Reader.h":
    cdef cppclass H5Reader:
        H5Reader(const string&, int bin_size) except +
        unsigned long long exp_len
        unsigned int minX, minY, maxX, maxY, gene_num

        unsigned long long getExpLen() const
        unsigned int getGeneNum() const

        vector[unsigned long long] getExpData(vector[unsigned int] & cell_index,
                        vector[unsigned int] & count)

        void getGeneData(vector[int] & gene_index, vector[string] & uniq_genes)


cdef class GEF:
    cdef H5Reader* c_h5r  # Hold a C++ instance which we're wrapping
    cdef unsigned long long exp_len
    cdef unsigned int minX, minY, maxX, maxY, gene_num

    h5py.h5d.DatasetID

    # def __cinit__(self, filepath):
    #     self.c_h5r = new H5Reader(filepath)
    #     self.exp_len = self.c_h5r.getExpLen()
    #     self.gene_num = self.c_h5r.getGeneNum()
    #     self.minX = self.c_h5r.minX
    #     self.minY = self.c_h5r.minY
    #     self.maxX = self.c_h5r.maxX
    #     self.maxY = self.c_h5r.maxY

    def __init__(self, filepath, bin_size):
        self.c_h5r = new H5Reader(filepath, bin_size)
        self.exp_len = self.c_h5r.getExpLen()
        self.gene_num = self.c_h5r.getGeneNum()
        self.minX = self.c_h5r.minX
        self.minY = self.c_h5r.minY
        self.maxX = self.c_h5r.maxX
        self.maxY = self.c_h5r.maxY

    def get_exp_data(self, cell_index, count):
        cdef vector[unsigned long long] uniq_cell =  self.c_h5r.getExpData(cell_index, count)
        return uniq_cell

    def get_exp_len(self):
        return self.c_h5r.getExpLen()

    def get_gene_num(self):
        return self.c_h5r.getGeneNum()

    def get_gene_data(self, gene_index, uniq_genes):
        return self.c_h5r.getGeneData(gene_index, uniq_genes)

    def __dealloc__(self):
        del self.c_h5r