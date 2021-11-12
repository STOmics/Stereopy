# -*- coding: utf-8 -*-
# distutils: language=c++
# cython: language_level=3, boundscheck=False
# cython: c_string_type=unicode, c_string_encoding=utf8
# Created by huangzhibo on 2021/10/22
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "H5Reader.h":
    cdef cppclass H5Reader:
        H5Reader(const string&, int bin_size) except +
        unsigned long long exp_len
        unsigned int minX, minY, maxX, maxY, gene_num, cell_num

        unsigned long long getExpLen() const
        unsigned int getGeneNum() const

        vector[unsigned long long] getExpData(unsigned int * cell_index, unsigned int * count)
        void getGeneData(unsigned int * gene_index, vector[string] & uniq_genes)


cdef class GEF:
    cdef H5Reader* c_h5r  # Hold a C++ instance which we're wrapping
    cdef unsigned long long exp_len
    cdef unsigned int minX, minY, maxX, maxY, gene_num

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

    def get_exp_data(self):
        cdef unsigned int[::1] cell_index = np.empty(self.exp_len, dtype=np.uint32)
        cdef unsigned int[::1] count = np.empty(self.exp_len, dtype=np.uint32)
        cdef vector[unsigned long long] uniq_cell =  self.c_h5r.getExpData(&cell_index[0], &count[0])
        return np.asarray(uniq_cell), np.asarray(cell_index), np.asarray(count)

    def get_exp_len(self):
        return self.c_h5r.getExpLen()

    def get_gene_num(self):
        return self.c_h5r.getGeneNum()

    def get_gene_data(self):
        cdef vector[string] uniq_genes
        uniq_genes.reserve(self.gene_num)
        cdef unsigned int[::1] gene_index = np.empty(self.exp_len, dtype=np.uint32)
        self.c_h5r.getGeneData(&gene_index[0], uniq_genes)
        return np.asarray(gene_index), np.asarray(uniq_genes)

    def __dealloc__(self):
        del self.c_h5r