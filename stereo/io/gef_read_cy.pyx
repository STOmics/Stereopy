# -*- coding: utf-8 -*-
# distutils: language=c++
# cython: language_level=3, boundscheck=False
# cython: c_string_type=unicode, c_string_encoding=utf8
# Created by huangzhibo on 2021/10/22


import numpy as np
cimport numpy as np
cimport cython
import h5py
from stereo.log_manager import logger

from libcpp.vector cimport vector
from libcpp.string cimport string

ctypedef np.npy_int32 INT32_t
ctypedef np.npy_uint32 UINT32_t
ctypedef np.npy_uint64 UINT64_t

cimport h5py

from .defs cimport *
#
# from ._objects cimport ObjectID
# from ._objects cimport pdefault
# from h5py.h5t cimport TypeID, typewrap, py_create
from h5py.h5s cimport SpaceID
# from .h5p cimport PropID



cimport h5py.h5t

cdef extern from "gef_read.h":
    vector[unsigned long long] cpp_uniq_cell_index(vector[unsigned int] & vec_x,
                        vector[unsigned int] & vec_y,
                        vector[unsigned int] & rows,
                        unsigned long long n_size);

    int cpp_gene_count_index(const vector[int] & gene_count, vector[int] & cols);

def get_uniq_cell(vector[unsigned int] & vec_x, vector[unsigned int] & vec_y, vector[unsigned int] & rows, unsigned long long n_size):
    cdef vector[unsigned long long] cells = cpp_uniq_cell_index(vec_x, vec_y, rows, n_size)
    return cells

def gene_count_index(const vector[int] & gene_count, vector[int] & cols):
    return cpp_gene_count_index(gene_count, cols)


cdef class GEF:
    cdef vector[UINT32_t] cols
    cdef vector[UINT32_t] rows
    cdef vector[string] genes
    cdef int cell_num
    cdef int gene_num
    cdef int bin_size
    cdef UINT64_t exp_size
    cdef string file_path

    def __init__(self, file_path, bin_size):
        self.file_path = file_path
        self.bin_size = bin_size
        self.cell_num = 0
        self.gene_num = 0
        self.exp_size = 0
        self.genes = vector[string]()
        self.cols = vector[UINT32_t]()
        self.rows = vector[UINT32_t]()
        self.build()

    def build(self):
        h5f = h5py.File(self.file_path, mode='r')
        bin_tag = 'bin{}'.format(self.bin_size)
        if bin_tag not in h5f['geneExp'].keys():
            raise Exception('The bin size {} info is not in the GEF file'.format(self.bin_size))

        h5_exp = h5f['geneExp'][bin_tag]['expression']
        h5_gene = h5f['geneExp'][bin_tag]['gene']
        self.genes = np.array(h5_gene['gene'])
        self.gene_num = len(self.genes)
        # self.cols = np.zeros((h5_exp.shape[0],), dtype='uint32')
        # h5_exp_xy = np.array((h5_exp['x'], h5_exp['y']))

        self.cols = np.zeros((self.exp_size,), dtype='uint32')
        # self.cols = vector[UINT32_t](self.exp_size)
        logger.info("gene_count_index start")
        gene_count_index(h5_gene['count'], self.cols)
        logger.info("gene_count_index end")

        # self.rows = vector[UINT32_t](self.exp_size)
        self.rows = np.zeros((self.exp_size,), dtype=np.uintc)
        # exp_x = np.array(h5_exp['x'], )
        # exp_y = np.array(h5_exp['y'])
        logger.info("get_uniq_cell start")
        # get_uniq_cell(h5_exp['x'], h5_exp['y'], self.rows, self.exp_size)
        cells = cpp_uniq_cell_index(h5_exp['x'], h5_exp['y'], self.rows, self.exp_size)
        # self.cells = cells
        logger.info("get_uniq_cell end")
        # self.cell_num = self.cells.size()

    def get_exp_data(self, cell_index, count):
        cpp_uniq_cell_index(self.h5_exp['x'], self.h5_exp['y'], self.rows)
        cdef vector[unsigned long long] uniq_cell =  self.c_h5r.getExpData(cell_index, count)
        return uniq_cell

    def get_exp_len(self):
        return self.c_h5r.getExpLen()

    def get_gene_num(self):
        return self.c_h5r.getGeneNum()

    def get_gene_data(self, gene_index, uniq_genes):
        return self.c_h5r.getGeneData(gene_index, uniq_genes)