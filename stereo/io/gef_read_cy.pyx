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

cdef extern from "gef_read.h":
    int cpp_uniq_cell_index(const unsigned int * vec_x,
                        const unsigned int * vec_y,
                        unsigned int * rows,
                        unsigned long long len);

    int cpp_gene_count_index(const vector[int] & gene_count, vector[int] & cols);

cdef get_uniq_cell(np.ndarray[UINT32_t, ndim=1] vec_x, np.ndarray[UINT32_t, ndim=1] vec_y, np.ndarray[UINT32_t, ndim=1] rows, unsigned long long len):
    return cpp_uniq_cell_index(<unsigned int*> vec_x.data, <unsigned int*> vec_y.data, <unsigned int*> rows.data, len)

def gene_count_index(const vector[int] & gene_count, vector[int] & cols):
    return cpp_gene_count_index(gene_count, cols)


cdef class GEF:
    cdef UINT32_t [:] cols
    cdef UINT32_t [:] rows
    cdef UINT64_t [:] cells
    # cdef vector[UINT32_t] cols
    # cdef vector[UINT32_t] rows
    # cdef vector[UINT64_t] cells
    cdef vector[string] genes
    cdef int cell_num
    cdef int gene_num
    cdef int bin_size
    cdef UINT64_t exp_size
    cdef string file_path

    def __init__(self, file_path, bin_size):
        self.file_path = file_path
        self.bin_size = bin_size
        # self.genes = None
        self.cell_num = 0
        self.gene_num = 0
        self.exp_size = 0
        # self.cols = None
        # self.rows = None
        # self.cells = None
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

        self.cols = np.zeros((self.exp_size,), dtype=np.uintc)
        # self.cols = vector[UINT32_t](self.exp_size)
        logger.info("gene_count_index start")
        gene_count_index(h5_gene['count'], self.cols)
        logger.info("gene_count_index end")

        # self.rows = vector[UINT32_t](self.exp_size)
        self.rows = np.zeros((self.exp_size,), dtype=np.uintc)
        exp_x = np.array(h5_exp['x'])
        exp_y = np.array(h5_exp['y'])
        logger.info("get_uniq_cell start")
        self.cells = get_uniq_cell(exp_x, exp_y, self.rows, self.exp_size)
        logger.info("get_uniq_cell end")
        self.cell_num = self.cells.size()
