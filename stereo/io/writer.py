#!/usr/bin/env python3
# coding: utf-8
"""
@file: writer.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/07/05  create file.
"""
from ..core.stereo_exp_data import StereoExpData
from ..log_manager import logger
from scipy.sparse import csr_matrix, issparse
import h5py
from stereo.io import h5ad
import pickle

def write_h5ad(data):
    """
    write the SetreoExpData into h5ad file.
    :return:
    """
    if data.output is None:
        logger.error("the output path must be set before writting.")
    with h5py.File(data.output, mode='w') as f:
        h5ad.write(data.genes, f, 'genes')
        h5ad.write(data.cells, f, 'cells')
        h5ad.write(data.position, f, 'position')
        sp_format = 'csr' if isinstance(data.exp_matrix, csr_matrix) else 'csc'
        if issparse(data.exp_matrix):
            h5ad.write(data.exp_matrix, f, 'exp_matrix', sp_format)
        else:
            h5ad.write(data.exp_matrix, f, 'exp_matrix')
        h5ad.write(data.bin_type, f, 'bin_type')


def write(data, output=None, output_type='h5ad'):
    """
    write the data as a h5ad file.

    :param data: the StereoExpData object.
    :param output: the output path. StereoExpData's output will be reset if the output is not None.
    :param output: the output type. StereoExpData's output will be written in output_type. Default setting is h5ad.
    :return:
    """
    if not isinstance(data, StereoExpData):
        raise TypeError
    if output is not None:
        data.output = output
        if output_type == 'h5ad':
            write_h5ad(data)


def save_pkl(obj, output):
    with open(output, "wb") as f:
        pickle.dump(obj, f)
    f.close()
