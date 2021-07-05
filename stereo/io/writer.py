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


def write_h5ad(data, output=None):
    """
    write the data as a h5ad file.

    :param data: the StereoExpData object.
    :param output: the output path. StereoExpData's output will be reset if the output is not None.
    :return:
    """
    if not isinstance(data, StereoExpData):
        raise TypeError
    if output is not None:
        data.output = output
    data.write_h5ad()
