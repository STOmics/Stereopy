#!/usr/bin/env python3
# coding: utf-8
"""
@file: spmatrix_helper.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/24  create file.
"""


def idx_chunks_along_axis(shape: tuple, axis: int, chunk_size: int):
    """
    Gives indexer tuples chunked along an axis.

    :param shape: Shape of array to be chunked
    :param axis: Axis to chunk along
    :param chunk_size: Size of chunk along axis
    :return: An iterator of tuples for indexing into an array of passed shape.
    """
    total = shape[axis]
    cur = 0
    mutable_idx = [slice(None) for i in range(len(shape))]
    while cur + chunk_size < total:
        mutable_idx[axis] = slice(cur, cur + chunk_size)
        yield tuple(mutable_idx)
        cur += chunk_size
    mutable_idx[axis] = slice(cur, None)
    yield tuple(mutable_idx)
