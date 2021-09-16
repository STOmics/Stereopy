#!/usr/bin/env python3
# coding: utf-8
"""
@file: phenograph.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/08/27  create file.
"""
import phenograph
import numpy as np


def run_phenograph(x: np.ndarray, phenograph_k: int):
    communities, _, _ = phenograph.cluster(x, k=phenograph_k, clustering_algo='leiden')
    cluster = communities.astype(str)
    return cluster
