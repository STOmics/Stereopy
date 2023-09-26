#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:__init__.py.py
@time:2021/03/05
"""
# Ignore errors for entire file
# flake8: noqa

import os
import shutil

from .correlation import pearson_corr
from .correlation import spearmanr_corr
from .pipeline_utils import cluster_bins_to_cellbins


def remove_file(path):
    if os.path.isfile(path):
        os.remove(path)
    if os.path.isdir(path):
        shutil.rmtree(path)
