#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:__init__.py.py
@time:2021/03/05
"""
import shutil
import os
from .correlation import pearson_corr, spearmanr_corr
# from .data_helper import select_group


def remove_file(path):
    if os.path.isfile(path):
        os.remove(path)
    if os.path.isdir(path):
        shutil.rmtree(path)
