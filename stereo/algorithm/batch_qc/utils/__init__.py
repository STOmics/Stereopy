#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:14
# @Author  : zhangchao
# @File    : __init__.py.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
# flake8: noqa
from .check_data import check_data
from .generate_palette import generate_palette
from .html_utils import (
    embed_text,
    embed_tabel,
    embed_table_imgs
)
from .pca_lowrank import pca_lowrank
from .print_time import print_time
