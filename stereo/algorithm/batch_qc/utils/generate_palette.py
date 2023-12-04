#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:32
# @Author  : zhangchao
# @File    : generate_palette.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from distinctipy import distinctipy


def generate_palette(category_nums: int):
    """
    generate color map
    :param category_nums: [int] Number of generated color maps
    :return: The generated list of colors.
    """
    palette = [distinctipy.get_hex(c) for c in distinctipy.get_colors(category_nums, n_attempts=1000, rng=42)]
    return palette
