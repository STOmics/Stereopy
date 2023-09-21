#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 11:02
# @Author  : zhangchao
# @File    : check_data.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from anndata import AnnData


def check_data(data: AnnData):
    # test that the count matrix contains valid inputs. More precisely, test that inputs inputs are non-negative integers. # noqa
    if (data.X.data % 1 != 0).any().any():
        raise ValueError("The count matrix should only contain integers.")
    if (data.X.data < 0).any().any():
        raise ValueError("The count matrix should only contain non-negative values.")
