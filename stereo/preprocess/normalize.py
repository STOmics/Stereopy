#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:normalize.py
@time:2021/03/05

change log:
    add basic functions of normalization. by Ping Qiu. 2021/03/05
    Refactor the code and add the quantile_norm function. by Ping Qiu. 2021/03/17
    add the zscore_disksmooth function. by Ping Qiu. 2021/05/28
"""
import pandas as pd
from ..log_manager import logger
from ..core.tool_base import ToolBase
from ..algorithm.normalization import normalize_total, quantile_norm, zscore_disksmooth
from typing import Optional


class Normalizer(ToolBase):
    """
    Normalizer of stereo.
    """
    def __init__(
            self,
            data,
            method: Optional[str] = 'normalize_total',
            target_sum: Optional[int] = 1,
            r: Optional[int] = 20
    ):
        super(Normalizer, self).__init__(data=data, method=method)
        self.target_num = target_sum
        self.r = r
        self.position = self.data.position

    @ToolBase.method.setter
    def method(self, method):
        m_range = ['normalize_total', 'quantile', 'zscore_disksmooth']
        self._method_check(method, m_range)

    def fit(self):
        """
        compute the scale value of self.exp_matrix.
        """
        nor_res = None
        self.data.sparse2array()  # TODO: add  normalize of sparseMatrix
        if self.method == 'normalize_total':
            nor_res = normalize_total(self.data.exp_matrix, self.target_num)
        elif self.method == 'quantile':
            nor_res = quantile_norm(self.data.exp_matrix.T)
            nor_res = nor_res.T
        elif self.method == 'zscore_disksmooth':
            nor_res = zscore_disksmooth(self.data.exp_matrix, self.position, self.r)
        else:
            pass
        self.result = pd.DataFrame(nor_res, columns=self.data.gene_names, index=self.data.cell_names)
        return nor_res
