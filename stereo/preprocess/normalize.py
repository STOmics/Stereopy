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


class Normalizer(ToolBase):
    """
    Normalizer of stereo.
    """
    def __init__(
            self,
            data,
            method='normalize_total',
            inplace=True,
            target_sum=1,
            # name='normalize',
            r=20
    ):
        super(Normalizer, self).__init__(data=data, method=method)
        self.target_num = target_sum
        self.inplace = inplace
        self.check_param()
        self.position = data.obsm['spatial']
        self.r = r
        self._method = method

    def check_param(self):
        """
        Check whether the parameters meet the requirements.

        """
        if self.method.lower() not in ['normalize_total', 'quantile', 'zscore_disksmooth']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, v):
        if v.lower() not in ['normalize_total', 'quantile', 'zscore_disksmooth']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')
        self._method = v

    def fit(self):
        """
        compute the scale value of self.exp_matrix.
        """
        nor_res = None
        self.sparse2array()  # TODO: add  normalize of sparseMatrix
        if self.method == 'normalize_total':
            nor_res = normalize_total(self.data.exp_matrix, self.target_num)
        elif self.method == 'quantile':
            nor_res = quantile_norm(self.data.exp_matrix.T)
            nor_res = nor_res.T
        elif self.method == 'zscore_disksmooth':
            nor_res = zscore_disksmooth(self.data.exp_matrix, self.position, self.r)
        else:
            pass
        # if nor_res is not None and self.inplace and isinstance(self.data, AnnData):
        #     self.data.X = nor_res
        self.result.matrix = pd.Dataframe(nor_res, columns=self.data.genes, index=self.data.cells)
        return nor_res
