#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@rewritten by: qindanhua
@file:tool_base.py
@time:2021/06/15
"""
from stereo.log_manager import logger
from anndata import AnnData
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import inspect


class ToolBase(object):
    """
    A base tool for preparing

    Parameters
    ----------
    data : st data object
        .
    method : .
        the core method of the analysis
    """
    def __init__(
            self,
            data=None,
            method=None,
            name=None
    ):
        self.data = data
        self.method = method.lower() if method else None
        self.name = name if name else self.__class__.__name__

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = self.data_check(data)

    @staticmethod
    def data_check(data):
        """
        check data type
        :return:
        """
        data_type_allowed = (AnnData, pd.DataFrame, np.ndarray)
        if data is not None:
            if not isinstance(data, data_type_allowed):
                logger.error('the format of data must be AnnData or pd.DataFrame.')
                raise ValueError('the format of data must be AnnData or pd.DataFrame.')
        return data

    def extract_exp_matrix(self):
        """
        extract expression data array from input data
        :return: expression data frame [[], [], ..., []]
        """
        # exp_matrix = np.array()
        if isinstance(self.data, AnnData):
            exp_matrix = np.array(self.data.X)
        elif isinstance(self.data, pd.DataFrame):
            exp_matrix = np.array(self.data.values)
        else:
            exp_matrix = np.array(self.data)
        return exp_matrix

    def add_result(self):
        pass

    def merge_result(self):
        pass

    def fit(self):
        pass


if __name__ == '__main__':
    da = pd.DataFrame([[1, 2], [4, 3]], columns=['one', 'two'])
    da1 = pd.DataFrame([[1, 1], [4, 4]], columns=['three', 'four'])
    a = ToolBase(da, 't')
    # a = ToolBase()
    print(a.data)
    c = np.ndarray([2, 3])

    a.data = c
    print(a.data)
