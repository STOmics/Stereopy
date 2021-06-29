#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:tool_base.py
@time:2021/03/14

change log:
    rewritten by: qindanhua. 2021/06/15
"""
from stereo.log_manager import logger
import pandas as pd
import numpy as np
from ..core.stereo_result import StereoResult
from ..core.stereo_exp_data import StereoExpData
# from scipy.sparse import issparse
# import inspect
from typing import Optional


class ToolBase(object):
    """
    A base tool preparing data for analysis tools

    Parameters
    ----------
    data : expression matrix, a StereoExpData or pandas.Dataframe object. matrix format is:

            gene_1  gene_2  gene_3
    cell_1       0       1       0
    cell_2       1       3       0
    cell_3       2       2       2
    cell_4       0       0       0
    cell_5       3       3       3
    cell_6       4       0       1

    method : .
        the core method of the analysis
    """
    def __init__(
            self,
            data=None,
            method: str = None,
    ):
        self.data = data
        self._method = method.lower()
        self.result = StereoResult()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = self._data_check(data)

    @property
    def method(self):
        return self._method

    def _method_check(self, method, method_range):
        if method.lower() not in method_range:
            logger.error(f'method range in {method_range}')
            logger.error(f'{method} is out of range, please check.')
            self._method = self.method
        else:
            self._method = method
        return method.lower() in method_range

    @staticmethod
    def _params_range_check(value, value_min, value_max, value_type):
        if not isinstance(value, value_type):
            logger.error(f'{value} should be {value_type} type')
            return False
        if value < value_min or value > value_max:
            logger.error(f'{value} should be range in [{value_min}, {value_max}] type')
            return False

    @staticmethod
    def _data_check(data):
        """
        check data type
        :return:
        """
        data_type_allowed = (pd.DataFrame, StereoExpData)
        if data is not None:
            if not isinstance(data, data_type_allowed):
                logger.error('the format of data must be StereoExpData or pd.DataFrame.')
        if isinstance(data, pd.DataFrame):
            st_data = StereoExpData(
                exp_matrix=data.values,
                cells=pd.DataFrame(data.index),
                genes=pd.DataFrame(data.columns)
            )
        else:
            st_data = data
        return st_data

    @property
    def cell_names(self):
        return list(self.data.cells[0].values)

    def extract_exp_matrix(self):
        """
        extract expression data array from input data
        :return: expression data frame [[], [], ..., []]
        """
        return self.data.exp_matrix

    @staticmethod
    def _check_input_data(input_data):
        if input_data is None:
            input_df = StereoResult()
        else:
            if not isinstance(input_data, (ToolBase, StereoResult, pd.DataFrame, np.ndarray)):
                logger.error('the format of data must be AnnData or pd.DataFrame.')
            if isinstance(input_data, StereoResult):
                input_df = input_data
            elif isinstance(input_data, ToolBase):
                input_df = input_data.result
            elif isinstance(input_data, pd.DataFrame):
                input_df = StereoResult(input_data)
            else:
                input_df = StereoResult(pd.DataFrame(input_data))
        return input_df

    def add_result(self):
        pass

    def merge_result(self):
        pass

    def fit(self):
        pass

