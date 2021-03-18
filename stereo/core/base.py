#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:base.py
@time:2021/03/17
"""
from stereo.log_manager import logger
from anndata import AnnData
import pandas as pd
from scipy.sparse import issparse
from .stereo_result import StereoResult
from typing import Any


class Base(object):
    def __init__(self, data, method, inplace=False):
        self.data = data
        self.method = method
        self.inplace = inplace
        self.exp_matrix = self.data.values if isinstance(self.data, pd.DataFrame) else self.data.X.copy()

    def check_param(self):
        if not isinstance(self.data, AnnData) or not isinstance(self.data, pd.DataFrame):
            logger.error('the format of data must be AnnData or pd.DataFrame.')
            raise ValueError('the format of data must be AnnData or pd.DataFrame.')

    def sparse2array(self):
        """
        transform `self.exp_matrix` to np.array if it is sparseMatrix.`
        :return:
        """
        if issparse(self.exp_matrix):
            self.exp_matrix = self.exp_matrix.toarray()
        return self.exp_matrix

    def fit(self):
        pass

    def add_result(self, analysis: str, params: dict, res_data: Any, key_added: str = 'stereo'):
        """
        add the result of analysis into  StereoResult, which is be added in AnnData.uns[key_added]
        :param analysis: the name of analysis.
        :param params: the parameter dict of analysis.
        :param res_data: the result data of analysis.
        :param key_added: the name of AnnData.uns for the StereoResult
        :return:
        """
        if not isinstance(self.data, AnnData):
            return
        result = StereoResult() if key_added not in self.data.uns.keys() else self.data.uns[key_added]
        if not isinstance(result, StereoResult):
            logger.warning(f"your AnnData.uns['{key_added}'] is used by others, replacing it with a StereoResult.")
            result = StereoResult()
        result.add(res_key=analysis, res_data=res_data)
        result.update_params(k=analysis, v=params)
        self.data.uns[key_added] = result
