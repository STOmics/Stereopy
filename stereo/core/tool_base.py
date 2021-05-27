#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:tool_base.py
@time:2021/03/17
"""
from stereo.log_manager import logger
from anndata import AnnData
import pandas as pd
from scipy.sparse import issparse
import inspect


class ToolBase(object):
    def __init__(self, data, method, name=None):
        self.data = data
        self.method = method.lower()
        self.exp_matrix = self.data.values if isinstance(self.data, pd.DataFrame) else self.data.X.copy()
        self.name = name if name else self.__class__.__name__

    def check_param(self):
        if not isinstance(self.data, AnnData) and not isinstance(self.data, pd.DataFrame):
            logger.error('the format of data must be AnnData or pd.DataFrame.')
            raise ValueError('the format of data must be AnnData or pd.DataFrame.')

    def sparse2array(self):
        """
        transform `self.exp_matrix` to np.array if it is sparseMatrix.`

        """
        if issparse(self.exp_matrix):
            self.exp_matrix = self.exp_matrix.toarray()
        return self.exp_matrix

    def fit(self):
        pass

    def get_params(self, vars_info):
        params = {}
        for arg in inspect.signature(self.__init__).parameters:
            params[arg] = vars_info[arg]
        return params

    def add_result(self, result, key_added: str = 'stereo'):
        """
        add the result of analysis into anndata , which is be added in AnnData.uns[key_added]

        :param result: the result of analysis.
        :param key_added: the name of AnnData.uns for the StereoResult
        :return: anndata object adding tool's results
        """
        if not isinstance(self.data, AnnData):
            logger.error('the data must be a AnnData.')
            raise
        # if not isinstance(result, StereoResult):
        #     logger.warning("the result must be a StereoResult, please check.")
        #     raise
        if key_added in self.data.uns.keys():
            logger.warning(f"your AnnData.uns['{key_added}'] is used by others, replacing it with a StereoResult.")
        self.data.uns[key_added] = result
