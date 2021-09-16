#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:stereo_result.py
@time:2021/03/14

change log:
    rewritten by: qindanhua. 2021/06/15
"""
import numpy as np
import pandas as pd
from stereo.log_manager import logger
from typing import Optional
# from stereo.core.tool_base import ToolBase
# from collections import OrderedDict
# from typing import Any, MutableMapping, Mapping, Tuple


class StereoResult(object):
    """
    analysis result
    :param matrix: main result data frame
    :param name: analysis tool name

    """
    def __init__(
            self,
            matrix: Optional[pd.DataFrame] = None,
            name: str = '',

    ):
        self.name = name
        self._matrix = matrix
        self._cols = self._get_cols()

    def _get_cols(self):
        return [] if self.matrix is None else [str(i) for i in self.matrix.columns]

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if not isinstance(matrix, pd.DataFrame):
            logger.error(f'result data needs to be a Dataframe.')
        else:
            self._matrix = matrix

    @property
    def is_empty(self):
        """
        check if the matrix is empty
        :return: bool
        """
        return True if self.matrix is None else self.matrix.empty

    def __str__(self):
        self._cols = self._get_cols()
        describe_cols = ','.join(self._cols)
        class_info = f'{self.__class__.__name__} result of stereo tool {self.name},'
        class_info += f'a DataFrame which has {describe_cols} columns. \n'
        class_info += f'the shape is {self.matrix.shape if isinstance(self.matrix, pd.DataFrame) else None} \n'
        return class_info

    def __repr__(self):
        return self.__str__()

    def top_n(self, sort_key, top_n=10, ascend=False):
        """
        obtain the first k significantly different genes

        :param top_n:  the number of top n
        :param sort_key: sort by this column
        :param ascend: the ascend order of sorting.
        :return:
        """
        if self.matrix is not None:
            top_n_data = self.matrix.sort_values(by=sort_key, ascending=ascend).head(top_n)
            return top_n_data
        else:
            logger.warning('the analysis result data is None, return None.')
            return None

    def check_columns(self, cols):
        """
        check if column in matrix
        :param cols: column names, ['col1', 'cols2']
        :return: bool
        """
        cols_m = self.matrix.columns
        if len(set(cols_m) & set(cols)) == len(cols):
            return True
        else:
            return False


class SpatialLagResult(StereoResult):
    def __init__(self, matrix=None, name='spatial_lag'):
        super(SpatialLagResult, self).__init__(matrix=matrix, name=name)

    def top_markers(self, top_k=10, ascend=False):
        """
        obtain the first k significantly different genes

        :param top_k:  the number of top k
        :param ascend: the ascend order of sorting.
        :return:
        """
        if self.matrix is not None:
            coef_col = self.matrix.columns[self.matrix.columns.str.endswith('lag_coeff')]
            col1 = np.array([(i, i) for i in coef_col]).flatten()
            col2 = np.array([('genes', 'values') for _ in coef_col]).flatten()
            tmp = []
            for col in coef_col:
                top_df = self.matrix.sort_values(by=col, ascending=ascend).head(top_k)
                tmp.append(list(top_df.index))
                tmp.append(top_df[col])
            x = np.array(tmp).T
            top_res = pd.DataFrame(x, columns=[col1, col2], index=np.arange(top_k))
            return top_res
        else:
            logger.warning('the result of degs is None, return None.')
            return None
