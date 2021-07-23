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
from scipy.sparse import issparse
import functools
import time
from typing import Optional, Union
import requests
import os
from ..plots.scatter import plot_multi_scatter, plt


class ToolBase(object):
    """
    A base tool preparing data for analysis tools

    Parameters
    ----------

    :param: data : expression matrix, a StereoExpData or pandas.Dataframe object. matrix format is:
            gene_1  gene_2  gene_3
    cell_1       0       1       0
    cell_2       1       3       0
    cell_3       2       2       2
    :param groups: group information matrix, at least two columns, treat first column as sample name,
    and the second as group name e.g
    >>> pd.DataFrame({'bin_cell': ['cell_1', 'cell_2'], 'cluster': ['1', '2']})
      bin_cell cluster
    0   cell_1       1
    1   cell_2       2
    :param: method : the core method of the analysis
    """
    def __init__(
            self,
            data: Optional[Union[StereoExpData, pd.DataFrame]] = None,
            groups: Optional[Union[StereoResult, pd.DataFrame]] = None,
            method: str = 'stereo',
    ):
        self.data = data
        self.groups = groups
        self._method = method
        self.result = StereoResult()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = self._data_check(data)

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, groups):
        self._groups = self._group_check(groups)

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
                raise ValueError('the format of data must be StereoExpData or pd.DataFrame.')
        if isinstance(data, pd.DataFrame):
            st_data = StereoExpData(
                exp_matrix=data.values,
                cells=np.array(data.index),
                genes=np.array(data.columns)
            )
        else:
            st_data = data
        return st_data

    def _group_check(self, groups):
        if groups is None:
            pass
        else:
            if not isinstance(groups, pd.DataFrame):
                raise ValueError(f'the format of group data must be pd.DataFrame.')
            group_index = groups.index
            if list(group_index) == list(self.data.cell_names):
                logger.info(f'read group information, grouping by {groups.columns[0]} column.')
                return groups
            else:
                cells = groups.iloc[:, 0].values
                if not list(cells) == list(self.data.cell_names):
                    raise ValueError(f'cell index is not match')
                else:
                    logger.info(f'read group information, grouping by {groups.columns[1]} column.')
                    group_info = pd.DataFrame({'group': groups.iloc[:, 1].values}, index=cells)
                return group_info

    def extract_exp_matrix(self):
        """
        extract expression data array from input data
        :return: expression data frame [[], [], ..., []]
        """
        return self.data.exp_matrix.toarray() if issparse(self.data.exp_matrix) else self.data.exp_matrix

    def sparse2array(self):
        """
        transform expression matrix to array if it is parse matrix
        :return:
        """
        if issparse(self.data.exp_matrix):
            self.data.exp_matrix = self.data.exp_matrix.toarray()

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

    @staticmethod
    def download_ref(ref_dir):
        logger.info("downloading reference expression matrix")
        url = 'https://raw.githubusercontent.com/molindoudou/bio_tools/main/data/FANTOM5/ref_sample_epx.csv'
        url2 = 'https://raw.githubusercontent.com/molindoudou/bio_tools/main/data/FANTOM5/cell_map.csv'
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        for u in [url, url2]:
            try:
                r = requests.get(u)
                file_name = u.split('/')[-1]
                with open(os.path.join(ref_dir, file_name), "wb") as code:
                    code.write(r.content)
            except IOError:
                logger.error(f'can not download reference file from {u}')
        logger.info('download reference matrix done')

    @classmethod
    def check_fit(cls, func):
        @functools.wraps(func)
        def wrapper(self, *args, **kw):
            if self.data is None:
                raise ValueError(f'data must be set if running fit()')
            logger.info('start running {}'.format(time.asctime(time.localtime(time.time()))))
            return func(self, *args, **kw)
        logger.info('end running {}'.format(time.asctime(time.localtime(time.time()))))
        return wrapper

    def plot_top_gene_scatter(self, file_path=None):
        df = pd.DataFrame(self.data.exp_matrix, columns=self.data.gene_names, index=self.data.cell_names)
        sum_top_genes = list(df.sum().sort_values(ascending=False).index[:3])
        plot_multi_scatter(self.data.position[:, 0], self.data.position[:, 1],
                           color_values=np.array(df[sum_top_genes]).T,
                           color_bar=True, ncols=2)
        if file_path:
            plt.savefig(file_path)

    def add_result(self):
        pass

    def merge_result(self):
        pass

    def fit(self):
        pass

