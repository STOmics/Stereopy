#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:stereo_result.py
@time:2021/03/18
"""
from typing import Optional


class StereoResult(object):
    def __init__(self, name: str = 'stereo', param: Optional[dict] = None, data: Optional[dict] = None):
        self.name = name
        self.data = {} if data is None else data
        self.params = {} if param is None else param

    def update_params(self, v):
        self.params = v

    def update_data(self, k, v):
        self.data[k] = v

    def __str__(self):
        class_info = f'{self.__class__.__name__} of {self.name}. \n'
        class_info += f'  params: {self.params}\n'
        for i in self.data:
            class_info += f'  result infomation: \n'
            class_info += f'    {i}: {type(self.data[i])} \n'
        return class_info

    def __repr__(self):
        return self.__str__()


class DimReduceResult(StereoResult):
    def __init__(self, name: str = 'stereo', param: Optional[dict] = None, data: Optional[dict] = None):
        super(DimReduceResult, self).__init__(name, param, data)
        