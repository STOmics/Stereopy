#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: qindanhua
@file:stereo_result.py
@time:2021/06/16
"""
from typing import Optional
import numpy as np
import pandas as pd
from stereo.log_manager import logger
from stereo.core.tool_base import ToolBase
from collections import OrderedDict
from typing import Any, MutableMapping, Mapping, Tuple


class StereoResult(object):
    """

    """
    def __init__(
            self,
            # result=None,
            result: MutableMapping[str, Any] = None,
            name: str = None,
            params: Optional[dict] = None,

    ):
        print('log_1')
        self.result = result
        self.params = params
        self.name = name

    # @property
    # def result(self) -> MutableMapping:
    #     print('log_2')
    #     """Unstructured annotation (ordered dictionary)."""
    #     # uns = _overloaded_uns(self)
    #     return self._result
    #
    # @result.setter
    # def result(self, value: [Tuple[str, Any], MutableMapping]):
    #     print('log_3')
    #     if value is not None:
    #         if not isinstance(value, (MutableMapping, Tuple)):
    #             raise ValueError(
    #                 "Only mutable mapping types (e.g. dict) are allowed for `.uns`."
    #             )
    #     self._result = value
    #
    # @result.deleter
    # def result(self):
    #     print('log_4')
    #     self.result = OrderedDict()

    # def set(self, name, value):
    #     self.

    # def add(self, value, name):
    #     self.result[name] = value
    #
    # def delete(self, name):
    #     self.result.pop(name)
    #
    # def to_csv(self):
    #     pass


if __name__ == '__main__':
    n = 'pca'
    r = np.ndarray([2, 3])
    res = StereoResult({n: r})
    print(res.result)
    res.add(r, 'b')
    print(res.result)
    res.delete('b')
    print(res.result)

    # print(res.data)
    # print(res.method)
    # print(res.__str__)

    # class BaseResult(ToolBase):
    #     """
    #
    #     """
    #     def __init__(
    #             self,
    #             data=None,
    #             name: str = 'stereo',
    #             param: Optional[dict] = None
    #     ):
    #         super(BaseResult, self).__init__(data)
    #         self.name = name
    #         self.params = {} if param is None else param
    #
    #     def update_params(self, v):
    #         self.params = v
    #
    #     def __str__(self):
    #         class_info = f'{self.__class__.__name__} of {self.name}. \n'
    #         class_info += f'  params: {self.params}\n'
    #         return class_info
    #
    #     def __repr__(self):
    #         return self.__str__()
