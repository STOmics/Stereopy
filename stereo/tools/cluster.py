#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:cluster.py
@time:2021/03/19
"""
from ..core.tool_base import ToolBase


class Cluter(ToolBase):
    def __init__(self, data, method, inplace=False, name=None):
        super(Cluter, self).__init__(data, method, inplace=inplace, name=name)
