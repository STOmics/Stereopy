#!/usr/bin/env python3
# coding: utf-8
"""
@author: Leying Wang wangleying@genomics.cn
@last modified by: Leying Wang
@file: spatial_pattern_score.py
@time: 2021/4/19 17:20

change log:
    2021/05/20 rst supplement. by: qindanhua.
    2021/06/20 adjust for restructure base class . by: qindanhua.
"""

import pandas as pd
from ..core.tool_base import ToolBase


class SpatialPatternScore(ToolBase):
    """
    calculate spatial pattern score
    """
    def __init__(
            self,
            data=None,
            method='enrichment',
    ):
        super(SpatialPatternScore, self).__init__(data=data, method=method)

    def fit(self):
        """
        run
        """
        self.data.sparse2array()
        df = pd.DataFrame(self.data.exp_matrix, columns=self.data.gene_names, index=self.data.cell_names)
        result = self.get_func_by_path('stereo.algorithm.spatial_pattern_score', 'spatial_pattern_score')(df)
        self.result = result

    def plot(self):
        pass
