#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 9:48
# @Author  : zhangchao
# @File    : early_stop.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import numpy as np


class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop_flag = False
        self.loss_min = np.Inf

    def __call__(self, loss):
        if np.isnan(loss):
            self.stop_flag = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.loss_min = loss
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.best_score = score
            self.counter = 0
            self.loss_min = loss
