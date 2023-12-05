#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 9:48
# @Author  : zhangchao
# @File    : loss.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCEFocalLoss(nn.Module):
    def __init__(self, n_batch, gamma=2, alpha=.25, reduction="mean"):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = 1.
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.reduction = reduction
        self.n_batch = n_batch

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        batch_mask = F.one_hot(target, self.n_batch)
        prob = (pt * batch_mask).sum(1).view(-1, 1)
        log_p = prob.log()
        loss = -self.alpha * (torch.pow(1 - prob, self.gamma)) * log_p

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
