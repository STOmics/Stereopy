#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 5:59 PM
# @Author  : zhangchao
# @File    : domain_specific_batch_norm.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn


class DomainSpecificBN1d(nn.Module):
    def __init__(self, feat_dims, n_domain, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DomainSpecificBN1d, self).__init__()
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(feat_dims, eps, momentum, affine, track_running_stats) for _ in range(n_domain)
        ])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_idx):
        return self.bns[domain_idx[0]](x)