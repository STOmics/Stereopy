#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 6:00 PM
# @Author  : zhangchao
# @File    : embed_layer.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn

from . import DomainSpecificBN1d


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, act=nn.ELU(), drop_rate=0.2):
        super(EmbeddingLayer, self).__init__()
        self.fc = nn.Linear(input_dims, output_dims)
        if n_domain == 1:
            self.bns = nn.BatchNorm1d(output_dims)
        elif n_domain > 1:
            self.bns = DomainSpecificBN1d(output_dims, n_domain)
        else:
            self.bns = None

        if act:
            self.ac = act
        else:
            self.ac = None

        if drop_rate > 0:
            self.drop = nn.Dropout(p=drop_rate)
        else:
            self.drop = None

    def forward(self, x, domain_idx=None):
        output = self.fc(x)
        if self.bns:
            if self.bns.__class__.__name__ == "BatchNorm1d":
                output = self.bns(output)
            elif self.bns.__class__.__name__ == "DomainSpecificBN1d":
                output = self.bns(output, domain_idx)
            else:
                raise AttributeError

        if self.ac:
            output = self.ac(output)

        if self.drop:
            output = self.drop(output)

        return output
