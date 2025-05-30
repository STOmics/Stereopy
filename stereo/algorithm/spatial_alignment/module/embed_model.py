#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 9:42 AM
# @Author  : zhangchao
# @File    : embed_model.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn

from . import EmbeddingLayer
from . import ResidualLayer


class FeatEmbed(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, n_layers=2, act=nn.ELU(), p=0.2):
        super(FeatEmbed, self).__init__()
        self.net = nn.ModuleList([
            EmbeddingLayer(input_dims, output_dims, n_domain, act=act, drop_rate=p)
        ])
        for _ in range(n_layers - 1):
            self.net.append(EmbeddingLayer(output_dims, output_dims, n_domain, act=act, drop_rate=p))
        self.output_dims = output_dims

    def forward(self, x, domain_idx=None):
        for idx, ly in enumerate(self.net):
            x = ly(x, domain_idx)
        return x


class ResidualEmbed(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, n_layers, act, p):
        super(ResidualEmbed, self).__init__()
        self.net = nn.ModuleList([
            ResidualLayer(input_dims, output_dims, n_domain, n_layers=2, act=act, p=p)
        ])

        for _ in range(n_layers - 1):
            self.net.append(ResidualLayer(output_dims, output_dims, n_domain, n_layers=2, act=act, p=0.2))

    def forward(self, x, domain_idx=None):
        for idx, ly in enumerate(self.net):
            x = ly(x, domain_idx)
        return x
