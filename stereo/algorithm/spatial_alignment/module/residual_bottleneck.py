#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 6:04 PM
# @Author  : zhangchao
# @File    : residual_bottleneck.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn

from . import EmbeddingLayer


class ResidualLayer(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, n_layers=2, act=nn.LeakyReLU(), p=0.2):
        super(ResidualLayer, self).__init__()
        self.trans = EmbeddingLayer(input_dims, output_dims, n_domain, act, p) if input_dims != output_dims else None
        self.net = nn.ModuleList([
            EmbeddingLayer(input_dims, output_dims, n_domain, act=act, drop_rate=p)
        ])
        for _ in range(n_layers - 1):
            self.net.append(EmbeddingLayer(output_dims, output_dims, n_domain, act=act, drop_rate=p))

        self.out_ly = act

    def forward(self, x, domain_idx=None):
        if self.trans is not None:
            identity = self.trans(x, domain_idx)
        else:
            identity = x
        for idx, ly in enumerate(self.net):
            x = ly(x, domain_idx)

        return self.out_ly(identity + x)
