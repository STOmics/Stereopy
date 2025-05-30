#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 6:02 PM
# @Author  : zhangchao
# @File    : graph_vae.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling


class GraphEncoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(GraphEncoder, self).__init__()
        self.gcn = GCNConv(input_dims, output_dims, improved=True)
        self.mu = GCNConv(output_dims, output_dims, improved=True)
        self.var = GCNConv(output_dims, output_dims, improved=True)

    def forward(self, x, edge_index, edge_weight):
        feat_x = self.gcn(x, edge_index, edge_weight).relu_()
        mu = self.mu(feat_x, edge_index, edge_weight)
        var = self.var(feat_x, edge_index, edge_weight)
        return mu, var


class GraphVAE(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(GraphVAE, self).__init__()
        self.edge_index = None
        self.edge_weight = None
        self.feat_x = None
        self.vgae = VGAE(GraphEncoder(input_dims, output_dims))

    def forward(self, x, edge_index, edge_weight):
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.feat_x = self.vgae.encode(x, edge_index, edge_weight)
        return self.feat_x

    def graph_loss(self, z=None, edge_index=None, edge_weight=None):
        assert self.feat_x is not None and self.edge_index is not None and self.edge_weight is not None
        z = self.feat_x if z is None else z
        edge_index = self.edge_index if edge_index is None else edge_index
        edge_weight = self.edge_weight if edge_weight is None else edge_weight

        pos_dec = self.vgae.decoder(z, edge_index, sigmoid=False)
        pos_loss = F.binary_cross_entropy_with_logits(pos_dec, edge_weight)
        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_dec = self.vgae.decoder(z, neg_edge_index, sigmoid=False)
        neg_loss = -F.logsigmoid(-neg_dec).mean()
        return pos_loss + neg_loss
