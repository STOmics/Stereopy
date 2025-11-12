#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 9:44 AM
# @Author  : zhangchao
# @File    : dgi.py
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FeatEmbed, ResidualEmbed, GraphVAE, EmbeddingLayer


class Encoder(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, act, p):
        super().__init__()
        self.reduce = FeatEmbed(
            input_dims=input_dims, output_dims=1024, n_domain=n_domain, n_layers=1, act=act, p=p)
        self.residual = ResidualEmbed(
            input_dims=1024, output_dims=output_dims, n_domain=n_domain, n_layers=2, act=act, p=p)
        self.graph = GraphVAE(output_dims, output_dims)
        self.trans_g = FeatEmbed(output_dims, output_dims, n_domain, 2, act, 0)
        self.trans1 = EmbeddingLayer(output_dims * 2, output_dims, n_domain, act=act, drop_rate=0)
        self.trans2 = EmbeddingLayer(output_dims, output_dims, 0, act=act, drop_rate=0)

    def forward(self, x, edge_index, edge_weight, domain_idx):
        feat_x = self.reduce(x, domain_idx)
        feat_x = self.residual(feat_x, domain_idx)
        feat_g = self.graph(feat_x, edge_index, edge_weight)
        feat_g = self.trans_g(feat_g, domain_idx)
        feat = torch.cat([feat_x, feat_g], dim=1)
        latent_x = self.trans1(feat, domain_idx)
        latent_x = self.trans2(latent_x, domain_idx)
        return latent_x


class DGI(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, act, p):
        super().__init__()
        self.encoder = Encoder(input_dims, output_dims, n_domain, act, p)
        self.dis_ly = nn.Bilinear(output_dims, output_dims, 1)

    def forward(self, x, edge_index, edge_weight, domain_idx, neigh_mask):
        pos_x = self.encoder(x, edge_index, edge_weight, domain_idx)
        graph_loss = 1 / x.size(0) * self.encoder.graph.vgae.kl_loss() + self.encoder.graph.graph_loss(z=pos_x)
        pos_summary = self.readout(pos_x, neigh_mask)

        cor = self.corruption(x, edge_index, edge_weight, domain_idx)
        neg_x = self.encoder(*cor)

        return pos_x, neg_x, pos_summary, graph_loss

    def corruption(self, x, edge_index, edge_weight, domain_idx):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight, domain_idx

    def readout(self, z, neigh_mask):
        v_sum = torch.mm(neigh_mask, z)
        r_sum = torch.sum(neigh_mask, 1)
        r_sum = r_sum.expand((v_sum.shape[1], r_sum.shape[0])).T
        global_z = v_sum / r_sum
        global_z = F.normalize(global_z, p=2, dim=1)
        return global_z

    def discriminate(self, z, summary, sigmoid=True):
        # print(f"z = {z}")
        # print(f"summary = {summary}")
        value = self.dis_ly(z, summary)
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_x, neg_x, summary):
        pos_loss = -torch.log(self.discriminate(pos_x, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_x, summary, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss
