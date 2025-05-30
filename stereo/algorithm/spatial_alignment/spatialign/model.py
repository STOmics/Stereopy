#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 9:50 AM
# @Author  : zhangchao
# @File    : spatialign.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
from abc import ABC
from collections import defaultdict

import numpy as np
import random
import torch
import torch.nn as nn
# from torch import device

from ..module import DGI, EmbeddingLayer, scale_mse
from ..utils import get_format_time, get_running_time, Dataset


class DGIAlignment(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, act, p):
        super().__init__()
        self.dgi = DGI(input_dims, output_dims, n_domain, act, p)
        self.decoder = EmbeddingLayer(
            input_dims=output_dims, output_dims=input_dims, n_domain=n_domain, act=act, drop_rate=p)

    def forward(self, x, edge_index, edge_weight, domain_idx, neigh_mask):
        latent_x, neg_x, pos_summary, graph_loss = self.dgi(x, edge_index, edge_weight, domain_idx, neigh_mask)
        dgi_loss = self.dgi.loss(latent_x, neg_x, pos_summary)
        recon_x = self.decoder(latent_x, domain_idx)
        recon_loss = scale_mse(recon_x=recon_x, x=x)
        return graph_loss, dgi_loss, recon_loss, latent_x, recon_x


class Base(ABC):
    ckpt_path: str = None
    model: DGIAlignment = None
    dataset: Dataset = None
    device: torch.device = None
    header_bank: defaultdict = None

    def set_seed(self, seed=42, n_thread=24):
        torch.set_num_threads(n_thread)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def set_device(self, gpu):
        if torch.cuda.is_available() and gpu is not None:
            if (float(gpu) % 2 == 0) and float(gpu) >= 0:
                device = torch.device(f"cuda:{gpu}")
            else:
                print(f"{get_format_time}  Got an invalid GPU device ids, can not using GPU device to training...")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        return device

    def init_path(self, save_path):
        assert save_path is not None, "Error, Got an invalid save path"
        ckpt_path = osp.join(save_path, "ckpt")
        res_path = osp.join(save_path, "res")
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(res_path, exist_ok=True)
        return ckpt_path, res_path

    def save_checkpoint(self):
        assert osp.exists(self.ckpt_path)
        torch.save(self.model.state_dict(), osp.join(self.ckpt_path, "spatialign.bgi"))

    def load_checkpoint(self):
        ckpt_file = osp.join(self.ckpt_path, "spatialign.bgi")
        assert osp.exists(ckpt_file)
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        state_dict = self.model.state_dict()
        trained_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
        state_dict.update(trained_dict)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    @get_running_time
    def init_bank(self):
        self.model.eval()
        header_bank = defaultdict()
        for idx, data in enumerate(self.dataset.loader_list):
            data = data.to(self.device)
            graph_loss, dgi_loss, recon_loss, latent_x, recon_x = self.model(
                x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, domain_idx=data.domain_idx,
                neigh_mask=data.neigh_graph)
            header_bank[idx] = latent_x.detach()
        return header_bank

    @torch.no_grad()
    def update_bank(self, idx, feat, alpha=0.5):
        self.header_bank[idx] = feat * alpha + (1 - alpha) * self.header_bank[idx]
