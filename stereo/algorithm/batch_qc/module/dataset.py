#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 9:47
# @Author  : zhangchao
# @File    : dataset.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import numpy as np
from torch.utils.data import Dataset


class DnnDataset(Dataset):
    def __init__(self, data, batch_idx):
        assert data.shape[0] == batch_idx.shape[0]
        self.batch_idx = batch_idx
        norm_factor = np.linalg.norm(data, axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1
        self.data = data / norm_factor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].squeeze()
        y = self.batch_idx[idx]
        return x, y
