#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 9:49
# @Author  : zhangchao
# @File    : classifier.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import random
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stereo.log_manager import logger
from .dataset import DnnDataset
from .early_stop import EarlyStopping
from .loss import MultiCEFocalLoss


class BatchModel(nn.Module):
    def __init__(self, input_dims, n_batch):
        super(BatchModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, 512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, n_batch),
            nn.Dropout()
        )

    def forward(self, x):
        return self.net(x)


class BatchClassifier:
    def __init__(
            self,
            input_dims,
            n_batch,
            data_x,
            batch_idx,
            batch_size=4096,
            gpu="0",
            data_loader_num_workers=-1,
            num_threads=-1):
        self.set_seed(num_threads=num_threads)
        if isinstance(gpu, (str, int)) and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.device = torch.device("cpu")
        self.model = BatchModel(input_dims, n_batch)
        self.model.to(self.device)
        self.train_loader, self.test_loader = self.convert_loader(data=data_x, batch_idx=batch_idx,
                                                                  batch_size=batch_size,
                                                                  num_workers=data_loader_num_workers)
        self.loss_fn = MultiCEFocalLoss(n_batch, gamma=2, alpha=.25, reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
        # self.scaler = torch.cuda.amp.GradScaler()

    def set_seed(self, seed=42, num_threads=-1):
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def convert_loader(self, data, batch_idx, batch_size, num_workers):
        num_spot = data.shape[0]
        self.train_size = int(num_spot * 0.7)
        index = np.arange(num_spot)
        np.random.shuffle(index)
        train_index, test_index = index[:self.train_size], index[self.train_size:]
        train_dataset = data[train_index]
        test_dataset = data[test_index]
        train_batch = batch_idx[train_index]
        test_batch = batch_idx[test_index]

        if num_workers <= 0 or num_workers > cpu_count():
            num_workers = cpu_count()

        train_loader = DataLoader(
            dataset=DnnDataset(train_dataset, train_batch),
            batch_size=min(batch_size, int(num_spot * 0.7)),
            drop_last=True,
            shuffle=True,
            num_workers=num_workers)
        test_loader = DataLoader(
            dataset=DnnDataset(test_dataset, test_batch),
            batch_size=min(batch_size, int(num_spot * 0.3)),
            drop_last=False,
            shuffle=False,
            num_workers=num_workers)
        return train_loader, test_loader

    def train(self, max_epochs=500, save_path=None):
        assert save_path is not None
        os.makedirs(save_path, exist_ok=True)
        self.model.train()
        early_stop = EarlyStopping(patience=10)

        for eph in range(max_epochs):
            epoch_acc = []
            epoch_loss = []
            for idx, data in enumerate(self.train_loader):
                x, y = data
                x = x.float().to(self.device, non_blocking=True)
                y = y.long().to(self.device)
                if self.device.type == 'cuda' and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        y_hat = self.model(x)
                        loss = self.loss_fn(y_hat, y)
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                epoch_loss.append(loss.item())
                total = y.size(0)
                predict = y_hat.argmax(1)
                correct = predict.eq(y).sum().item()
                acc = correct / total * 100.
                epoch_acc.append(acc)
            early_stop(np.mean(epoch_loss))
            if early_stop.counter == 0:
                torch.save(self.model.state_dict(), osp.join(save_path, "batch_model.bgi"))
            if early_stop.stop_flag:
                logger.info("Model Training Finished!")
                logger.info(f"Trained checkpoint file has been saved to {save_path}")
                break

    def validation(self, pt_path):
        assert osp.exists(osp.join(pt_path, "batch_model.bgi"))
        checkpoint = torch.load(osp.join(pt_path, "batch_model.bgi"), map_location=lambda storage, loc: storage)
        state_dict = self.model.state_dict()
        trained_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
        state_dict.update(trained_dict)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        batch_acc = []
        for idx, data in enumerate(self.train_loader):
            x, y = data
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
            predict = F.softmax(y_hat, dim=1).argmax(1)
            correct = predict.eq(y).sum().item()
            total = y.size(0)
            acc = correct / total * 100.
            batch_acc.append(acc)

    def test(self, pt_path):
        assert osp.exists(osp.join(pt_path, "batch_model.bgi"))
        checkpoint = torch.load(osp.join(pt_path, "batch_model.bgi"), map_location=lambda storage, loc: storage)
        state_dict = self.model.state_dict()
        trained_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
        state_dict.update(trained_dict)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        batch_acc = []
        for idx, data in enumerate(self.test_loader):
            x, y = data
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
            predict = F.softmax(y_hat, dim=1).argmax(1)
            correct = predict.eq(y).sum().item()
            total = y.size(0)
            acc = correct / total
            batch_acc.append(acc)
        acc_mean = np.mean(batch_acc)
        return acc_mean
