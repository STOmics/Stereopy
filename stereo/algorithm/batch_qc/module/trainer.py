#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:06
# @Author  : zhangchao
# @File    : trainer.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .classifier import BatchClassifier


def domain_variance_score(
        merge_data: AnnData,
        n_batch: int,
        use_rep: str = "X_pca",
        batch_key: str = "batch",
        batch_size: int = 4096,
        gpu: Union[str, int] = '0',
        data_loader_num_workers: int = -1,
        num_threads: int = -1,
        save_path: str = "./"
):
    assert use_rep in merge_data.obsm_keys()
    classifier = BatchClassifier(
        input_dims=merge_data.obsm[use_rep].shape[1],
        n_batch=n_batch,
        data_x=merge_data.obsm[use_rep],
        batch_idx=merge_data.obs[batch_key].cat.codes,
        batch_size=batch_size,
        gpu=gpu,
        data_loader_num_workers=data_loader_num_workers,
        num_threads=num_threads
    )
    classifier.train(max_epochs=500, save_path=save_path)
    test_acc = classifier.test(pt_path=save_path)
    df = pd.DataFrame(data={
        "n_batch": n_batch,
        "n_sample": merge_data.shape[0],
        "Train Size": classifier.train_size,
        "Accept Rate": np.around(1 - test_acc, decimals=4)
    },
        index=["domain variance"]
    )
    return df
