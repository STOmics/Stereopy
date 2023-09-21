#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:38
# @Author  : zhangchao
# @File    : heatmap.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from anndata import AnnData

from ..utils import generate_palette


def sample_heatmap(merge_data: AnnData, feat_key: str = "X_pca", metric: str = "correlation", batch_key: str = "batch"):
    plt.figure(figsize=(12.8, 7.2))
    palette = generate_palette(category_nums=merge_data.obs[batch_key].cat.categories.size)
    color_list = [palette[i] for i in merge_data.obs["batch"].cat.codes]

    if feat_key.endswith("corrcoef"):
        columns = merge_data.obs.index.tolist()
    else:
        columns = [f"PC{i + 1}" for i in range(merge_data.obsm[feat_key].shape[1])]
    pc_data = pd.DataFrame(data=merge_data.obsm[feat_key],
                           columns=columns,
                           index=merge_data.obs.index.tolist())
    sn.clustermap(pc_data,
                  row_colors=color_list,
                  metric=metric,
                  col_cluster=False)

    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
    return fig_buffer
