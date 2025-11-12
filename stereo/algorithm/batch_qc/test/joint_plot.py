#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:36
# @Author  : zhangchao
# @File    : joint_plot.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sn
from anndata import AnnData


def joint_plot(data: AnnData, batch_key: str = "batch", use_rep: str = "X_pca"):
    plt.figure(figsize=(12.8, 7.2))
    g = sn.jointplot(x=data.obsm[use_rep][:, 0],
                     y=data.obsm[use_rep][:, 1],
                     s=3,
                     hue=data.obs[batch_key])
    g.set_axis_labels("Dim1", "Dim2", fontsize=16)
    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
    return fig_buffer
