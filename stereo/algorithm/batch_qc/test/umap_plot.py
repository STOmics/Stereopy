#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:37
# @Author  : zhangchao
# @File    : umap_plot.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sn
from anndata import AnnData

from ..utils import generate_palette


def umap_plot(merge_data: AnnData, visualize_key: str = "batch"):
    plt.figure(figsize=(12.8, 7.2))
    palette = generate_palette(category_nums=merge_data.obs[visualize_key].cat.categories.size)
    sn.scatterplot(x=merge_data.obsm["X_umap"][:, 0],
                   y=merge_data.obsm["X_umap"][:, 1],
                   hue=merge_data.obs[visualize_key],
                   palette=palette,
                   s=5,
                   alpha=0.5)
    plt.title(f"Visualization UMAP, Color by: {visualize_key}", fontsize=20, pad=15)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("UMAP1", fontsize=20)
    plt.ylabel("UMAP2", fontsize=20)
    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, fontsize=10)
    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format="png", bbox_inches='tight', dpi=300)
    return fig_buffer
