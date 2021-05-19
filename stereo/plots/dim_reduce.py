#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:dim_reduce.py
@time:2021/04/14
"""
from .scatter import plot_scatter
from anndata import AnnData


def plot_dim_reduce(
        adata: AnnData,
        obs_key: list = ["clustering"],
        pos_key: str = "dim_reduce",
        plot_cluster: list = None,
        bad_color: str = "lightgrey",
        ncols: int = 2,
        dot_size: int = None,
        color_list=['violet', 'turquoise', 'tomato', 'teal', 'tan', 'silver', 'sienna', 'red', 'purple', 'plum', 'pink',
                    'orchid', 'orangered', 'orange', 'olive', 'navy', 'maroon', 'magenta', 'lime',
                    'lightgreen', 'lightblue', 'lavender', 'khaki', 'indigo', 'grey', 'green', 'gold', 'fuchsia',
                    'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral', 'chocolate', 'chartreuse', 'brown', 'blue',
                    'black', 'beige', 'azure', 'aquamarine', 'aqua'],
):
    plot_scatter(adata=adata, plot_key=obs_key, pos_key=pos_key, plot_cluster=plot_cluster, bad_color=bad_color,
                 ncols=ncols, dot_size=dot_size, color_list=color_list)
