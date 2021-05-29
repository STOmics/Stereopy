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
        plot_key: list = ["clustering"],
        dim_reduce_key: str = "dim_reduce",
        show_names: list = None,
        bad_color: str = "lightgrey",
        ncols: int = 2,
        dot_size: int = None,
        color_list=['violet', 'turquoise', 'tomato', 'teal', 'tan', 'silver', 'sienna', 'red', 'purple', 'plum', 'pink',
                    'orchid', 'orangered', 'orange', 'olive', 'navy', 'maroon', 'magenta', 'lime',
                    'lightgreen', 'lightblue', 'lavender', 'khaki', 'indigo', 'grey', 'green', 'gold', 'fuchsia',
                    'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral', 'chocolate', 'chartreuse', 'brown', 'blue',
                    'black', 'beige', 'azure', 'aquamarine', 'aqua'],
):
    """
    plot scatter of dimensionality reduction

    :param adata: the annotation data which contents cluster and dimensionality reduction's analysis results.
    :param plot_key: the list of  gene's name or the cluster task's name, defined when running 'Clustering' tool by setting 'name' property.
    :param dim_reduce_key: the dimensionality reduction task's name, defined when running 'DimReduce' tool by setting 'name' property.
    :param show_names: the name list of plot_key to show.
    :param bad_color: set nan values color.
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param color_list: whether to invert y-axis.
    :return: None
    """
    plot_scatter(adata=adata, plot_key=plot_key, pos_key=dim_reduce_key, plot_cluster=show_names, bad_color=bad_color,
                 ncols=ncols, dot_size=dot_size, color_list=color_list)
