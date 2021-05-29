#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:clustering.py
@time:2021/04/13
"""
from .scatter import plot_scatter
from anndata import AnnData
from ._params_doc import anndata


def plot_spatial_cluster(
        adata: AnnData,
        cluster_names: list = ["phenograph"],
        pos_key: str = "spatial",
        plot_cluster: list = None,
        bad_color: str = "lightgrey",
        ncols: int = 2,
        dot_size: int = None,
        color_list=['violet', 'turquoise', 'tomato', 'teal','tan', 'silver', 'sienna', 'red', 'purple', 'plum', 'pink',
                    'orchid', 'orangered', 'orange', 'olive', 'navy', 'maroon', 'magenta', 'lime',
                    'lightgreen', 'lightblue', 'lavender', 'khaki', 'indigo', 'grey', 'green', 'gold', 'fuchsia',
                    'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral', 'chocolate', 'chartreuse', 'brown', 'blue', 'black',
                    'beige', 'azure', 'aquamarine', 'aqua',
                    ],
):
    """
    showing spatial bin-cell distribution after clustering

    :param adata: the annotation data which contents cluster's analysis results.
    :param cluster_names: the cluster task's name, defined when running 'Clustering' tool by setting 'name' property.
    :param pos_key: the coordinates of data points for scatter plots. the data points are stored in adata.obsm[pos_key]. choice: "spatial", "X_umap", "X_pca".
    :param plot_cluster: the name list of clusters to show.
    :param bad_color: set nan values color.
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param color_list: whether to invert y-axis.
    :return: None
    """
    plot_scatter(adata=adata, plot_key=cluster_names, pos_key=pos_key, plot_cluster=plot_cluster, bad_color=bad_color,
                 ncols=ncols, dot_size=dot_size, color_list=color_list)
