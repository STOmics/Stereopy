#!/usr/bin/env python3
# coding: utf-8
"""
@author: Shixu He  heshixu@genomics.cn
@last modified by: Shixu He
@file:scatter_plt.py
@time:2021/03/15
"""

from anndata import AnnData

from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, to_hex
from matplotlib import gridspec

import numpy as np
import pandas as pd

from ...log_manager import logger

def scatter(x: list, y: list, ax: Axes = None, dot_colors=None, marker=".", dot_size=1, cmap = None, plotnonfinite=True, **kwargs):
    """
    Simplified scatter plot function, which wraps matplotlib.axes.Axes.scatter .
    :param x,y: Data position list.
    :param ax: plot axes.
    :param color: list of colors.
    :param marker: marker type.
    :param size: marker size.
    :param cmap: Color map.
    :param plotnonfinite: whether to plot bad point.
    :return: plot axes.
    """


    if (len(x)<=0 or len(y)<=0):
        logger.warning("x position or y position has no data.")
    if (dot_colors == None):
        dot_colors = ["gray"]
    if (isinstance(dot_colors, str)):
        dot_colors = [dot_colors]

    if (ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if (cmap == None):
        cmap = get_cmap()

    pathcollection = ax.scatter(
        x, y,
        c = dot_colors,
        marker = marker,
        s = dot_size,
        cmap = cmap,
        plotnonfinite = plotnonfinite,
        **kwargs,
    )

    return pathcollection

def plot_cluster_result(adata: AnnData, obs_key: list = ["phenograph"], pos_key = "spatial", plot_cluster: list= None, bad_color = "lightgrey", ncols = 2, dot_size = None, invert_y = False,
color_list = ['violet', 'turquoise', 'tomato', 'teal', 
            'tan', 'silver','sienna', 'red','purple', 
            'plum', 'pink','orchid','orangered','orange', 
            'olive', 'navy','maroon','magenta','lime', 
            'lightgreen','lightblue','lavender','khaki', 
            'indigo', 'grey','green','gold','fuchsia', 
            'darkgreen','darkblue','cyan','crimson','coral', 
            'chocolate','chartreuse','brown','blue', 'black', 
            'beige', 'azure','aquamarine','aqua']): # scatter plot, 聚类后表达矩阵空间分布
    """
    Plot spatial distribution of specified obs data.
    ============ Arguments ============
    :param adata: AnnData object.
    :param obs_key: specified obs cluster key list, for example: ["phenograph"].
    :param pos_key: the coordinates of data points for scatter plots. the data points are stored in adata.obsm[pos_key]. choice: "spatial", "X_umap", "X_pca".
    :param plot_cluster: the name list of clusters to show.
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param cmap: Color map.
    :param invert_y: whether to invert y-axis.
    ============ Return ============
    None.
    ============ Example ============
    plot_spatial_cluster(adata = adata)
    """
    #sc.pl.embedding(adata, basis="spatial", color=["total_counts", "n_genes_by_counts"],size=30)
    
    if dot_size is None:
        dot_size = 120000 / adata.shape[0]
    
    ncols = min(ncols, len(obs_key))
    nrows = np.ceil(len(obs_key) / ncols).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = plt.figure(figsize=(ncols * 10, nrows * 8))
    left = 0.2 / ncols
    bottom = 0.13 / nrows
    axs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        left=left,
        right=1 - (ncols - 1) * left - 0.01 / ncols,
        bottom=bottom,
        top=1 - (nrows - 1) * bottom - 0.1 / nrows,
        #hspace=hspace,
        #wspace=wspace,
    )

    if color_list is None:
        cmap = get_cmap()
    else:
        cmap = ListedColormap(color_list)
    cmap.set_bad(bad_color)
        # 把特定值改为 np.nan 之后，可以利用 cmap.set_bad("white") 来遮盖掉这部分数据

    for i, key in enumerate(obs_key):
        color_data = adata.obs_vector(key)
        pcLogic = False

        #color_data = np.asarray(color_data_raw, dtype=float)
        order = np.argsort(~pd.isnull(color_data), kind="stable")
        spatial_data = np.array(adata.obsm[pos_key])[:, 0 : 2]
        color_data = color_data[order]
        spatial_data = spatial_data[order, :]

        color_dict = {}
        has_na = False
        if pd.api.types.is_categorical_dtype(color_data):
            pcLogic = True
            if plot_cluster is None:
                plot_cluster = list(color_data.categories)
        if pcLogic:
            clusterN = len(np.unique(color_data))
            if len(color_list)<clusterN:
                color_list = color_list * clusterN
                cmap = ListedColormap(color_list)
                cmap.set_bad(bad_color)
            if (len(color_data.categories) > len(plot_cluster)):
                color_data = color_data.replace(color_data.categories.difference(plot_cluster), np.nan)
                has_na = True
            color_dict = {str(k): to_hex(v) for k, v in enumerate(color_list)}
            color_data = color_data.map(color_dict)
            if (pd.api.types.is_categorical_dtype(color_data)):
                color_data = pd.Categorical(color_data)
            if has_na:
                color_data = color_data.add_categories([to_hex(bad_color)])
                color_data = color_data.fillna(to_hex(bad_color))
                #color_dict["NA"]
            
        
        # color_data 是图像中各个点的值，也对应了每个点的颜色。data_points则对应了各个点的坐标
        ax = fig.add_subplot(axs[i]) # ax = plt.subplot(axs[i]) || ax = fig.add_subplot(axs[1, 1]))
        ax.set_title(key)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("spatial1")
        ax.set_ylabel("spatial2")
        pathcollection = scatter(
                    spatial_data[:, 0],
                    spatial_data[:, 1],
                    ax = ax,
                    marker = ".",
                    dot_colors = color_data,
                    dot_size = dot_size
                )
        if pcLogic:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])
            #valid_cate = color_data.categories
            cat_num = len(adata.obs_vector(key).categories)
            for label in adata.obs_vector(key).categories:
                ax.scatter([], [], c=color_dict[label], label=label)
            ax.legend(
                frameon=False,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                ncol=(1 if cat_num <= 14 else 2 if cat_num <= 30 else 3),
                #fontsize=legend_fontsize,
            )
        else:
            plt.colorbar(pathcollection, ax=ax, pad=0.01, fraction=0.08, aspect=30)
        ax.autoscale_view()
        #ax.invert_yaxis()
