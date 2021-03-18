#!/usr/bin/env python3
# coding: utf-8
"""
@author: Shixu He  heshixu@genomics.cn
@last modified by: Shixu He
@file:plot_utils.py
@time:2021/03/15
"""

from anndata import AnnData
import pandas as pd
import numpy as np
import math

from matplotlib.colors import Normalize, ListedColormap, to_hex
from matplotlib import gridspec
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import seaborn

from ._plot_basic.scatter_plt import scatter
from ._plot_basic.heatmap_plt import heatmap, _plot_categories_as_colorblocks, _plot_gene_groups_brackets

from ..log_manager import logger

def plot_spatial_distribution(adata: AnnData, obs_key: list = ["total_counts", "n_genes_by_counts"], ncols = 2, dot_size = None, color_list = None, invert_y = False): # scatter plot, 表达矩阵空间分布
    """
    Plot spatial distribution of specified obs data.
    :param adata: AnnData object.
    :param obs_key: specified obs key list, for example: ["total_counts", "n_genes_by_counts"]
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param cmap: Color map.
    :param invert_y: whether to invert y-axis.
    :return: None.
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
        # 把特定值改为 np.nan 之后，可以利用 cmap.set_bad("white") 来遮盖掉这部分数据

    for i, key in enumerate(obs_key):
        #color_data = np.asarray(adata.obs_vector(key), dtype=float)
        color_data = adata.obs_vector(key)
        order = np.argsort(~pd.isnull(color_data), kind="stable")
        spatial_data = np.array(adata.obsm["spatial"])[:, 0 : 2]
        color_data = color_data[order]
        spatial_data = spatial_data[order, :]
        
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
                    ax,
                    marker=".",
                    color= color_data,
                    size = dot_size,
                    cmap = cmap,
                )
        plt.colorbar(pathcollection, ax=ax, pad=0.01, fraction=0.08, aspect=30)
        ax.autoscale_view()
        #ax.invert_yaxis()

def plot_spatial_cluster(adata: AnnData, obs_key: list = ["phenograph"], plot_cluster: list= None, bad_color = "lightgrey", ncols = 2, dot_size = None, invert_y = False,
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
    :param adata: AnnData object.
    :param obs_key: specified obs cluster key list, for example: ["phenograph"]
    :param ncols: numbr of plot columns.
    :param dot_size: marker size.
    :param cmap: Color map.
    :param invert_y: whether to invert y-axis.
    :return: None.
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
        spatial_data = np.array(adata.obsm["spatial"])[:, 0 : 2]
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
                    marker=".",
                    color= color_data,
                    size = dot_size
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
    

def plot_to_select_filter_value(): # scatter plot, 线粒体分布图
    '''

    '''

def plot_variable_gene(): # scatter plot, 表达量差异-均值图
    '''

    '''

def plot_cluster_umap(): # scatter plot，聚类结果PCA/umap图
    '''

    '''

def plot_expression_difference(): # scatter plot, 差异基因显著性图，类碎石图
    '''

    '''

def plot_violin_distribution(adata): # 小提琴统计图
    '''
    绘制数据的分布小提琴图
    :param adata: AnnData object.
    :return: None
    '''
    _, axs = plt.subplots(1, 3, figsize=(15, 4))
    seaborn.violinplot(y=adata.obs['total_counts'], ax=axs[0])
    seaborn.violinplot(y=adata.obs['n_genes_by_counts'], ax=axs[1])
    seaborn.violinplot(y=adata.obs['pct_counts_mt'], ax=axs[2])

def plot_heatmap_maker_genes(adata: AnnData = None, cluster_method = "phenograph", marker_uns_key = None, num_show_gene = 8, show_labels=True, order_cluster = True, marker_clusters = None, cluster_colors_array = None, **kwargs): # heatmap, 差异基因热图
    '''
    
    '''

    if marker_uns_key is None:
        marker_uns_key = 'marker_genes'

    #if cluster_method is None:
    #    cluster_method = str(adata.uns[marker_uns_key]['params']['groupby'])

    #cluster_colors_array = adata.uns["phenograph_colors"]

    if marker_clusters is None: 
        marker_clusters = adata.uns[marker_uns_key]['names'].dtype.names
    
    if not set(marker_clusters).issubset(set(adata.uns[marker_uns_key]['names'].dtype.names)):
        marker_clusters = adata.uns[marker_uns_key]['names'].dtype.names

    gene_names_dict = {}  # dict in which each cluster is the keyand the num_show_gene are the values

    for cluster in marker_clusters:
        # get all genes that are 'non-nan'
        genes_array = adata.uns[marker_uns_key]['names'][cluster]
        genes_array = genes_array[~pd.isnull(genes_array)]

        if len(genes_array) == 0:
            logger.warning("Cluster {} has no genes.".format(cluster))
            continue
        gene_names_dict[cluster] = list(genes_array[:num_show_gene])

    adata._sanitize()

    gene_names = []
    gene_group_labels = []
    gene_group_positions = []
    start = 0
    for label, gene_list in gene_names_dict.items():
        if isinstance(gene_list, str):
            gene_list = [gene_list]
        gene_names.extend(list(gene_list))
        gene_group_labels.append(label)
        gene_group_positions.append((start, start + len(gene_list) - 1))
        start += len(gene_list)

    draw_df = pd.DataFrame(index=adata.obs_names)
    uniq_gene_names = np.unique(gene_names)
    draw_df = pd.concat(
        [draw_df, pd.DataFrame(adata.X[tuple([slice(None), adata.var.index.get_indexer(uniq_gene_names)])], columns=uniq_gene_names, index=adata.obs_names)],
        axis=1
    )

    # add obs values
    draw_df = pd.concat([draw_df, adata.obs[cluster_method]], axis=1)

    # reorder columns to given order (including duplicates keys if present)
    draw_df = draw_df[list([cluster_method]) + list(uniq_gene_names)]
    draw_df = draw_df[gene_names].set_index(draw_df[cluster_method].astype('category'))
    if order_cluster:
        draw_df = draw_df.sort_index()

    # From scanpy
    # define a layout of 2 rows x 3 columns
    # first row is for 'brackets' (if no brackets needed, the height of this row
    # is zero) second row is for main content. This second row is divided into
    # three axes:
    #   first ax is for the categories defined by `cluster_method`
    #   second ax is for the heatmap
    #   fourth ax is for colorbar

    kwargs.setdefault("figsize", (10, 10))
    kwargs.setdefault("colorbar_width", 0.2)
    colorbar_width = kwargs.get("colorbar_width")
    figsize = kwargs.get("figsize")
    
    cluster_block_width = kwargs.setdefault("cluster_block_width", 0.2) if order_cluster else 0
    if figsize is None:
        height = 6
        if show_labels:
            heatmap_width = len(gene_names) * 0.3
        else:
            heatmap_width = 8
        width = heatmap_width + cluster_block_width
    else:
        width, height = figsize
        heatmap_width = width - cluster_block_width

    if gene_group_positions is not None and len(gene_group_positions) > 0:
        # add some space in case 'brackets' want to be plotted on top of the image
        height_ratios = [0.15, height]
    else:
        height_ratios = [0, height]

    width_ratios = [
        cluster_block_width,
        heatmap_width,
        colorbar_width,
    ]

    fig = plt.figure(figsize=(width, height))

    axs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=width_ratios,
        wspace=0.15 / width,
        hspace=0.13 / height,
        height_ratios=height_ratios,
    )

    heatmap_ax = fig.add_subplot(axs[1, 1])

    width, height = fig.get_size_inches()
    max_cbar_height = 4.0
    if height > max_cbar_height:
        # to make the colorbar shorter, the
        # ax is split and the lower portion is used.
        axs2 = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=axs[1, 2],
            height_ratios=[height - max_cbar_height, max_cbar_height],
        )
        heatmap_cbar_ax = fig.add_subplot(axs2[1])
    else:
        heatmap_cbar_ax = fig.add_subplot(axs[1, 2])

    heatmap(df = draw_df, ax = heatmap_ax, 
            norm = Normalize(vmin=None, vmax=None), plot_colorbar=True, colorbar_ax=heatmap_cbar_ax,
            show_labels = True, plot_hline = True)


    if order_cluster:
        _plot_categories_as_colorblocks(
            fig.add_subplot(axs[1, 0]), draw_df, colors=cluster_colors_array, orientation='left'
        )

    # plot cluster legends on top of heatmap_ax (if given)
    if gene_group_positions is not None and len(gene_group_positions) > 0:
        _plot_gene_groups_brackets(
            fig.add_subplot(axs[0, 1], sharex=heatmap_ax),
            group_positions=gene_group_positions,
            group_labels=gene_group_labels,
            rotation=None,
            left_adjustment=-0.3,
            right_adjustment=0.3,
        )
    
    #plt.savefig()

