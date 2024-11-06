import os

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

from stereo.core.stereo_exp_data import StereoExpData
from stereo.log_manager import logger
from stereo.stereo_config import stereo_conf

def assess_start_cluster(
    data: StereoExpData,
    use_col: str = 'cluster'
):
    """Assess the entropy value to identify the starting cluster

    Parameters
    ----------
    adata
        Anndata

    Returns
    -------
        adata
    """
    
    exp_matrix = data.exp_matrix.toarray() if data.issparse() else data.exp_matrix
    entropy_list = entropy(exp_matrix, axis=1)
    data.cells['entropy'] = entropy_list
    mean_entropy_sorted_in_cluster = data.cells.obs[[use_col, 'entropy']].groupby([use_col]).mean().sort_values(by='entropy', ascending=False)
    logger.info(f'Cluster order sorted by entropy value: {list(mean_entropy_sorted_in_cluster.index)}')
    return mean_entropy_sorted_in_cluster

def assess_start_cluster_plot(
    data: StereoExpData,
    use_col='cluster',
    palette='stereo_30',
    width=10,
    height=9
):
    """ Plot the entropy value and Stem Cell Differentiation score of each cluster

    Parameters
    ----------
    adata
        Anndata

    Returns
    -------
        figure
    """
    # clusters = data.tl.result['Mean_Entropy_sorted_in_cluster'].index.to_numpy()
    clusters = data.cells[use_col].cat.categories
    colors = stereo_conf.get_colors(palette, n=len(clusters))
    color_palette = dict(zip(clusters, colors))
    clusters_order = data.tl.result['Mean_Entropy_sorted_in_cluster'].index.to_numpy()
    plt.figure(figsize=(width, height))
    ax1=sns.boxplot(data=data.cells.obs, x=use_col, y='entropy', linewidth=0.8, palette=color_palette,
                    order=clusters_order, showfliers=False)

    ax1.set_xticklabels(clusters_order, rotation=40, ha='center', fontsize=15)
    ax1.set_xlabel(' ')
    ax1.set_ylabel('entropy value',fontsize=20)


    plt.tight_layout()

    return ax1.get_figure()

    
    
        
    
    
        
