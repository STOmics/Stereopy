import numpy as np
import anndata as ad
from anndata import AnnData
from scipy.sparse import issparse
from typing import List, Tuple, Optional
import ot
from .utils import *
import pandas as pd

from stereo.core.stereo_exp_data import StereoExpData

def transfer_matrix(
    data1: StereoExpData,
    data2: StereoExpData,
    layer: str = None,
    spatial_key: str = "spatial",
    alpha: float = 0.1, 
    epsilon = 0.01,
    rho = np.inf,
    G_1 = None,
    G_2 = None,
    **kwargs
):
    
    
    """
    Calculates transfer matrix between two time. 
    
    Args:
        adata1: the first time(source) data.
        adata2: the second time(target) data.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        spatial_key: Key in .obsm containing coordinates for each cell.
        alpha:  Alignment tuning parameter. Note:0 <= alpha <= 1. When ``alpha = 0`` only the gene expression data is taken into account,
               while ``alpha =1`` only the spatial coordinates are taken into account.
        epsilon: weight for entropy regularization term,defaults to 1.0.
        rho: weight for KL divergence penalizing unbalanced transport,defaults to 100.0.
        G_1: distance matrix within spatial data 1 (spots, spots),defult is None.
        G_2: distance matrix within spatial data 2(spots, spots),defult is None.
        
   
    Returns:
        matrix(dim:n_obs*m_obs) of transition probability.
    """
    ## data process
    common_genes = list(set(data1.gene_names) & set(data2.gene_names))
    data1 = data1.sub_by_name(gene_name=common_genes)
    data2 = data2.sub_by_name(gene_name=common_genes)
    
    ## get gene expression matrix
    exp_martix_1 = data1.get_exp_matrix(use_raw=False, layer=layer, to_array=True)
    exp_martix_2 = data2.get_exp_matrix(use_raw=False, layer=layer, to_array=True)
    
    ## Calculate spatial coordinates diatances
    spa_dist_1 = spatial_dist(data1, spatial_key=spatial_key)
    spa_dist_2 = spatial_dist(data2, spatial_key=spatial_key)
    
    ## Calculate gene expression dissimilarity
    M = gene_dist(exp_martix_1, exp_martix_2)
    
    ## init distributions
    weight_matrix = np.exp(1-M)
    p_1 = np.sum(weight_matrix, axis=1)
    p_2 = np.sum(weight_matrix, axis=0)
    p_1 = p_1/np.sum(p_1)
    p_2 = p_2/np.sum(p_2)
    
    ## Run unbalanced ot
    if alpha > 0.0:
        if G_1 is None:
            G_1 = spa_dist_1/np.max(spa_dist_1)
        if G_2 is None:
            G_2 = spa_dist_2/np.max(spa_dist_2)
    ## 01. None Graph structure
    # balanced ot, rho=inf
    if alpha == 0.0 and np.isinf(rho):
        pi = ot.sinkhorn(p_1, p_2, M, epsilon, **kwargs)
    # unbalanced ot,rho<inf
    elif alpha == 0.0 and not np.isinf(rho):
        pi = uot(p_1, p_2, M, epsilon, rho = rho)
    ## 02. Graph structure considered
    else:
        pi = usot(p_1, p_2, M, G_1, G_2, alpha, epsilon = epsilon, rho = rho)
    
    return pi


def generate_animate_input(
    pi_list: List[np.array],
    data_list: List[StereoExpData],
    spatial_key: str = 'spatial',
    time: str = 'batch',
    annotation: str = 'Celltype'
):  
    """
    Generate animate transfer input of two or more times. 
    
    Args:
        pi_list: transfer matrix (nd.array) list.
        adata_list: AnnData list.
        spatial_key: Key in .obsm containing coordinates for each cell.
        time: time Key in .obs.
        annotation: cell type Key in .obs.
    Returns:
        DataFrame of transfer info between all times, as input of dispaly animate plot.

    """
    ## 1.check adata.obs contains keys:x,y,time.
    for i, data in enumerate(data_list):
        
        if data.cells.obs.columns.isin(['x','y','time','cell_id','annotation']).sum() == 5:
            pass
        else:
            data_list[i]= pre_check_adata(data, spatial_key=spatial_key, time=time, annotation=annotation)

    ## 2.merge transfer matrix data for two or more time, return pi_matrix, such as slice1,slice2,slice3,......
    map_list=[]
    for i in range(0, len(data_list) - 1):
        map_data,pi=pi_process(pi_list[i] ,data_list[i], data_list[i+1])
        map_data['slice'+str(i+1)] =pi.index
        map_data['slice'+str(i+2)]=pi.idxmax(axis=1)
        map_data['pi_value'+str(i+1)]=pi.max(axis=1)
        map_list.append(map_data)
        if i >=1 and i <= len(data_list)-1:
             # map_list[i]=pd.merge(map_list[i-1], map_list[i], left_on='slice'+str(i+1), right_on='slice'+str(i+1)).drop(
             #             ['pi_value'+str(i),'pi_value'+str(i+1)], axis=1)
            map_list[i]=pd.merge(map_list[i-1], map_list[i], left_on='slice'+str(i+1), right_on='slice'+str(i+1))                  
    if len(pi_list) == 1:
        #map_list[-1] =map_list[0].drop(['pi_value'+str(i+1)], axis=1)
        map_list[-1] =map_list[0]
    pi_matrix = map_list[-1]
    pi_matrix_picol = pi_matrix.copy()
    pi_matrix.drop(columns=pi_matrix.columns[pi_matrix.columns.str.startswith('pi')], inplace=True)
    
    ## 3.merge adata.obs for two or more time.
    data_info = pd.concat(([i.cells.obs for i in data_list]), axis=0)
    
    ## 4.merge pi_matrix & data_info, to add coord info and annotation
    for i, element in enumerate(map_list[-1].columns):
        pi_matrix = pd.merge(pi_matrix, data_info, left_on=element, right_on='cell_id').drop(['time','cell_id'], axis=1)
    
    ## 4.1 get sub columns,slice1,annotation,x,y
    pi_matrix_columns = pi_matrix.columns.to_list()   
    slice_list = map_list[-1].columns.to_list()
    annotation_list = [i for i in pi_matrix_columns if i.startswith("annotation")]
    x_list = [i for i in pi_matrix_columns if i.startswith("x")]
    y_list = [i for i in pi_matrix_columns if i.startswith("y")]
    lable_list = slice_list + annotation_list + x_list + y_list
    pi_matrix_coord = pi_matrix[lable_list]
    ## 4.2 rename sub solumns.
    annotation_dic = {ele:"slice"+str(i+1)+"_annotation" for i,ele in enumerate(annotation_list)}
    x_dic = {ele:"slice"+str(i+1)+"_x" for i,ele in enumerate(x_list)}
    y_dic = {ele:"slice"+str(i+1)+"_y" for i,ele in enumerate(y_list)}
    pi_matrix_coord=pi_matrix_coord.rename(columns={**annotation_dic,**x_dic,**y_dic})
   
    ## 4.3 calculate transfer line slop and intercept.
    for i, element in enumerate(slice_list):
        if i >=1 and i <= len(slice_list)-1:
            pi_matrix_coord['k_line_'+str(i)+str(i+1)]=(pi_matrix_coord["slice"+str(i)+"_y"] -pi_matrix_coord["slice"+str(i+1)+"_y"]) / (pi_matrix_coord["slice"+str(i)+"_x"] - pi_matrix_coord["slice"+str(i+1)+"_x"])
            pi_matrix_coord['b_line_'+str(i)+str(i+1)] = pi_matrix_coord["slice"+str(i)+"_y"] -pi_matrix_coord['k_line_'+str(i)+str(i+1)]*pi_matrix_coord["slice"+str(i)+"_x"]
    
    ## add 
    for col in pi_matrix_picol.columns:  
        if col.startswith('pi'):  
            pi_matrix_coord[col] = pi_matrix_picol[col]
    ## return pandas as animate_transfer input.
    return pi_matrix_coord

def pi_process(
    pi: np.array,
    data1: StereoExpData,
    data2: StereoExpData,
):  
    """
    transfer matrix process to get DataFrame. 
    """
    pi = pd.DataFrame(pi)
    pi.index = data1.cell_names
    pi.columns = data2.cell_names
    map_data = pd.DataFrame(index=pi.index)
    return map_data, pi

def map_data(
    pi: np.ndarray,
    data1: StereoExpData,
    data2: StereoExpData,
):
    map_data, pi = pi_process(pi, data1, data2)
    map_data['slice1'] = pi.index
    map_data['slice2'] = pi.idxmax(axis=1)
    map_data['pi_value'] = pi.max(axis=1)
    return map_data
