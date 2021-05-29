#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:get_stereo_data.py
@time:2021/04/01
"""
import numpy as np


def get_cluster_res(adata, data_key='clustering'):
    cluster_data = adata.uns[data_key].cluster
    cluster = cluster_data['cluster'].astype(str).astype('category').values
    return cluster


def get_reduce_x(data, data_key='reduce_dim'):
    reduce_data = data.uns[data_key]
    reduce_x = reduce_data.x_reduce
    return reduce_x


def get_position_array(data, obs_key='spatial'):
    return np.array(data.obsm[obs_key])[:, 0: 2]


def get_degs_res(data, group_key, data_key='find_marker', top_k=None):
    degs_dict = data.uns[data_key]
    degs_data = degs_dict[group_key]
    if top_k is not None:
        return degs_data.top_k_marker(top_k_genes=top_k, sort_key='scores')
    else:
        return degs_data.degs_data


def get_find_marker_group(data, data_key='find_marker'):
    return [i for i in data.uns[data_key].keys()]


def get_spatial_lag_group(data, data_key='spatial_lag'):
    lag_res = data.uns[data_key]
    lag_coeff = list(lag_res.score.columns[lag_res.score.columns.str.endswith('lag_coeff')])
    if 'const_lag_coeff' in lag_coeff:
        lag_coeff.remove('const_lag_coeff')
    if 'W_log_exp_lag_coeff' in lag_coeff:
        lag_coeff.remove('W_log_exp_lag_coeff')
    return [i.strip('_lag_coeff') for i in lag_coeff]
