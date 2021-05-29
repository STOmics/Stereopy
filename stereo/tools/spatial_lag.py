#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:spatial_lag.py
@time:2021/04/19
"""
from ..core.tool_base import ToolBase
from ..utils.data_helper import get_cluster_res, get_position_array
from anndata import AnnData
from pysal.model import spreg
from pysal.lib import weights
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
from ..core.stereo_result import SpatialLagResult
from ..log_manager import logger


class SpatialLag(ToolBase):
    """
    spatial lag model, calculate bin-cell's lag coefficient, lag z-stat and p-value
    """
    def __init__(self, data: AnnData, method='gm_lag', name='spatial_lag', cluster=None, genes=None,
                 random_drop=True, drop_dummy=None, n_neighbors=8):
        """
        initialization

        :param data: anndata object contenting cluster results
        :param method: method
        :param name: tool name, will be used as a key when adding tool result to andata object.
        :param cluster: the 'Clustering' tool name, defined when running 'Clustering' tool
        :param genes: specify genes, default using all genes
        :param random_drop: randomly drop bin-cells if True
        :param drop_dummy: drop specify clusters
        :param n_neighbors: number of neighbors
        """
        super(SpatialLag, self).__init__(data=data, method=method, name=name)
        self.param = self.get_params(locals())
        self.cluster = self.data.uns[cluster].cluster
        self.genes = genes
        self.random_drop = random_drop
        self.drop_dummy = drop_dummy
        self.n_neighbors = n_neighbors
        self.position = get_position_array(self.data, obs_key='spatial')

    def fit(self):
        """
        run analysis
        """
        x, uniq_group = self.get_data()
        res = self.gm_model(x, uniq_group)
        result = SpatialLagResult(name=self.name, param=self.param, score=res)
        self.add_result(result=result, key_added=self.name)
        return result

    def get_data(self):
        """
        get cluster result, convert cluster Series to dummy codes

        :return: cluster dummy codes and cluster names
        """
        group_num = self.cluster['cluster'].value_counts()
        max_group, min_group, min_group_ncells = group_num.index[0], group_num.index[-1], group_num[-1]
        df = pd.DataFrame({'group': self.cluster['cluster']})
        drop_columns = None
        if self.random_drop:
            df.iloc[sample(np.arange(self.data.n_obs).tolist(), min_group_ncells), :] = 'others'
            drop_columns = ['group_others']
        if self.drop_dummy:
            group_inds = np.where(df['group'] == self.drop_dummy)[0]
            df.iloc[group_inds, :] = 'others'
            drop_columns = ['group_others', 'group_' + str(self.drop_dummy)]
        x = pd.get_dummies(data=df, drop_first=False)
        if drop_columns is not None:
            x.drop(columns=drop_columns, inplace=True)
        uniq_group = set(self.cluster['cluster']).difference([self.drop_dummy]) if self.drop_dummy is not None \
            else set(self.cluster['cluster'])
        return x, list(uniq_group)

    def get_genes(self):
        """
        get specify genes

        :return : gene names
        """
        if self.genes is None:
            genes = self.data.var.index
        else:
            genes = self.data.var.index.intersection(self.genes)
        return genes

    def gm_model(self, x, uniq_group):
        """
        run gm model

        :param x:
        :param uniq_group:
        :return:
        """
        knn = weights.distance.KNN.from_array(self.position, k=self.n_neighbors)
        knn.transform = 'R'
        genes = self.get_genes()
        result = pd.DataFrame(index=genes)
        vars_info = ['const'] + uniq_group + ['W_log_exp']
        for i in vars_info:
            result[str(i) + '_lag_coeff'] = None
            result[str(i) + '_lag_zstat'] = None
            result[str(i) + '_lag_pval'] = None
        for i, cur_g in tqdm(enumerate(genes),
                             desc="performing GM_lag_model and assign coefficient and p-val to cell type"):
            x['log_exp'] = self.data[:, cur_g].X
            try:
                model = spreg.GM_Lag(x[['log_exp']].values, x.values,
                                     w=knn, name_y='log_exp')
                a = pd.DataFrame(model.betas, model.name_x + ['W_log_exp'], columns=['coef'])
                b = pd.DataFrame(model.z_stat, model.name_x + ['W_log_exp'], columns=['z_stat', 'p_val'])
                df = a.merge(b, left_index=True, right_index=True)
                for ind, g in enumerate(vars_info):
                    result.loc[cur_g, str(g) + '_lag_coeff'] = df.iloc[ind, 0]
                    result.loc[cur_g, str(g) + '_lag_zstat'] = df.iloc[ind, 1]
                    result.loc[cur_g, str(g) + '_lag_pval'] = df.iloc[ind, 2]
            except Exception as e:
                logger.error(f'spatial lag get an error: {e}.')
                for ind, g in enumerate(vars_info):
                    result.loc[cur_g, str(g) + '_lag_coeff'] = np.nan
                    result.loc[cur_g, str(g) + '_lag_zstat'] = np.nan
                    result.loc[cur_g, str(g) + '_lag_pval'] = np.nan
        return result
