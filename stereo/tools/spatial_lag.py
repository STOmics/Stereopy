#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:spatial_lag.py
@time:2021/04/19
"""
from ..core.tool_base import ToolBase
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
    spatial lag model, calculate cell-bin's lag coefficient, lag z-stat and p-value

    :param data: StereoExpData object contenting cluster results
    :param groups: group information matrix
    :param genes: specify genes, default using all genes
    :param random_drop: randomly drop bin-cells if True
    :param drop_dummy: drop specify clusters
    :param n_neighbors: number of neighbors
    """
    def __init__(
            self,
            data,
            groups=None,
            genes=None,
            random_drop=True,
            drop_dummy=None,
            n_neighbors=8
    ):
        super(SpatialLag, self).__init__(data=data, groups=groups, method='gm_lag')
        self.genes = genes
        self.random_drop = random_drop
        self.drop_dummy = drop_dummy
        self.n_neighbors = n_neighbors

    def fit(self):
        """
        run analysis
        """
        self.data.sparse2array()
        x, uniq_group = self.get_data()
        res = self.gm_model(x, uniq_group)
        result = SpatialLagResult(res)
        return result

    def get_data(self):
        """
        get cluster result, convert cluster Series to dummy codes

        :return: cluster dummy codes and cluster names
        """
        group_num = self.groups['group'].value_counts()
        max_group, min_group, min_group_ncells = group_num.index[0], group_num.index[-1], group_num[-1]
        df = pd.DataFrame({'group': self.groups['group']})
        drop_columns = None
        if self.random_drop:
            df.iloc[sample(np.arange(len(self.data.cell_names)).tolist(), min_group_ncells), :] = 'others'
            drop_columns = ['group_others']
        if self.drop_dummy:
            group_inds = np.where(df['group'] == self.drop_dummy)[0]
            df.iloc[group_inds, :] = 'others'
            drop_columns = ['group_others', 'group_' + str(self.drop_dummy)]
        x = pd.get_dummies(data=df, drop_first=False)
        if drop_columns is not None:
            x.drop(columns=drop_columns, inplace=True)
        uniq_group = set(self.groups['group']).difference([self.drop_dummy]) if self.drop_dummy is not None \
            else set(self.groups['group'])
        return x, list(uniq_group)

    def get_genes(self):
        """
        get specify genes

        :return : gene names
        """
        if self.genes is None:
            genes = self.data.gene_names
        else:
            genes = self.data.gene_names.intersection(self.genes)
        return genes

    def gm_model(self, x, uniq_group):
        """
        run gm model

        :param x:
        :param uniq_group:
        :return:
        """
        knn = weights.distance.KNN.from_array(self.data.position, k=self.n_neighbors)
        knn.transform = 'R'
        genes = self.get_genes()
        result = pd.DataFrame(index=genes)
        vars_info = ['const'] + uniq_group + ['W_log_exp']
        for i in vars_info:
            result[str(i) + '_lag_coeff'] = None
            result[str(i) + '_lag_zstat'] = None
            result[str(i) + '_lag_pval'] = None
        data_pd = pd.DataFrame(self.data.exp_matrix, columns=self.data.gene_names, index=self.data.cell_names)
        for i, cur_g in tqdm(enumerate(genes),
                             desc="performing GM_lag_model and assign coefficient and p-val to cell type"):
            x['log_exp'] = data_pd[cur_g].values
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
        self.result.matrix = result
        return result
