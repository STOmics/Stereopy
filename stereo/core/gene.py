"""
@file: gene.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/29  create file.
"""
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import spmatrix

from stereo.log_manager import logger

class Gene(object):
    def __init__(
            self,
            var: Optional[pd.DataFrame] = None,
            gene_name: Optional[np.ndarray] = None
        ):
        if var is not None:
            if not isinstance(var, pd.DataFrame):
                raise TypeError("var must be a DataFrame.")
            self._var = var
            if gene_name is not None:
                self.gene_name = gene_name
        else:
            self._var = pd.DataFrame(index=gene_name if gene_name is None else gene_name.astype('U'))
        self._matrix = dict()
        self._pairwise = dict()
        # self.loc = self._var.loc

    def __contains__(self, item):
        return item in self._var.columns

    # def __setattr__(self, key, value):
    #     if key in {'_var', '_matrix', '_pairwise', 'gene_name', 'loc'}:
    #         object.__setattr__(self, key, value)
    #     else:
    #         if value is not None:
    #             self._var[key] = value

    def __setitem__(self, key, value):
        self._var[key] = value

    def __getitem__(self, key):
        # if key not in self._var.columns:
        #     return None
        return self._var[key]
    
    def __len__(self):
        return self.size
    
    @property
    def matrix(self):
        return self._matrix
    
    @property
    def pairwise(self):
        return self._pairwise

    @property
    def size(self):
        return self.gene_name.size
    
    @property
    def loc(self):
        return self._var.loc
    
    @property
    def iloc(self):
        return self._var.iloc
    
    @property
    def to_csv(self):
        return self._var.to_csv
    
    @property
    def var(self):
        return self._var

    @property
    def n_cells(self):
        if 'n_cells' not in self._var.columns:
            return None
        return self._var['n_cells'].to_numpy()

    @n_cells.setter
    def n_cells(self, values):
        if values is not None:
            self._var['n_cells'] = values

    @property
    def n_counts(self):
        if 'n_counts' not in self._var.columns:
            return None
        return self._var['n_counts'].to_numpy()

    @n_counts.setter
    def n_counts(self, values):
        if values is not None:
            self._var['n_counts'] = values

    @property
    def mean_umi(self):
        if 'mean_umi' not in self._var.columns:
            return None
        return self._var['mean_umi'].to_numpy()

    @mean_umi.setter
    def mean_umi(self, values):
        if values is not None:
            self._var['mean_umi'] = values

    @property
    def gene_name(self):
        """
        get the genes name.

        :return: genes name.
        """
        return self._var.index.to_numpy().astype('U')

    @gene_name.setter
    def gene_name(self, name: np.ndarray):
        """
        set the name of gene.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('gene name must be a np.ndarray.')
        self._var = self._var.reindex(name)
    
    @property
    def real_gene_name(self):
        if 'real_gene_name' in self._var.columns:
            return self._var['real_gene_name'].to_numpy().astype('U')
        else:
            return None
    
    @real_gene_name.setter
    def real_gene_name(self, real_gene_name):
        if not isinstance(real_gene_name, np.ndarray):
            raise TypeError('gene name must be a np.ndarray.')
        self._var['real_gene_name'] = real_gene_name

    def sub_set(self, index):
        """
        get the subset of Gene by the index info, the Gene object will be inplaced by the subset.

        :param index: a numpy array of index info.
        :return: the subset of Gene object.
        """
        if isinstance(index, pd.Series):
            index = index.to_numpy()
        self._var = self._var.iloc[index].copy()
        for col in self._var.columns:
            if self._var[col].dtype.name == 'category':
                self._var[col] = self._var[col].cat.remove_unused_categories()
        
        for key, value in self._matrix.items():
            if isinstance(value, pd.DataFrame):
                self._matrix[key] = value.iloc[index].copy()
                self._matrix[key].reset_index(drop=True, inplace=True)
            elif isinstance(value, (np.ndarray, spmatrix)):
                self._matrix[key] = value[index]
            else:
                logger.warning(f'Subsetting from {key} of type {type(value)} in gene.matrix is not supported.')

        for key, value in self._pairwise.items():
            if isinstance(value, pd.DataFrame):
                columns = value.columns[index]
                self._pairwise[key] = value.iloc[index][columns].copy()
                self._pairwise[key].reset_index(drop=True, inplace=True)
            elif isinstance(value, (np.ndarray, spmatrix)):
                if len(value.shape) != 2:
                    logger.warning(f'Subsetting from {key} of shape {value.shape} in gene.pairwise is not supported.')
                    continue
                self._pairwise[key] = value[index][:, index]
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, pd.DataFrame):
                        columns = v.columns[index]
                        self._pairwise[key][k] = v.iloc[index][columns].copy()
                        self._pairwise[key][k].reset_index(drop=True, inplace=True)
                    elif isinstance(v, (np.ndarray, spmatrix)):
                        self._pairwise[key][k] = v[index][:, index]
                    else:
                        logger.warning(f'Subsetting from {key}.{k} of type {type(v)} in gene.pairwise is not supported.')
            else:
                logger.warning(f'Subsetting from {key} of type {type(value)} in gene.pairwise is not supported.')
        return self

    def to_df(self, copy=False):
        """
        Transform StereoExpData object to pd.DataFrame.

        :return: a dataframe of Gene.
        """
        return self._var.copy(deep=True) if copy else self._var

    def __str__(self):
        format_genes = ['gene_name']
        for attr_name in self._var.columns:
            format_genes.append(attr_name)
        return f"\ngenes: {format_genes}" if format_genes else ""

    def _repr_html_(self):
        return self._var._repr_html_()


class AnnBasedGene(Gene):

    def __init__(self, based_ann_data: AnnData, gene_name: Optional[np.ndarray] = None):
        self.__based_ann_data = based_ann_data
        super(AnnBasedGene, self).__init__(gene_name=gene_name)

    def __setattr__(self, key, value):
        if key == '_var':
            return
        object.__setattr__(self, key, value)

    def __str__(self):
        return str(self.__based_ann_data.var)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.__based_ann_data.var[item]

    def __contains__(self, item):
        return item in self.__based_ann_data.var.columns

    @property
    def _var(self):
        return self.__based_ann_data.var
    
    @property
    def matrix(self):
        return self.__based_ann_data.varm
    
    @property
    def pairwise(self):
        return self.__based_ann_data.varp
    
    # @property
    # def loc(self):
    #     return self.__based_ann_data.var.loc
    
    # @property
    # def iloc(self):
    #     return self.__based_ann_data.var.iloc

    @property
    def gene_name(self) -> np.ndarray:
        """
        get the genes name.

        :return: genes name.
        """
        return self.__based_ann_data.var_names.values.astype('U')

    @gene_name.setter
    def gene_name(self, name: np.ndarray):
        """
        set the name of gene.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('gene name must be a np.ndarray object.')
        if name.size != self.__based_ann_data.n_vars:
            raise ValueError(f'The length of gene names must be {self.__based_ann_data.n_vars}, but now is {name.size}')
        self.__based_ann_data.var_names = name
        # self.__based_ann_data._inplace_subset_var(name)
    
    @property
    def real_gene_name(self):
        if 'real_gene_name' in self.__based_ann_data.var.columns:
            return self.__based_ann_data.var['real_gene_name'].to_numpy().astype('U')
        else:
            return None

    def to_df(self, copy=False):
        return self.__based_ann_data.var.copy(deep=True) if copy else self.__based_ann_data.var
