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


class Gene(object):
    def __init__(self, gene_name: Optional[np.ndarray]):
        self._var = pd.DataFrame(index=gene_name if gene_name is None else gene_name.astype('U'))
        self._matrix = dict()
        self._pairwise = dict()
        self.loc = self._var.loc

    def __contains__(self, item):
        return item in self._var.columns

    def __setattr__(self, key, value):
        if key in {'_var', '_matrix', '_pairwise', 'gene_name', 'loc'}:
            object.__setattr__(self, key, value)
        else:
            if value is not None:
                self._var[key] = value

    def __setitem__(self, key, value):
        self._var[key] = value

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
            raise TypeError('gene name must be a np.ndarray object.')
        self._var = self._var.reindex(name)

    def sub_set(self, index):
        """
        get the subset of Gene by the index info， the Gene object will be inplaced by the subset.

        :param index: a numpy array of index info.
        :return: the subset of Gene object.
        """
        if type(index) is list:
            self._var = self._var.iloc[index].copy()
        elif index.dtype == bool:
            self._var = self._var[index].copy()
        else:
            self._var = self._var.iloc[index].copy()
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


class AnnBasedGene(Gene):

    def __init__(self, based_ann_data: AnnData, gene_name: Optional[np.ndarray]):
        self.__based_ann_data = based_ann_data
        super().__init__(gene_name)
        # self._var = self.__based_ann_data.var

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
    def gene_name(self) -> np.ndarray:
        """
        get the genes name.

        :return: genes name.
        """
        return self.__based_ann_data.var_names.values.astype(str)

    @gene_name.setter
    def gene_name(self, name: np.ndarray):
        """
        set the name of gene.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('gene name must be a np.ndarray object.')
        self.__based_ann_data._inplace_subset_var(name)

    def to_df(self):
        return self.__based_ann_data.var
