#!/usr/bin/env python3
# coding: utf-8
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
        self._gene_name = gene_name if gene_name is None else gene_name.astype('U')
        self.n_cells = None
        self.n_counts = None

    @property
    def gene_name(self):
        """
        get the genes name.

        :return: genes name.
        """
        return self._gene_name

    @gene_name.setter
    def gene_name(self, name: np.ndarray):
        """
        set the name of gene.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('gene name must be a np.ndarray object.')
        self._gene_name = name.astype('U')

    def sub_set(self, index):
        """
        get the subset of Gene by the index info， the Gene object will be inplaced by the subset.

        :param index: a numpy array of index info.
        :return: the subset of Gene object.
        """
        if self.gene_name is not None:
            self.gene_name = self.gene_name[index]
        if self.n_cells is not None:
            self.n_cells = self.n_cells[index]
        if self.n_counts is not None:
            self.n_counts = self.n_counts[index]
        return self

    def to_df(self):
        """
        transform Gene object to pd.DataFrame.

        :return: a dataframe of Gene.
        """
        attributes = {
            'n_counts': self.n_counts,
            'n_cells': self.n_cells,
        }
        df = pd.DataFrame(attributes, index=self.gene_name)
        return df


class AnnBasedGene(Gene):

    def __init__(self, based_ann_data: AnnData, gene_name: Optional[np.ndarray]):
        self.__based_ann_data = based_ann_data
        super(AnnBasedGene, self).__init__(gene_name)

    def __str__(self):
        return str(self.__based_ann_data.var)

    def __repr__(self):
        return self.__str__()

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
