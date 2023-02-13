#!/usr/bin/env python3
# coding: utf-8
"""
@file: cell.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/29  create file.
    2021/08/17  add get_property and to_df function to file, by wuyiran.
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData


class Cell(object):
    def __init__(self, cell_name: Optional[np.ndarray] = None, cell_border: Optional[np.ndarray] = None,
                 batch: Optional[Union[np.ndarray, list, int, str]] = None):
        self._cell_name = cell_name
        self._cell_border = cell_border
        self._batch = self._set_batch(batch) if batch is not None else None
        self.total_counts = None
        self.pct_counts_mt = None
        self.n_genes_by_counts = None

    @property
    def cell_name(self):
        """
        get the name of cell.

        :return: cell name
        """
        return self._cell_name

    @cell_name.setter
    def cell_name(self, name: np.ndarray):
        """
        set the name of cell.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('cell name must be a np.ndarray object.')
        self._cell_name = name

    @property
    def cell_border(self):
        return self._cell_border

    @cell_border.setter
    def cell_boder(self, cell_border: np.ndarray):
        if not isinstance(cell_border, np.ndarray):
            raise TypeError('cell border must be a np.ndarray object.')
        self._cell_border = cell_border

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch: Union[np.ndarray, list, int]):
        self._batch = self._set_batch(batch)

    def _set_batch(self, batch: Union[np.ndarray, list, int]):
        if batch is None:
            return None

        if not isinstance(batch, np.ndarray) and not isinstance(batch, list) and not isinstance(batch,
                                                                                                int) and not isinstance(
            batch, str):
            raise TypeError('batch must be np.ndarray or list or int or str')

        if isinstance(batch, int):
            batch = str(batch)
        if isinstance(batch, str):
            return np.repeat(batch, len(self.cell_name))
        else:
            return (np.array(batch) if isinstance(batch, list) else batch).astype('U')

    def sub_set(self, index):
        """
        get the subset of Cell by the index info， the Cell object will be inplaced by the subset.

        :param index: a numpy array of index info.
        :return: the subset of Cell object.
        """
        if self.cell_name is not None:
            self.cell_name = self.cell_name[index]
        if self.cell_boder is not None:
            self.cell_boder = self.cell_boder[index]
        if self.total_counts is not None:
            self.total_counts = self.total_counts[index]
        if self.pct_counts_mt is not None:
            self.pct_counts_mt = self.pct_counts_mt[index]
        if self.n_genes_by_counts is not None:
            self.n_genes_by_counts = self.n_genes_by_counts[index]
        if self.batch is not None:
            self.batch = self.batch[index]
        return self

    def get_property(self, name):
        """
        get the property value by the name.

        :param name: the name of property.
        :return: the property.
        """
        if name == 'total_counts':
            return self.total_counts
        if name == 'pct_counts_mt':
            return self.pct_counts_mt
        if name == 'n_genes_by_counts':
            return self.n_genes_by_counts

    def to_df(self):
        """
        transform Cell object to pd.DataFrame.

        :return: a dataframe of Cell.
        """
        attributes = {
            'total_counts': self.total_counts,
            'pct_counts_mt': self.pct_counts_mt,
            'n_genes_by_counts': self.n_genes_by_counts
        }
        if self._batch is not None:
            attributes['batch'] = self._batch
        df = pd.DataFrame(attributes, index=self.cell_name)
        return df


class AnnBasedCell(Cell):

    def __init__(self, based_ann_data: AnnData, cell_name: Optional[np.ndarray] = None,
                 cell_border: Optional[np.ndarray] = None,
                 batch: Optional[Union[np.ndarray, list, int, str]] = None):
        self.__based_ann_data = based_ann_data
        super(AnnBasedCell, self).__init__(cell_name, cell_border, batch)

    def __str__(self):
        return str(self.__based_ann_data.obs)

    def __repr__(self):
        return self.__str__()

    @property
    def cell_name(self) -> np.ndarray:
        """
        get the name of cell.

        :return: cell name
        """
        return self.__based_ann_data.obs_names.values.astype(str)

    @cell_name.setter
    def cell_name(self, name: np.ndarray):
        """
        set the name of cell.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('cell name must be a np.ndarray object.')
        self.__based_ann_data._inplace_subset_obs(name)

    @property
    def total_counts(self):
        return self.__based_ann_data.obs['total_counts']

    @total_counts.setter
    def total_counts(self, new_total_counts):
        if new_total_counts is not None:
            self.__based_ann_data.obs['total_counts'] = new_total_counts

    @property
    def pct_counts_mt(self):
        return self.__based_ann_data.obs['pct_counts_mt']

    @pct_counts_mt.setter
    def pct_counts_mt(self, new_pct_counts_mt):
        if new_pct_counts_mt is not None:
            self.__based_ann_data.obs['pct_counts_mt'] = new_pct_counts_mt

    @property
    def n_genes_by_counts(self):
        return self.__based_ann_data.obs['n_genes_by_counts']

    @n_genes_by_counts.setter
    def n_genes_by_counts(self, new_n_genes_by_counts):
        if new_n_genes_by_counts is not None:
            self.__based_ann_data.obs['n_genes_by_counts'] = new_n_genes_by_counts
