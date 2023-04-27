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

    def __init__(
            self,
            cell_name: Optional[np.ndarray],
            cell_border: Optional[np.ndarray] = None,
            batch: Optional[Union[np.ndarray, list, int, str]] = None
    ):
        self._obs = pd.DataFrame(index=cell_name if cell_name is None else cell_name.astype('U'))
        self._matrix = dict()
        self._pairwise = dict()
        if batch is not None:
            self._obs['batch'] = self._set_batch(batch)
        self._cell_border = cell_border

    def __contains__(self, item):
        return item in self._obs.columns or item in self._matrix or item in self._pairwise

    def __setattr__(self, key, value):
        if key in {'_obs', '_matrix', '_pairwise', '_cell_border', 'cell_name', 'cell_border'}:
            object.__setattr__(self, key, value)
        elif key == 'batch':
            self._obs[key] = self._set_batch(value)
        else:
            self._obs[key] = value

    def __setitem__(self, key, value):
        self._obs[key] = value

    def __getitem__(self, key):
        return self._obs[key]

    @property
    def total_counts(self):
        if 'total_counts' not in self._obs.columns:
            return None
        return self._obs['total_counts'].values

    @total_counts.setter
    def total_counts(self, value):
        self._obs['total_counts'] = value

    @property
    def pct_counts_mt(self):
        if 'pct_counts_mt' not in self._obs.columns:
            return None
        return self._obs['pct_counts_mt'].values

    @pct_counts_mt.setter
    def pct_counts_mt(self, value):
        self._obs['pct_counts_mt'] = value

    @property
    def n_genes_by_counts(self):
        if 'n_genes_by_counts' not in self._obs.columns:
            return None
        return self._obs['n_genes_by_counts'].values

    @n_genes_by_counts.setter
    def n_genes_by_counts(self, value):
        self._obs['n_genes_by_counts'] = value

    @property
    def cell_name(self):
        """
        get the name of cell.

        :return: cell name
        """
        return self._obs.index.to_numpy().astype('U')

    @cell_name.setter
    def cell_name(self, name: np.ndarray):
        """
        set the name of cell.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('cell name must be a np.ndarray object.')
        self._obs = self._obs.reindex(name)

    @property
    def cell_border(self):
        return self._cell_border

    @cell_border.setter
    def cell_border(self, cell_border: np.ndarray):
        if not isinstance(cell_border, np.ndarray):
            raise TypeError('cell border must be a np.ndarray object.')
        self._cell_border = cell_border

    @property
    def batch(self):
        if 'batch' not in self._obs.columns:
            return None
        return self._obs['batch'].values

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

        if self.cell_border is not None:
            self.cell_border = self.cell_border[index]
        if type(index) is list:
            self._obs = self._obs.iloc[index]
        elif index.dtype == bool:
            self._obs = self._obs[index]
        else:
            self._obs = self._obs.iloc[index]
        return self

    def get_property(self, name):
        """
        get the property value by the name.

        :param name: the name of property.
        :return: the property.
        """
        return self._obs[name].to_numpy()

    def to_df(self):
        """
        Transform StereoExpData object to pd.DataFrame.

        :return: a dataframe of Cell.
        """
        return self._obs.copy(deep=True)

    def __str__(self):
        format_cells = ['cell_name']
        for attr_name in self._obs.columns:
            format_cells.append(attr_name)
        return f"\ncells: {format_cells}" if format_cells else ""


class AnnBasedCell(Cell):

    def __init__(self, based_ann_data: AnnData, cell_name: Optional[np.ndarray] = None,
                 cell_border: Optional[np.ndarray] = None,
                 batch: Optional[Union[np.ndarray, list, int, str]] = None):
        self.__based_ann_data = based_ann_data
        super(AnnBasedCell, self).__init__(cell_name, cell_border, batch)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

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
