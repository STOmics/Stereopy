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
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import spmatrix

from stereo.log_manager import logger


class Cell(object):

    def __init__(
            self,
            obs: Optional[pd.DataFrame] = None,
            cell_name: Optional[np.ndarray] = None,
            cell_border: Optional[np.ndarray] = None,
            batch: Optional[Union[np.ndarray, list, int, str]] = None
    ):
        if obs is not None:
            if not isinstance(obs, pd.DataFrame):
                raise TypeError("obs must be a DataFrame.")
            self._obs = obs
            if cell_name is not None:
                self.cell_name = cell_name
        else:
            self._obs = pd.DataFrame(index=cell_name if cell_name is None else cell_name.astype('U'))
        # self.loc = self._obs.loc
        self._matrix = dict()
        self._pairwise = dict()
        if batch is not None:
            self._obs['batch'] = self._set_batch(batch)
        self._cell_border = cell_border
        self.cell_point = None

    def __contains__(self, item):
        return item in self._obs.columns

    # def __setattr__(self, key, value):
    #     if key in {'_obs', '_matrix', '_pairwise', '_cell_border', 'cell_name', 'cell_border', 'loc', 'cell_point'}:
    #         object.__setattr__(self, key, value)
    #     elif key == 'batch':
    #         self._obs[key] = self._set_batch(value)
    #     else:
    #         if value is not None:
    #             self._obs[key] = value

    def __setitem__(self, key, value):
        if value is not None:
            self._obs[key] = value

    def __getitem__(self, key):
        # if key not in self._obs.columns:
        #     return None
        return self._obs[key]
    
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
        return self._obs.index.size
    
    @property
    def loc(self):
        return self._obs.loc
    
    @property
    def iloc(self):
        return self._obs.iloc
    
    @property
    def to_csv(self):
        return self._obs.to_csv
    
    @property
    def obs(self):
        return self._obs

    @property
    def total_counts(self):
        if 'total_counts' not in self._obs.columns:
            return None
        return self._obs['total_counts'].to_numpy()

    @total_counts.setter
    def total_counts(self, value):
        self._obs['total_counts'] = value

    @property
    def pct_counts_mt(self):
        if 'pct_counts_mt' not in self._obs.columns:
            return None
        return self._obs['pct_counts_mt'].to_numpy()

    @pct_counts_mt.setter
    def pct_counts_mt(self, value):
        self._obs['pct_counts_mt'] = value

    @property
    def n_genes_by_counts(self):
        if 'n_genes_by_counts' not in self._obs.columns:
            return None
        return self._obs['n_genes_by_counts'].to_numpy()

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
        return self._obs['batch'].to_numpy()
    
    @batch.setter
    def batch(self, batch):
        self._obs['batch'] = self._set_batch(batch)

    def _set_batch(self, batch: Union[np.ndarray, list, int, str]):
        if batch is None:
            return None

        if not isinstance(batch, np.ndarray) and not isinstance(batch, list) \
                and not isinstance(batch, int) and not isinstance(batch, str):
            raise TypeError('batch must be np.ndarray or list or int or str')

        if isinstance(batch, int):
            batch = str(batch)
        if isinstance(batch, str):
            return np.repeat(batch, len(self.cell_name)).astype('U')
        else:
            return (np.array(batch) if isinstance(batch, list) else batch).astype('U')

    def sub_set(self, index):
        """
        get the subset of Cell by the index info, the Cell object will be inplaced by the subset.

        :param index: a numpy array of index info.
        :return: the subset of Cell object.
        """

        if self.cell_border is not None:
            self.cell_border = self.cell_border[index]
        if isinstance(index, pd.Series):
            index = index.to_numpy()
        self._obs = self._obs.iloc[index].copy()
        for col in self._obs.columns:
            if self._obs[col].dtype.name == 'category':
                self._obs[col] = self._obs[col].cat.remove_unused_categories()
        for key, value in self._matrix.items():
            if isinstance(value, pd.DataFrame):
                self._matrix[key] = value.iloc[index].copy()
                self._matrix[key].reset_index(drop=True, inplace=True)
            elif isinstance(value, (np.ndarray, spmatrix)):
                self._matrix[key] = value[index]
            else:
                logger.warning(f'Subsetting from {key} of type {type(value)} in cell.matrix is not supported.')

        for key, value in self._pairwise.items():
            if isinstance(value, pd.DataFrame):
                columns = value.columns[index]
                self._pairwise[key] = value.iloc[index][columns].copy()
                self._pairwise[key].reset_index(drop=True, inplace=True)
            elif isinstance(value, (np.ndarray, spmatrix)):
                if len(value.shape) != 2:
                    logger.warning(f'Subsetting from {key} of shape {value.shape} in cell.pairwise is not supported.')
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
                        logger.warning(f'Subsetting from {key}.{k} of type {type(v)} in cell.pairwise is not supported.')
            else:
                logger.warning(f'Subsetting from {key} of type {type(value)} in cell.pairwise is not supported.')
        return self

    def get_property(self, name):
        """
        get the property value by the name.

        :param name: the name of property.
        :return: the property.
        """
        return self._obs[name].to_numpy()

    def to_df(self, copy=False):
        """
        Transform StereoExpData object to pd.DataFrame.

        :return: a dataframe of Cell.
        """

        obs = self._obs.copy(deep=True) if copy else self._obs
        if 'batch' in obs.columns:
            obs['batch'] = obs['batch'].astype('category')
        return obs

    def __str__(self):
        format_cells = ['cell_name']
        for attr_name in self._obs.columns:
            format_cells.append(attr_name)
        return f"\ncells: {format_cells}" if format_cells else ""

    def _repr_html_(self):
        obs: pd.DataFrame = self.to_df()
        return obs._repr_html_()


class AnnBasedCell(Cell):

    def __init__(self, based_ann_data: AnnData, cell_name: Optional[np.ndarray] = None,
                 cell_border: Optional[np.ndarray] = None,
                 batch: Optional[Union[np.ndarray, list, int, str]] = None):
        self.__based_ann_data = based_ann_data
        super(AnnBasedCell, self).__init__(cell_name=cell_name)
        if cell_border is not None:
            self.cell_border = cell_border
        if batch is not None:
            self.batch = batch

    def __setattr__(self, key, value):
        if key == '_obs':
            return
        # elif key == 'batch':
        #     self.__based_ann_data.obs[key] = self._set_batch(value)
        #     self.__based_ann_data.obs[key] = self.__based_ann_data.obs[key].astype('category')
        else:
            object.__setattr__(self, key, value)

    def __str__(self):
        return str(self.__based_ann_data.obs)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        # if item not in self.__based_ann_data.obs.columns:
        #     return None
        return self.__based_ann_data.obs[item]

    def __contains__(self, item):
        return item in self.__based_ann_data.obs.columns

    @property
    def _obs(self):
        return self.__based_ann_data.obs
    
    @property
    def matrix(self):
        return self.__based_ann_data.obsm
    
    @property
    def pairwise(self):
        return self.__based_ann_data.obsp
    
    # @property
    # def loc(self):
    #     return self.__based_ann_data.obs.loc
    
    # @property
    # def iloc(self):
    #     return self.__based_ann_data.obs.iloc

    @property
    def cell_name(self) -> np.ndarray:
        """
        get the name of cell.

        :return: cell name
        """
        return self.__based_ann_data.obs_names.values.astype('U')

    @cell_name.setter
    def cell_name(self, name: np.ndarray):
        """
        set the name of cell.

        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError('cell name must be a np.ndarray object.')
        if name.size != self.__based_ann_data.n_obs:
            raise ValueError(f'The length of cell names must be {self.__based_ann_data.n_obs}, but now is {name.size}')
        self.__based_ann_data.obs_names = name
        # self.__based_ann_data._inplace_subset_obs(name)

    @property
    def total_counts(self):
        if 'total_counts' not in self.__based_ann_data.obs.columns:
            return None
        return self.__based_ann_data.obs['total_counts'].to_numpy()

    @total_counts.setter
    def total_counts(self, new_total_counts):
        if new_total_counts is not None:
            self.__based_ann_data.obs['total_counts'] = new_total_counts

    @property
    def pct_counts_mt(self):
        if 'pct_counts_mt' not in self.__based_ann_data.obs.columns:
            return None
        return self.__based_ann_data.obs['pct_counts_mt'].to_numpy()

    @pct_counts_mt.setter
    def pct_counts_mt(self, new_pct_counts_mt):
        if new_pct_counts_mt is not None:
            self.__based_ann_data.obs['pct_counts_mt'] = new_pct_counts_mt

    @property
    def n_genes_by_counts(self):
        if 'n_genes_by_counts' not in self.__based_ann_data.obs.columns:
            return None
        return self.__based_ann_data.obs['n_genes_by_counts'].to_numpy()

    @n_genes_by_counts.setter
    def n_genes_by_counts(self, new_n_genes_by_counts):
        if new_n_genes_by_counts is not None:
            self.__based_ann_data.obs['n_genes_by_counts'] = new_n_genes_by_counts
    
    # @property
    # def batch(self):
    #     if 'batch' not in self.__based_ann_data._obs.columns:
    #         return None
    #     return self.__based_ann_data._obs['batch'].to_numpy()
    
    @Cell.batch.setter
    def batch(self, batch):
        self.__based_ann_data.obs['batch'] = self._set_batch(batch)
        self.__based_ann_data.obs['batch'] = self.__based_ann_data.obs['batch'].astype('category')

    @property
    def cell_border(self):
        return self.__based_ann_data.obsm.get('cell_border', None)
    
    @cell_border.setter
    def cell_border(self, cell_border: np.ndarray):
        if not isinstance(cell_border, np.ndarray):
            raise TypeError('cell border must be a np.ndarray object.')
        if len(cell_border.shape) != 3:
            raise Exception(f'The cell border must have 3 dimensions, but now {len(cell_border.shape)}.')
        self.__based_ann_data.obsm['cell_border'] = cell_border

    def to_df(self, copy=False):
        return self.__based_ann_data.obs.copy(deep=True) if copy else self.__based_ann_data.obs
