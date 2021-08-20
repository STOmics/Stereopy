#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:stereo_exp_data.py
@time:2021/03/22

change log:
    2021/08/12  add to_andata function , by wuyiran.
"""
from .data import Data
import pandas as pd
import numpy as np
from typing import Optional, Union
from scipy.sparse import spmatrix, csr_matrix, issparse
from shapely.geometry import Point, MultiPoint
import h5py
from . import h5ad
from .cell import Cell
from .gene import Gene
from ..log_manager import logger
import copy
from anndata import AnnData
#from ..io.reader import read

class StereoExpData(Data):
    def __init__(
            self,
            file_path: Optional[str] = None,
            file_format: Optional[str] = None,
            bin_type: Optional[str] = None,
            bin_size: int = 100,
            exp_matrix: Optional[Union[np.ndarray, spmatrix]] = None,
            genes: Optional[Union[np.ndarray, Gene]] = None,
            cells: Optional[Union[np.ndarray, Cell]] = None,
            position: Optional[np.ndarray] = None,
            output: Optional[str] = None,
            partitions: int = 1,
            anndata: Optional[AnnData] = None):
        """
        a Data designed for express matrix of spatial omics. It can directly set the corresponding properties
        information to initialize the data. If the file path is not None, we will read the file information to
        initialize the properties.

        :param file_path: the path of express matrix file.
        :param file_format: the file format of the file_path.
        :param bin_type: the type of bin, if file format is stereo-seq file. `bins` or `cell_bins`.
        :param bin_size: size of bin to merge if bin type is 'bins'.
        :param exp_matrix: the express matrix.
        :param genes: the gene object which contain some info of gene.
        :param cells: the cell object which contain some info of cell.
        :param position: the spatial location.
        :param output: the path of output.
        :param partitions: the number of multi-process cores, used when processing files in parallel.
        :param anndata: object in Anndata format which will be transformed into StereoExpData
        """
        super(StereoExpData, self).__init__(file_path=file_path, file_format=file_format,
                                            partitions=partitions, output=output)
        self._exp_matrix = exp_matrix
        self._genes = genes if isinstance(genes, Gene) else Gene(gene_name=genes)
        self._cells = cells if isinstance(cells, Cell) else Cell(cell_name=cells)
        self._position = position
        self._bin_type = bin_type
        self.bin_size = bin_size
        # self.init()

    def init(self):
        self.check()
        if self.file is not None:
            logger.info("init the property of StereoData from a file, which take some time if file is too large.")
            #read(self,bin_size=self.bin_size)
        logger.info("init finish.")

    def sub_by_index(self, cell_index=None, gene_index=None):
        """
        get sub data by cell index or gene index list.

        :param cell_index: a list of cell index.
        :param gene_index: a list of gene index.
        :return:
        """
        if cell_index is not None:
            self.exp_matrix = self.exp_matrix[cell_index, :]
            self.position = self.position[cell_index, :]
            self.cells = self.cells.sub_set(cell_index)
        if gene_index is not None:
            self.exp_matrix = self.exp_matrix[:, gene_index]
            self.genes = self.genes.sub_set(gene_index)
        return self

    def sub_by_name(self, cell_name: Optional[Union[np.ndarray, list]] = None,
                    gene_name: Optional[Union[np.ndarray, list]] = None):
        """
        get sub data by cell name list or gene name list.

        :param cell_name: a list of cell name.
        :param gene_name: a list of gene name.
        :return:
        """
        data = copy.deepcopy(self)
        cell_index = [np.argwhere(data.cells.cell_name == i)[0][0] for i in cell_name] \
            if cell_name is not None else None
        gene_index = [np.argwhere(data.genes.gene_name == i)[0][0] for i in
                      gene_name] if gene_name is not None else None
        return data.sub_by_index(cell_index, gene_index)

    def check(self):
        """
        checking whether the params is in the range.

        :return:
        """
        super(StereoExpData, self).check()
        self.bin_type_check(self._bin_type)

    @staticmethod
    def bin_type_check(bin_type):
        """
        check whether the bin type is in range.

        :param bin_type: bin type value, 'bins' or 'cell_bins'.
        :return:
        """
        if (bin_type is not None) and (bin_type not in ['bins', 'cell_bins']):
            logger.error(f"the bin type `{bin_type}` is not in the range, please check!")
            raise Exception

    @property
    def gene_names(self):
        """
        get the gene names.

        :return:
        """
        return self.genes.gene_name

    @property
    def cell_names(self):
        """
        get the cell names.

        :return:
        """
        return self.cells.cell_name

    @property
    def genes(self):
        """
        get the value of self._genes.

        :return:
        """
        return self._genes

    @genes.setter
    def genes(self, gene):
        """
        set the value of self._genes.

        :param gene: a object of Gene
        :return:
        """
        self._genes = gene

    @property
    def cells(self):
        """
        get the value of self._cells

        :return:
        """
        return self._cells

    @cells.setter
    def cells(self, cell):
        """
        set the value of self._cells.

        :param cell: a object of Cell
        :return:
        """
        self._cells = cell

    @property
    def exp_matrix(self):
        """
        get the value of self._exp_matrix.

        :return:
        """
        return self._exp_matrix

    @exp_matrix.setter
    def exp_matrix(self, pos_array):
        """
        set the value of self._exp_matrix.

        :param pos_array: np.ndarray or sparse.spmatrix.
        :return:
        """
        self._exp_matrix = pos_array

    @property
    def bin_type(self):
        """
        get the value of self._bin_type.

        :return:
        """
        return self._bin_type

    @bin_type.setter
    def bin_type(self, b_type):
        """
        set the value of self._bin_type.

        :param b_type: the value of bin type, 'bins' or 'cell_bins'.
        :return:
        """
        self.bin_type_check(b_type)
        self._bin_type = b_type

    @property
    def position(self):
        """
        get the value of self._position.

        :return:
        """
        return self._position

    @position.setter
    def position(self, pos):
        """
        set the value of self._position.

        :param pos: the value of position, a np.ndarray .
        :return:
        """
        self._position = pos

    def parse_bin_coor(self, df, bin_size):
        """
        merge bins to a bin unit according to the bin size, also calculate the center coordinate of bin unit,
        and generate cell id of bin unit using the coordinate after merged.

        :param df: a dataframe of the bin file.
        :param bin_size: the size of bin to merge.
        :return:
        """
        x_min = df['x'].min()
        y_min = df['y'].min()
        df['bin_x'] = self.merge_bin_coor(df['x'].values, x_min, bin_size)
        df['bin_y'] = self.merge_bin_coor(df['y'].values, y_min, bin_size)
        df['cell_id'] = df['bin_x'].astype(str) + '_' + df['bin_y'].astype(str)
        df['x_center'] = self.get_bin_center(df['bin_x'], x_min, bin_size)
        df['y_center'] = self.get_bin_center(df['bin_y'], y_min, bin_size)
        return df

    def parse_cell_bin_coor(self, df):
        gdf = df.groupby('cell_id').apply(lambda x: self.make_multipoint(x))
        return gdf

    @staticmethod
    def make_multipoint(x):
        p = [Point(i) for i in zip(x['x'], x['y'])]
        mlp = MultiPoint(p).convex_hull
        x_center = mlp.centroid.x
        y_center = mlp.centroid.y
        return pd.Series({'cell_point': mlp, 'x_center': x_center, 'y_center': y_center})

    @staticmethod
    def merge_bin_coor(coor: np.ndarray, coor_min: int, bin_size: int):
        return np.floor((coor-coor_min)/bin_size).astype(np.int)

    @staticmethod
    def get_bin_center(bin_coor: np.ndarray, coor_min: int, bin_size: int):
        return bin_coor*bin_size+coor_min+int(bin_size/2)

    def to_df(self):
        df = pd.DataFrame(
            self.exp_matrix.toarray() if issparse(self.exp_matrix) else self.exp_matrix,
            columns=self.gene_names,
            index=self.cell_names
        )
        return df

    def read_by_bulk(self):
        pass


    def to_andata(self):
        andata = AnnData(X=self.exp_matrix,
                         obs=self.cells.to_df(),
                         var=self.genes.to_df(),
                         )
        return andata

