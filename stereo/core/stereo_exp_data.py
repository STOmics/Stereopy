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
import copy
from warnings import warn
from typing import Optional, Union

import pandas as pd
import numpy as np
from scipy.sparse import spmatrix, issparse, csr_matrix

from .data import Data
from .cell import Cell, AnnBasedCell
from .gene import Gene, AnnBasedGene
from ..log_manager import logger


class StereoExpData(Data):
    def __init__(
            self,
            file_path: Optional[str] = None,
            file_format: Optional[str] = None,
            bin_type: Optional[str] = None,
            bin_size: Optional[int] = 100,
            exp_matrix: Optional[Union[np.ndarray, spmatrix]] = None,
            genes: Optional[Union[np.ndarray, Gene]] = None,
            cells: Optional[Union[np.ndarray, Cell]] = None,
            position: Optional[np.ndarray] = None,
            output: Optional[str] = None,
            partitions: Optional[int] = 1,
            offset_x: Optional[str] = None,
            offset_y: Optional[str] = None,
            attr: Optional[dict] = None,
            merged: bool = False
    ):

        """
        The core data object is designed for expression matrix of spatial omics, which can be set 
        corresponding properties directly to initialize the data. 

        Parameters
        -------------------
        file_path
            the path to input file of expression matrix.
        file_format
            the format of input file.
        bin_type
            the type of bin, if the file format is Stereo-seq file including `'bins'` or `'cell_bins'`.
        bin_size
            the size of the bin to merge, when `bin_type` is `'bins'`.
        exp_matrix
            the expression matrix.
        genes
            the gene object which contains information of gene level.
        cells
            the cell object which contains information of cell level.
        position
            spatial location information.
        output
            the path to output file.
        partitions
            the number of multi-process cores, used when processing files in parallel.
        offset_x
            the x value of the offset . 
        offset_y
            the y value of the offset .
        attr
            attribute information from GEF file.

        """
        super(StereoExpData, self).__init__(file_path=file_path, file_format=file_format,
                                            partitions=partitions, output=output)
        self._exp_matrix = exp_matrix
        self._genes = genes if isinstance(genes, Gene) else Gene(gene_name=genes)
        self._cells = cells if isinstance(cells, Cell) else Cell(cell_name=cells)
        self._position = position
        self._position_offset = None
        self._bin_type = bin_type
        self.bin_size = bin_size
        self._tl = None
        self._plt = None
        self.raw = None
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._attr = attr
        self._merged = merged
        self._sn = self.get_sn_from_path(file_path)

    def get_sn_from_path(self, file_path):
        """
        Get the SN information of input file.
        """
        if file_path is None:
            return None

        from os import path
        return path.basename(file_path).split('.')[0].strip()

    @property
    def plt(self):
        """
        Call the visualization module.        
        """
        if self._plt is None:
            from ..plots.plot_collection import PlotCollection
            self._plt = PlotCollection(self)
        return self._plt

    @property
    def tl(self):
        """
        call StPipeline method.
        """
        if self._tl is None:
            from .st_pipeline import StPipeline
            self._tl = StPipeline(self)
        return self._tl

    def sub_by_index(self, cell_index=None, gene_index=None):
        """
        Get data subset by indexl list of cells or genes.

        :param cell_index: a list of cell index.
        :param gene_index: a list of gene index.
        :return:
        """
        if cell_index is not None:
            self.exp_matrix = self.exp_matrix[cell_index, :]
            self.position = self.position[cell_index, :] if self.position is not None else None
            self.cells = self.cells.sub_set(cell_index)
        if gene_index is not None:
            self.exp_matrix = self.exp_matrix[:, gene_index]
            self.genes = self.genes.sub_set(gene_index)
        return self

    def sub_by_name(self, cell_name: Optional[Union[np.ndarray, list]] = None,
                    gene_name: Optional[Union[np.ndarray, list]] = None):
        """
        Get data subset by name list of cells or genes.

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

    def sub_exp_matrix_by_name(
            self,
            cell_name: Optional[Union[np.ndarray, list, str, int]] = None,
            gene_name: Optional[Union[np.ndarray, list, str]] = None,
            order_preserving: bool = True
    ) -> Union[np.ndarray, spmatrix]:
        new_exp_matrix = self.exp_matrix
        if cell_name is not None:
            if isinstance(cell_name, str) or isinstance(cell_name, int):
                cell_name = [cell_name]
            if order_preserving:
                index = [np.argwhere(self.cell_names == c)[0][0] for c in cell_name]
            else:
                index = np.isin(self.cell_names, cell_name)
            new_exp_matrix = new_exp_matrix[index]
        if gene_name is not None:
            if isinstance(gene_name, str):
                gene_name = [gene_name]
            if order_preserving:
                index = [np.argwhere(self.gene_names == g)[0][0] for g in gene_name]
            else:
                index = np.isin(self.gene_names, gene_name)
            new_exp_matrix = new_exp_matrix[:, index]
        return new_exp_matrix
            

    def check(self):
        """
        Check whether the parameters meet the requirement.

        :return:
        """
        super(StereoExpData, self).check()
        self.bin_type_check(self._bin_type)

    @staticmethod
    def bin_type_check(bin_type):
        """
        Check whether the bin type is from specific options.

        :param bin_type: bin type value, 'bins' or 'cell_bins'.
        :return:
        """
        if (bin_type is not None) and (bin_type not in ['bins', 'cell_bins']):
            logger.error(f"the bin type `{bin_type}` is not in the range, please check!")
            raise Exception

    @property
    def shape(self):
        """
        Get the shape of expression matrix.

        :return:
        """
        return self.exp_matrix.shape

    @property
    def gene_names(self):
        """
        Get the gene names.

        :return:
        """
        return self.genes.gene_name

    @property
    def cell_names(self):
        """
        Get the cell names.

        :return:
        """
        return self.cells.cell_name

    @property
    def cell_borders(self):
        """
        Get the cell borders.
        """
        return self.cells.cell_border

    @property
    def genes(self):
        """
        Get the gene object.

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
    def genes_matrix(self):
        """
        Get the genes matrix.
        """
        return self._genes._matrix

    @property
    def genes_pairwise(self):
        """
        Get the genes pairwise.
        """
        return self._genes._pairwise

    @property
    def cells(self):
        """
        Get the cell object.

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
    def cells_matrix(self):
        """
        Get the cells matrix.
        """
        return self._cells._matrix

    @property
    def cells_pairwise(self):
        """
        Get the cells pairwise.
        """
        return self._cells._pairwise

    @property
    def exp_matrix(self) -> Union[np.ndarray, spmatrix]:
        """
        Get the expression matrix.

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
        Get the bin type.

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
        Get the information of spatial location.

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

    @property
    def position_offset(self):
        """
        Get the offset of position in gef.

        """
        return self._position_offset

    @position_offset.setter
    def position_offset(self, position_offset):
        self._position_offset = position_offset

    @property
    def offset_x(self):
        """
        Get the x value of the offset.

        :return:
        """
        return self._offset_x

    @offset_x.setter
    def offset_x(self, min_x):
        """

        :param min_x: offset of x.
        :return:
        """
        self._offset_x = min_x

    @property
    def offset_y(self):
        """
        Get the y value of the offset.

        :return:
        """
        return self._offset_y

    @offset_y.setter
    def offset_y(self, min_y):
        """

        :param min_y: offset of y.
        :return:
        """
        self._offset_y = min_y

    @property
    def attr(self):
        """
        Get the attribute information.

        :return:
        """
        return self._attr

    @attr.setter
    def attr(self, attr):
        """

        :param attr: dict of attr.
        :return:
        """
        self._attr = attr

    @property
    def merged(self):
        """
        Get the flag whether merged.
        """
        return self._merged

    @merged.setter
    def merged(self, merged):
        self._merged = merged

    @property
    def sn(self):
        """
        Get the sample name.
        """
        return self._sn

    @sn.setter
    def sn(self, sn):
        self._sn = sn

    def to_df(self):
        """
        Transform StereoExpData object to pd.DataFrame.

        :return:
        """
        df = pd.DataFrame(
            self.exp_matrix.toarray() if issparse(self.exp_matrix) else self.exp_matrix,
            columns=self.gene_names,
            index=self.cell_names
        )
        return df

    def sparse2array(self):
        """
        Transform expression matrix to array if it is parse matrix.

        :return:
        """
        if issparse(self.exp_matrix):
            self.exp_matrix = self.exp_matrix.toarray()
        return self.exp_matrix

    def array2sparse(self):
        """
        Transform expression matrix to sparse matrix if it is ndarray.

        :return:
        """
        if not issparse(self.exp_matrix):
            self.exp_matrix = csr_matrix(self.exp_matrix)
        return self.exp_matrix

    def __str__(self):
        format_str = f"StereoExpData object with n_cells X n_genes = {self.shape[0]} X {self.shape[1]}"
        format_str += f"\nbin_type: {self.bin_type}"
        if self.bin_type == 'bins':
            format_str += f"\n{'bin_size: %d' % self.bin_size}"
        format_str += f"\noffset_x = {self.offset_x}"
        format_str += f"\noffset_y = {self.offset_y}"
        format_str += str(self.cells)
        format_str += str(self.genes)
        if self.cells_matrix:
            format_str += f"\ncells_matrix = {list(self.cells_matrix.keys())}"
        if self.genes_matrix:
            format_str += f"\ngenes_matrix = {list(self.cells_matrix.keys())}"
        if self.cells_pairwise:
            format_str += f"\ncells_pairwise = {list(self.cells._pairwise.keys())}"
        if self.genes_pairwise:
            format_str += f"\ngenes_pairwise = {list(self.genes._pairwise.keys())}"
        format_key_record = {
            key: value
            for key, value in self.tl.key_record.items() if value
        }
        warn(
            'FutureWarning: `pca`, `neighbors`, `cluster`, `umap` will be inaccessible in result in future version.'
            '\nMake sure your code access result from the right property, such as `pca` and `umap` will be in the '
            '`StereoExpData.cells_matrix`.',
            category=FutureWarning
        )
        if format_key_record:
            format_str += f"\nkey_record: {format_key_record}"
        return format_str

    def __repr__(self):
        return self.__str__()

    def issparse(self):
        """
        Check whether the matrix is sparse matrix type.
        """
        return issparse(self.exp_matrix)


class AnnBasedStereoExpData(StereoExpData):

    def __init__(self, h5ad_file_path: str, *args, **kwargs):
        if 'based_ann_data' in kwargs:
            based_ann_data = kwargs.pop('based_ann_data')
        else:
            based_ann_data = None
        super(AnnBasedStereoExpData, self).__init__(*args, **kwargs)
        import anndata
        if based_ann_data:
            assert type(based_ann_data) is anndata.AnnData
            self._ann_data = based_ann_data
        else:
            self._ann_data = anndata.read_h5ad(h5ad_file_path)
        self._genes = AnnBasedGene(self._ann_data, self._genes.gene_name)
        self._cells = AnnBasedCell(self._ann_data, self._cells.cell_name)
        from .st_pipeline import AnnBasedStPipeline
        self._tl = AnnBasedStPipeline(self._ann_data, self)

    def __str__(self):
        return str(self._ann_data)

    def __repr__(self):
        return self.__str__()

    @property
    def exp_matrix(self):
        return self._ann_data.X

    @exp_matrix.setter
    def exp_matrix(self, pos_array: spmatrix):
        self._ann_data.X = pos_array

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, gene: AnnBasedGene):
        self._genes = gene

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cell: AnnBasedCell):
        self._cells = cell

    @property
    def plt(self):
        """
        Call the visualization module. 
        """
        if self._plt is None:
            from ..plots.plot_collection import PlotCollection
            self._plt = PlotCollection(self)
        return self._plt

    @property
    def tl(self):
        """
        call StPipeline method.
        """
        return self._tl

    @property
    def position(self):
        if {'x', 'y'} - set(self._ann_data.obs.columns.values):
            self._ann_data.obs.loc[:, ['x', 'y']] = \
                np.array(list(self._ann_data.obs.index.str.split('-', expand=True)), dtype=np.uint32)
        return self._ann_data.obs.loc[:, ['x', 'y']].values

    def sub_by_name(self, cell_name: Optional[Union[np.ndarray, list]] = None,
                    gene_name: Optional[Union[np.ndarray, list]] = None):
        self._ann_data.obs_names_make_unique()
        self._ann_data.var_names_make_unique()
        if cell_name:
            self._ann_data._inplace_subset_obs(cell_name)
        if gene_name:
            self._ann_data._inplace_subset_var(gene_name)
        return self
