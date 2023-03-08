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
from scipy.sparse import spmatrix, issparse, csr_matrix
from .cell import Cell, AnnBasedCell
from .gene import Gene, AnnBasedGene
from ..log_manager import logger
import copy


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
            offset_x: Optional[str] = None,
            offset_y: Optional[str] = None,
            attr: Optional[dict] = None,
            merged: bool = False
    ):

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
        :param offset_x: the x of the offset.
        :param offset_y: the y of the offset.
        :param attr: attributions from gef file.
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
        if file_path is None:
            return None

        from os import path
        return path.basename(file_path).split('.')[0].strip()

    @property
    def plt(self):
        if self._plt is None:
            from ..plots.plot_collection import PlotCollection
            self._plt = PlotCollection(self)
        return self._plt

    @property
    def tl(self):
        if self._tl is None:
            from .st_pipeline import StPipeline
            self._tl = StPipeline(self)
        return self._tl

    def sub_by_index(self, cell_index=None, gene_index=None):
        """
        get sub data by cell index or gene index list.

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
    def shape(self):
        return self.exp_matrix.shape

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
    def cell_borders(self):
        return self.cells.cell_boder

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

    @property
    def position_offset(self):
        return self._position_offset

    @position_offset.setter
    def position_offset(self, position_offset):
        self._position_offset = position_offset

    @property
    def offset_x(self):
        """
        get the x of self._offset_x.

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
        get the offset_y of self._offset_y.

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
        get the attr of self._attr.

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
        return self._merged

    @merged.setter
    def merged(self, merged):
        self._merged = merged

    @property
    def sn(self):
        return self._sn

    @sn.setter
    def sn(self, sn):
        self._sn = sn

    def to_df(self):
        """
        transform StereoExpData to pd.DataFrame.

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
        transform expression matrix to array if it is parse matrix.

        :return:
        """
        if issparse(self.exp_matrix):
            self.exp_matrix = self.exp_matrix.toarray()
        return self.exp_matrix

    def array2sparse(self):
        """
        transform expression matrix to sparse matrix if it is ndarray

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
        format_cells = []
        for attr_name in [('_cell_name', 'cell_name'), 'total_counts', 'n_genes_by_counts', 'pct_counts_mt']:
            if type(attr_name) is tuple:
                real_name, show_name = attr_name[0], attr_name[1]
            else:
                real_name = show_name = attr_name
            # `is not None` is ugly but object in __dict__ may be a pandas.DataFrame
            if self.cells.__dict__.get(real_name, None) is not None:
                format_cells.append(show_name)
        if format_cells:
            format_str += f"\ncells: {format_cells}"
        format_genes = []
        for attr_name in [('_gene_name', 'gene_name'), 'n_counts', 'n_cells']:
            if type(attr_name) is tuple:
                real_name, show_name = attr_name[0], attr_name[1]
            else:
                real_name = show_name = attr_name
            if self.genes.__dict__.get(real_name, None) is not None:
                format_genes.append(show_name)
        if format_genes:
            format_str += f"\ngenes: {format_genes}"
        # TODO: no decide yet
        # format_str += "\nposition: T"
        format_key_record = {key: value for key, value in self.tl.key_record.items() if value}
        if format_key_record:
            format_str += f"\nkey_record: {format_key_record}"
        return format_str

    def __repr__(self):
        return self.__str__()
    
    def issparse(self):
        return issparse(self.exp_matrix)


class AnnBasedStereoExpData(StereoExpData):

    def __init__(self, h5ad_file_path: str, *args, **kwargs):
        super(AnnBasedStereoExpData, self).__init__(*args, **kwargs)
        import anndata
        self._ann_data = anndata.read_h5ad(h5ad_file_path)
        self._genes = AnnBasedGene(self._ann_data, self._genes._gene_name)
        self._cells = AnnBasedCell(self._ann_data, self._cells._cell_name)
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
        if self._plt is None:
            from ..plots.plot_collection import PlotCollection
            self._plt = PlotCollection(self)
        return self._plt

    @property
    def tl(self):
        return self._tl

    @property
    def position(self):
        if {'x', 'y'} - set(self._ann_data.obs.columns.values):
            self._ann_data.obs.loc[:, ['x', 'y']] = \
                np.array(list(self._ann_data.obs.index.str.split('-', expand=True)), dtype=np.uint32)
        return self._ann_data.obs.loc[:, ['x', 'y']].values
    

