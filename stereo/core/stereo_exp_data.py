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
from typing import Union, Sequence, Optional, Tuple
from scipy.sparse import spmatrix, issparse
from .cell import Cell
from .gene import Gene
from ..log_manager import logger
from copy import deepcopy
from .st_pipeline import StPipeline


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
        self._bin_type = bin_type
        self.bin_size = bin_size
        self.tl = StPipeline(self)
        self.plt = self.get_plot()
        self.raw = None
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._attr = attr

    def get_plot(self):
        """
        import plot function

        :return:
        """
        from ..plots.plot_collection import PlotCollection

        return PlotCollection(self)

    def sub_by_index(self, cell_index=None, gene_index=None):
        """
        get sub data by cell index or gene index list.

        :param cell_index: a list of cell index.
        :param gene_index: a list of gene index.
        :return:
        """
        data = self.copy()
        if cell_index is not None:
            data.exp_matrix = data.exp_matrix[cell_index, :]
            data.position = data.position[cell_index, :] if data.position is not None else None
            data.cells = data.cells.sub_set(cell_index)
        if gene_index is not None:
            data.exp_matrix = data.exp_matrix[:, gene_index]
            data.genes = data.genes.sub_set(gene_index)
        return data

    def sub_by_name(self, cell_name: Optional[Union[np.ndarray, list]] = None,
                    gene_name: Optional[Union[np.ndarray, list]] = None):
        """
        get sub data by cell name list or gene name list.

        :param cell_name: a list of cell name.
        :param gene_name: a list of gene name.
        :return:
        """
        data = self.copy()
        return data[cell_name, gene_name]

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
        
    def __getitem__(self, index):
        cell_index, gene_index = self.get_data_index(index)
        return self.sub_by_index(cell_index=cell_index, gene_index=gene_index)
        
    def get_data_index(self, index):
        if isinstance(index, tuple):
            if len(index) == 1:
                index_0, index_1 = index[0], slice(None)
            
            elif len(index) == 2:
                index_0, index_1 = index
            
            else:
                raise ValueError('data can only be sliced with 2 dimensions')
            
        else: 
            index_0, index_1 = index, slice(None)

        index_0 = self._normalize_index(index_0, self.cell_names)
        index_1 = self._normalize_index(index_1, self.gene_names)
        return index_0, index_1

    @property
    def shape(self):
        """
        get the shape of self._exp_matrix.

        :return:
        """
        return self._exp_matrix.shape
    
    def __repr__(self):
        """
        pretty print.
        """
        from rich import print
        from rich.tree import Tree
        from rich.text import Text
        from rich.padding import Padding
        
        tree = Tree(f"[bold #88cc00]StereoExpData "
                    f"{self.shape[0]} cells ✖ {self.shape[1]} genes", 
                    guide_style="#b1b300")

        cell_description = 'cells'
        gene_description = 'genes'
        result_description = 'tl.result'

        gene_tree = tree.add(gene_description)
        cell_tree = tree.add(cell_description)
        result_tree = tree.add(result_description)

        cell_prop = ['total_counts', 
                    'pct_counts_mt', 
                    'n_genes_by_counts']
        cell_prop = [cell_qc for cell_qc in cell_prop
                    if self.cells.__dict__.get(cell_qc) is not None]
        cell_tree.add(Text(' '.join(cell_prop)), style='dim #ff5050') #ff5050

        gene_prop = ['n_cells', 'n_counts']
        gene_prop = [gene_qc for gene_qc in gene_prop
                    if self.genes.__dict__.get(gene_qc) is not None]
        gene_tree.add(Text(' '.join(gene_prop)), style='dim #ff5050') #ff5050

        result_prop = ['hvg', 'pca', 
                    'neighbors', 'umap', 
                    'cluster', 'marker_genes']
        result_prop = [result for result in result_prop 
                    if len(self.tl.__dict__['key_record'][result]) > 0]

        result_tree.add(Text(' '.join(result_prop)), style='dim')
        result_tree.add(' '.join(self.tl.result.keys()))
        if self.position is not None:
            tree.add("position")

        print(Padding.indent(tree, 5), end='')
        
        return ' '
    
    def copy(self):
       return deepcopy(self)
    
    @staticmethod    
    def _normalize_index(
        indexer: Union[
            slice,
            np.integer,
            int,
            str,
            list,
            Sequence[Union[int, np.integer]],
            np.ndarray,
            pd.Index,
        ],
        index: np.ndarray,
    ) -> Union[slice, int, np.ndarray]:  # ndarray of int
        # reference: anndata
        """
        normalize the index
        """
        if isinstance(indexer, list):
            indexer = np.array(indexer)
        
        # if not isinstance(index, pd.RangeIndex):
        #     assert (
        #         index.dtype != float and index.dtype != int
        #     ), "Don't call _normalize_index with non-categorical/string names"

        # the following is insanely slow for sequences,
        def name_idx(i):
            if isinstance(i, str):
                i = np.where(index == i)
            return i

        if isinstance(indexer, slice):
            start = name_idx(indexer.start)
            stop = name_idx(indexer.stop)
            # string slices can only be inclusive, so +1 in that case
            if isinstance(indexer.stop, str):
                stop = stop if stop is None else stop + 1
            step = indexer.step
            return slice(start, stop, step)
        
        elif isinstance(indexer, (np.integer, int)):
            return indexer
        
        elif isinstance(indexer, str):
            return np.where(index == indexer)  # int
        
        elif isinstance(indexer, (Sequence, np.ndarray, pd.Index, np.matrix)):
            if hasattr(indexer, "shape") and (
                (indexer.shape == (index.shape[0], 1))
                or (indexer.shape == (1, index.shape[0]))
            ):
                indexer = np.ravel(indexer)

            if issubclass(indexer.dtype.type, (np.integer, np.floating)):
                return indexer  # Might not work for range indexes
            
            elif issubclass(indexer.dtype.type, np.bool_):
                print(indexer.shape, index.shape)
                if indexer.shape != index.shape:
                    raise IndexError(
                        f"Boolean index does not match Data’s shape along this "
                        f"dimension. Boolean index has shape {indexer.shape} while "
                        f"data index has shape {index.shape}."
                    )
                positions = np.where(indexer)[0]
                return positions  # np.ndarray[int]
            else:  # indexer should be string array
                positions = np.where(np.in1d(index, indexer))[0]
                inter = np.in1d(indexer, index)

                if not np.all(inter):
                    not_found = np.array(indexer)[~inter]
                    raise KeyError(
                        f"Values {list(not_found)}, from {list(indexer)}, "
                        "are not valid cell/gene names or indices."
                    )
                return positions  # np.ndarray[int]
        else:
            raise IndexError(f"Unknown indexer {indexer!r} of type {type(indexer)}")
        

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
