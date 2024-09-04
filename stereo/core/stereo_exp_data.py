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
from typing import Optional
from typing import Union
from warnings import warn

import anndata
import numba
import numpy as np
import pandas as pd
from scipy.sparse import (
    spmatrix,
    issparse,
    csr_matrix
)

from .cell import AnnBasedCell
from .cell import Cell
from .data import Data
from .gene import AnnBasedGene
from .gene import Gene
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
            position_z: Optional[np.ndarray] = None,
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
        self._raw_position = None
        self._position = position
        self._position_z = position_z
        self._position_offset = None
        self._position_min = None
        self._bin_type = bin_type
        self._bin_size = bin_size
        self._tl = None
        self._plt = None
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._attr = attr if attr is not None else {'resolution': 500}
        self._merged = merged
        self._sn = self.get_sn_from_path(file_path)
        self.center_coordinates = False

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
            self.position_z = self.position_z[cell_index] if self.position_z is not None else None
            self.cells = self.cells.sub_set(cell_index)
        if gene_index is not None:
            self.exp_matrix = self.exp_matrix[:, gene_index]
            self.genes = self.genes.sub_set(gene_index)
        return self

    @numba.jit(cache=True, forceobj=True, nogil=True)
    def get_index(self, data, names):
        index_list = []
        for name in names:
            index_list.append(np.argwhere(data == name)[0][0])

        return index_list

    def sub_by_name(self, cell_name: Optional[Union[np.ndarray, list]] = None,
                    gene_name: Optional[Union[np.ndarray, list]] = None):
        """
        Get data subset by name list of cells or genes.

        :param cell_name: a list of cell name.
        :param gene_name: a list of gene name.
        :return:
        """
        data = copy.deepcopy(self)
        cell_index, gene_index = None, None
        if cell_name is not None:
            cell_index = self.cells.obs.index.get_indexer(cell_name)
            cell_index = cell_index[cell_index != -1]
        if gene_name is not None:
            gene_index = self.genes.var.index.get_indexer(gene_name)
            gene_index = gene_index[gene_index != -1]
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
        return self._genes.matrix

    @property
    def genes_pairwise(self):
        """
        Get the genes pairwise.
        """
        return self._genes.pairwise

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
        return self._cells.matrix

    @property
    def cells_pairwise(self):
        """
        Get the cells pairwise.
        """
        return self._cells.pairwise

    @property
    def n_cells(self):
        """
        Get the number of cells.

        :return:
        """
        return self.cells.size

    @property
    def n_genes(self):
        """
        Get the number of genes.

        :return:
        """
        return self.genes.size

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
    def bin_size(self):
        """
        Get the bin size.

        :return:
        """
        return self._bin_size

    @bin_size.setter
    def bin_size(self, bin_size):
        self._bin_size = bin_size

    @property
    def raw_position(self):
        return self._raw_position

    @raw_position.setter
    def raw_position(self, pos):
        self._raw_position = pos

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
    def position_z(self):
        return self._position_z

    @position_z.setter
    def position_z(self, position_z):
        self._position_z = position_z

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
    def position_min(self):
        return self._position_min

    @position_min.setter
    def position_min(self, position_min):
        self._position_min = position_min

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

    @property
    def raw(self):
        return self.tl.raw

    @property
    def resolution(self):
        if self.attr is not None and 'resolution' in self.attr:
            return self.attr['resolution']
        else:
            return None

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
            format_str += f"\ngenes_matrix = {list(self.genes_matrix.keys())}"
        if self.cells_pairwise:
            format_str += f"\ncells_pairwise = {list(self.cells_pairwise.keys())}"
        if self.genes_pairwise:
            format_str += f"\ngenes_pairwise = {list(self.genes_pairwise.keys())}"
        format_key_record = {
            key: value
            for key, value in self.tl.key_record.items() if value
        }
        # warn(
        #     'FutureWarning: `pca`, `neighbors`, `cluster`, `umap` will be inaccessible in result in future version.'
        #     '\nMake sure your code access result from the right property, such as `pca` and `umap` will be in the '
        #     '`StereoExpData.cells_matrix`.',
        #     category=FutureWarning
        # )
        if format_key_record:
            format_str += f"\nkey_record: {format_key_record}"
        result_key = []
        for rks in self.tl.key_record.values():
            if rks is not None and len(rks) > 0:
                result_key += rks
        for rk in self.tl.result.keys():
            if rk not in result_key:
                result_key.append(rk)
        format_str += f"\nresult: {result_key}"
        return format_str

    def __repr__(self):
        return self.__str__()

    def issparse(self):
        """
        Check whether the matrix is sparse matrix type.
        """
        return issparse(self.exp_matrix)

    def reset_position(self):
        if self.position_offset is not None:
            batches = np.unique(self.cells.batch)
            position = self.position
            for bno in batches:
                idx = np.where(self.cells.batch == bno)[0]
                position[idx] -= self.position_offset[bno]
                position[idx] += self.position_min[bno]
            self.position = position
        self.position_offset = None
        self.position_min = None

    def __add__(self, other):
        from stereo.core.ms_data import MSData
        if isinstance(other, StereoExpData):
            return MSData([self, other])
        elif isinstance(other, MSData):
            return other.__add__(self)
        else:
            raise TypeError("only support StereoExpData and MSData!")
    
    def write(self, filename, to_anndata=False, **kwargs):
        if 'output' in kwargs:
            del kwargs['output']
        kwargs.setdefault('output', filename)
        kwargs.setdefault('split_batches', False)
        if to_anndata:
            from stereo.io.reader import stereo_to_anndata
            stereo_to_anndata(self, **kwargs)
        else:
            from stereo.io.writer import write_h5ad
            write_h5ad(self, **kwargs)
    
    def to_ann_based(self):
        from stereo.io.reader import stereo_to_anndata
        adata = stereo_to_anndata(self, flavor='scanpy', split_batches=False)
        return AnnBasedStereoExpData(based_ann_data=adata)
        


class AnnBasedStereoExpData(StereoExpData):

    def __init__(
            self,
            h5ad_file_path: str = None,
            based_ann_data: anndata.AnnData = None,
            bin_type: str = None,
            bin_size: int = None,
            spatial_key: str = 'spatial',
            *args,
            **kwargs
    ):
        super(AnnBasedStereoExpData, self).__init__(*args, **kwargs)
        if h5ad_file_path is None and based_ann_data is None:
            raise Exception("Must to input the 'h5ad_file_path' or 'based_ann_data'.")

        if h5ad_file_path is not None and based_ann_data is not None:
            raise Exception("'h5ad_file_path' and 'based_ann_data' only can input one of them")

        if based_ann_data:
            assert type(based_ann_data) is anndata.AnnData
            self._ann_data = based_ann_data
        else:
            self._ann_data = anndata.read_h5ad(h5ad_file_path)
        self._genes = AnnBasedGene(self._ann_data, self._genes.gene_name)
        self._cells = AnnBasedCell(self._ann_data, self._cells.cell_name)

        if 'resolution' in self._ann_data.uns:
            self.attr = {'resolution': self._ann_data.uns['resolution']}
            del self._ann_data.uns['resolution']
        
        if 'merged' in self._ann_data.uns:
            self.merged = self._ann_data.uns['merged']
            del self._ann_data.uns['merged']

        if bin_type is not None and 'bin_type' not in self._ann_data.uns:
            self._ann_data.uns['bin_type'] = bin_type

        if bin_size is not None and 'bin_size' not in self._ann_data.uns:
            self._ann_data.uns['bin_size'] = bin_size

        if h5ad_file_path is not None and 'sn' not in self._ann_data.uns:
            sn = self.get_sn_from_path(h5ad_file_path)
            if sn is not None:
                self._ann_data.uns['sn'] = pd.DataFrame([[-1, sn]], columns=['batch', 'sn'])

        from .st_pipeline import AnnBasedStPipeline
        self._tl = AnnBasedStPipeline(self._ann_data, self)
        if 'key_record' in self._ann_data.uns:
            key_record = self._ann_data.uns['key_record']
            self._tl._key_record = self._ann_data.uns['key_record'] = {key: list(value) for key, value in key_record.items()}

        if self._ann_data.raw:
            self._tl._raw = AnnBasedStereoExpData(based_ann_data=self._ann_data.raw.to_adata())

        self.spatial_key = spatial_key
        self.file_format = 'h5ad'

    def __str__(self):
        return str(self._ann_data)

    def __repr__(self):
        return self.__str__()

    # def __getattr__(self, name: str):
    #     if name.startswith('__'):
    #         raise AttributeError
    #     if hasattr(self._ann_data, name):
    #         return getattr(self._ann_data, name)
    #     else:
    #         return None

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
        if self.spatial_key in self._ann_data.obsm:
            return self._ann_data.obsm[self.spatial_key][:, [0, 1]]
        elif 'x' in self._ann_data.obs.columns and 'y' in self._ann_data.obs.columns:
            return self._ann_data.obs[['x', 'y']].to_numpy()
        return None

    # @position.setter
    # def position(self, pos):
    #     if 'spatial' in self._ann_data.obsm:
    #         self._ann_data.obsm['spatial'][:, [0, 1]] = pos

    @property
    def position_z(self):
        if self.spatial_key in self._ann_data.obsm:
            if self._ann_data.obsm[self.spatial_key].shape[1] >= 3:
                return self._ann_data.obsm[self.spatial_key][:, [2]]
            else:
                return None
        elif 'z' in self._ann_data.obs.columns:
            return self._ann_data.obs[['z']].to_numpy()
        return None

    @position.setter
    def position(self, position: np.ndarray):
        if len(position.shape) != 2:
            raise ValueError("the shape of position must be 2 dimensions.")
        if position.shape[1] != 2:
            raise ValueError("the length of position's second dimension must be 2.")
        if self.spatial_key in self._ann_data.obsm:
            self._ann_data.obsm[self.spatial_key][:, [0, 1]] = position
        elif 'x' in self._ann_data.obs.columns and 'y' in self._ann_data.obs.columns:
            self._ann_data.obs['x'] = position[:, 0]
            self._ann_data.obs['y'] = position[:, 1]
        else:
            self._ann_data.obsm[self.spatial_key] = position

    @position_z.setter
    def position_z(self, position_z: np.ndarray):
        if (position_z.shape) == 1:
            position_z = position_z.reshape(-1, 1)
        if self.spatial_key in self._ann_data.obsm:
            self._ann_data.obsm[self.spatial_key] = np.concatenate(
                [self._ann_data.obsm[self.spatial_key][:, [0, 1]], position_z], axis=1)
        else:
            self._ann_data.obs['z'] = position_z

    @property
    def bin_type(self):
        return self._ann_data.uns.get('bin_type', 'bins')

    @bin_type.setter
    def bin_type(self, bin_type):
        self.bin_type_check(bin_type)
        self._ann_data.uns['bin_type'] = bin_type

    @property
    def bin_size(self):
        return self._ann_data.uns.get('bin_size', 1)

    @bin_size.setter
    def bin_size(self, bin_size):
        self._ann_data.uns['bin_size'] = bin_size

    @property
    def sn(self):
        sn = None
        if 'sn' in self._ann_data.uns:
            sn_data: pd.DataFrame = self._ann_data.uns['sn']
            if sn_data.shape[0] == 1:
                sn = sn_data.iloc[0]['sn']
            else:
                sn = {}
                for _, row in sn_data.iterrows():
                    sn[row['batch']] = row['sn']
        return sn

    @sn.setter
    def sn(self, sn):
        if isinstance(sn, str):
            sn_list = [['-1', sn]]
        elif isinstance(sn, dict):
            sn_list = []
            for bno, sn in sn.items():
                sn_list.append([bno, sn])
        else:
            raise TypeError(f'sn must be type of str or dict, but now is {type(sn)}')
        self._ann_data.uns['sn'] = pd.DataFrame(sn_list, columns=['batch', 'sn'])

    def sub_by_index(self, cell_index=None, gene_index=None):
        if cell_index is not None:
            self._ann_data._inplace_subset_obs(cell_index)
        if gene_index is not None:
            self._ann_data._inplace_subset_var(gene_index)
        # if self._ann_data.raw:
        #     self.tl.raw = AnnBasedStereoExpData(based_ann_data=self._ann_data.raw.to_adata())
        return self

    def sub_by_name(
            self,
            cell_name: Optional[Union[np.ndarray, list]] = None,
            gene_name: Optional[Union[np.ndarray, list]] = None
    ):
        data = AnnBasedStereoExpData(self.file, based_ann_data=self._ann_data.copy())
        data._ann_data.obs_names_make_unique()
        data._ann_data.var_names_make_unique()
        if cell_name is not None:
            data._ann_data._inplace_subset_obs(cell_name)
        if gene_name is not None:
            data._ann_data._inplace_subset_var(gene_name)
        # if data._ann_data.raw:
        #     data.tl.raw = AnnBasedStereoExpData(based_ann_data=data._ann_data.raw.to_adata())
        return data

    # @staticmethod
    # def merge(*data, batch_key='batch'):
    #     from anndata import concat
    #     ann_data = concat([d._ann_data for d in data], axis=0, merge='same', label=batch_key, index_unique='-')
    #     return AnnBasedStereoExpData(based_ann_data=ann_data)
    
    
    @property
    def adata(self):
        return self._ann_data
    
    def write(self, filename, **kwargs):
        from stereo.io.reader import stereo_to_anndata
        if 'output' in kwargs:
            del kwargs['output']
        kwargs.setdefault('output', filename)
        kwargs.setdefault('split_batches', False)
        stereo_to_anndata(self, **kwargs)
