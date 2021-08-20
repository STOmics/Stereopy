#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:reader.py
@time:2021/03/05

change log:
    2021/03/05  add read_stereo_data function , by Ping Qiu.
    2021/08/12  move read_txt functions from StereoExpData here. Add read_ann_h5ad, andata_to_stereo function by Yiran Wu.

"""
import pandas as pd
import os
from ..core.stereo_exp_data import StereoExpData
from ..log_manager import logger
import h5py
from ..core import h5ad
from scipy.sparse import spmatrix, csr_matrix, issparse
from ..core.cell import Cell
from ..core.gene import Gene
import numpy as np
from anndata import AnnData

def read_txt(file_path, sep='\t', bin_type="bins", bin_size=100, is_sparse=True):
    """
    read the stereo-seq file, and generate the object of StereoExpData.

    :param sep: separator string
    :param bin_type: the type of bin, if file format is stereo-seq file. `bins` or `cell_bins`.
    :param bin_size: the size of bin to merge. The parameter only takes effect
                     when the value of data.bin_type is 'bins'.
    :param is_sparse: the matrix is sparse matrix if is_sparse is True else np.ndarray

    :return: an object of StereoExpData.
    """
    data = StereoExpData(file_path=file_path, bin_type=bin_type)
    df = pd.read_csv(str(data.file), sep=sep, comment='#', header=0)
    if 'MIDCounts' in df.columns:
        df.rename(columns={'MIDCounts': 'UMICount'}, inplace=True)
    df.dropna(inplace=True)
    gdf = None
    if data.bin_type == 'cell_bins':
        df.rename(columns={'label': 'cell_id'}, inplace=True)
        gdf = data.parse_cell_bin_coor(df)
    else:
        df = data.parse_bin_coor(df, bin_size)
    cells = df['cell_id'].unique()
    genes = df['geneID'].unique()
    cells_dict = dict(zip(cells, range(0, len(cells))))
    genes_dict = dict(zip(genes, range(0, len(genes))))
    rows = df['cell_id'].map(cells_dict)
    cols = df['geneID'].map(genes_dict)
    logger.info(f'the martrix has {len(cells)} cells, and {len(genes)} genes.')
    exp_matrix = csr_matrix((df['UMICount'], (rows, cols)), shape=(cells.shape[0], genes.shape[0]), dtype=np.int)
    data.cells = Cell(cell_name=cells)
    data.genes = Gene(gene_name=genes)
    data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
    if data.bin_type == 'bins':
        data.position = df.loc[:, ['x_center', 'y_center']].drop_duplicates().values
    else:
        data.position = gdf.loc[cells][['x_center', 'y_center']].values
        data.cells.cell_point = gdf.loc[cells]['cell_point'].values
    return data


def read_stereo(file_path):
    """
    read the h5ad file, and generate the object of StereoExpData.
    :return:
    """
    data = StereoExpData(file_path=file_path)
    if not data.file.exists():
        logger.error('the input file is not exists, please check!')
        raise FileExistsError('the input file is not exists, please check!')
    with h5py.File(data.file, mode='r') as f:
        for k in f.keys():
            if k == 'cells':
                data.cells = h5ad.read_group(f[k])
            elif k == 'genes':
                data.genes = h5ad.read_group(f[k])
            elif k == 'position':
                data.position = h5ad.read_dataset(f[k])
            elif k == 'bin_type':
                data.bin_type = h5ad.read_dataset(f[k])
            elif k == 'exp_matrix':
                if isinstance(f[k], h5py.Group):
                    data.exp_matrix = h5ad.read_group(f[k])
                else:
                    data.exp_matrix = h5ad.read_dataset(f[k])
            else:
                pass
    return data

def read_ann_h5ad(file_path):
    """
    read the h5ad file in Anndata format, and generate the object of StereoExpData.
    :param file_path: h5ad file path.
    :return: StereoExpData obj.
    """
    data = StereoExpData(file_path=file_path)

    ## basic
    attributes = ["obsm", "varm", "obsp", "varp", "uns", "layers"]
    df_attributes = ["obs", "var"]

    with h5py.File(data.file, mode='r') as f:

        for k in f.keys():
            if k == "raw" or k.startswith("raw."):
                continue
            if k == "X" :
                if isinstance(f[k], h5py.Group):
                    data.exp_matrix = h5ad.read_group(f[k])
                else:
                    data.exp_matrix = h5ad.read_dataset(f[k])

            elif k == "raw":
                assert False, "unexpected raw format"
            elif k == "obs":
                cells_df = h5ad.read_dataframe(f[k])
                data.position = np.array(list(cells_df.index.str.split('-', expand=True)), dtype=np.int)
                data.cells.cell_name = cells_df.index.values
                data.cells.total_counts = cells_df['total_counts']
                data.cells.pct_counts_mt = cells_df['pct_counts_mt']
                data.cells.n_genes_by_counts = cells_df['n_genes_by_counts']
            elif k == "var":
                genes_df = h5ad.read_dataframe(f[k])
                data.genes.gene_name = genes_df.index.values
                #data.genes.n_cells = genes_df['n_cells']
                #data.genes.n_counts = genes_df['n_counts']
            else:  # Base case
                pass
    return data



def read(file_path,file_format,sep='\t', bin_type="bins", bin_size=100, is_sparse=True,ann_format=False):
    """
    read different format file and generate the object of StereoExpData.
    :param file_path: the path of express matrix file.
    :param file_fomat: the file format of the file_path.

    :param sep: separator string
    :param bin_type: the type of bin, if file format is stereo-seq file. `bins` or `cell_bins`.
    :param bin_size: the size of bin to merge. The parameter only takes effect
                     when the value of data.bin_type is 'bins'.
    :param is_sparse: the matrix is sparse matrix if is_sparse is True else np.ndarray
    :param ann_format: if input data format is Anndata, it should be True. Defalut False.
    :return:
    """
    if file_format == 'txt':
        return read_txt(file_path,sep=sep, bin_type=bin_type, bin_size=bin_size, is_sparse=is_sparse)
    elif file_format == 'h5ad':
        if not ann_format:
            return read_stereo(file_path)
        else:
            return read_ann_h5ad(file_path)
    else:
        pass

def read_10x(path):
    pass

def andata_to_stereo(andata: AnnData, use_raw=False,pos_sep='-'):
    """
    transform the Anndata object into StereoExpData object.
    :param andata: input Anndata object,
    :param use_raw: use andata.raw.X if True else andata.X. Default is False.
    :param pos_sep: separator string of andata.obs.index. Default is '-'
    :return: StereoExpData obj.
    """
    # data matrix,including X,raw,layer
    data = StereoExpData()
    data.exp_matrix = andata.raw.X if use_raw else andata.X
    data.position = np.array(list(andata.obs.index.str.split(pos_sep, expand=True)), dtype=np.int)
    #obs -> cell
    data.cells.cell_name = np.array(andata.obs_names)
    data.cells.n_genes_by_counts = andata.obs['n_genes_by_counts'] if 'n_genes_by_counts' in andata.obs.columns.tolist() else None
    data.cells.total_counts = andata.obs['total_counts'] if 'total_counts' in andata.obs.columns.tolist() else None
    data.cells.pct_counts_mt = andata.obs['pct_counts_mt'] if 'pct_counts_mt' in andata.obs.columns.tolist() else None
    ##var
    data.genes.gene_name = np.array(andata.var_names)
    data.genes.n_cells = andata.var['n_cells'] if 'n_cells' in andata.var.columns.tolist() else None
    data.genes.n_counts = andata.var['n_counts'] if 'n_counts' in andata.var.columns.tolist() else None
    return data


def check_file(path, prefix, suffix):
    filename = f"{path}/{prefix}{suffix}"
    if os.path.isfile(filename):
        return filename
    elif suffix in {"matrix.mtx", "barcodes.tsv"} and os.path.isfile(f"{filename}.gz"):
        return f'{filename}.gz'
    elif suffix == "genes.tsv" and os.path.isfile(f'{path}/{prefix}features.tsv.gz'):
        return f'{path}/{prefix}features.tsv.gz'
    else:
        # logger.error(f"{path} is not exist!")
        # raise FileExistsError(f"can not find {path}/{prefix}{suffix}(or with .gz)!")
        raise ValueError(f"can not find {filename}(or with .gz).")


# def read_10x_data(path, prefix="", gene_ex_only=True):
#     """
#     read 10x data
#
#     :param path: the dictionary of the input files
#     :param prefix: the prefix of the input files
#     :param gene_ex_only: return gene expression data only if is True
#     :return: anndata
#     """
#     # 1.   check file status.
#     # if not os.path.exists(path):
#     #    logger.error(f"{path} is not exist!")
#     #     raise FileExistsError(f"{path} is not exist!")
#     basic_fileset = {'barcodes.tsv', 'genes.tsv', 'matrix.mtx'}
#     genefile = (f"{path}/{prefix}genes.tsv")
#     featurefile = (f"{path}/{prefix}features.tsv.gz")
#     adata = read_10x_mtx(path, prefix)
#     if os.path.isfile(genefile) or not gene_ex_only:
#         return adata
#     else:
#         gex_rows = list(map(lambda x: x == 'Gene Expression', adata.var['feature_types']))
#         return adata[:, gex_rows].copy()


# def read_10x_mtx(path, prefix="", var_names='gene_symbols', make_unique=True):
#     mtxfile = check_file(path, prefix, "matrix.mtx")
#     genesfile = check_file(path, prefix, "genes.tsv")
#     barcodesfile = check_file(path, prefix, "barcodes.tsv")
#     adata = read_mtx(mtxfile).T  # transpose
#     genes = pd.read_csv(genesfile, header=None, sep='\t')
#     gene_id = genes[0].values
#     gene_symbol = genes[1].values
#     if var_names == 'gene_symbols':
#         var_names = genes[1].values
#         # if make_unique:
#         #    var_names = anndata.utils.make_index_unique(pd.Index(var_names))
#         adata.var_names = var_names
#         adata.var['gene_ids'] = genes[0].values
#     elif var_names == 'gene_ids':
#         adata.var_names = genes[0].values
#         adata.var['gene_symbols'] = genes[1].values
#     else:
#         raise ValueError("`var_names` needs to be 'gene_symbols' or 'gene_ids'")
#     if os.path.isfile(f"{path}/{prefix}features.tsv.gz"):
#         adata.var['feature_types'] = genes[2].values
#
#     adata.obs_names = pd.read_csv(barcodesfile, header=None)[0].values
#     return adata
