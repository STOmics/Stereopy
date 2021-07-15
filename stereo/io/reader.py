#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:reader.py
@time:2021/03/05

change log:
    2021/03/05  add read_stereo_data function , by Ping Qiu.
"""
import pandas as pd
import os
from anndata import read_mtx
from ..core.stereo_exp_data import StereoExpData


def read_stereo(path, bin_type, bin_size):
    data = StereoExpData(file_path=path, file_format='txt', bin_type=bin_type, bin_size=bin_size)
    data.init()
    return data


def read_h5ad(path):
    data = StereoExpData(file_path=path, file_format='h5ad')
    data.init()
    return data


def read_10x(path):
    pass


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


def read_10x_data(path, prefix="", gene_ex_only=True):
    """
    read 10x data

    :param path: the dictionary of the input files
    :param prefix: the prefix of the input files
    :param gene_ex_only: return gene expression data only if is True
    :return: anndata
    """
    # 1.   check file status.
    # if not os.path.exists(path):
    #    logger.error(f"{path} is not exist!")
    #     raise FileExistsError(f"{path} is not exist!")
    basic_fileset = {'barcodes.tsv', 'genes.tsv', 'matrix.mtx'}
    genefile = (f"{path}/{prefix}genes.tsv")
    featurefile = (f"{path}/{prefix}features.tsv.gz")
    adata = read_10x_mtx(path, prefix)
    if os.path.isfile(genefile) or not gene_ex_only:
        return adata
    else:
        gex_rows = list(map(lambda x: x == 'Gene Expression', adata.var['feature_types']))
        return adata[:, gex_rows].copy()


def read_10x_mtx(path, prefix="", var_names='gene_symbols', make_unique=True):
    mtxfile = check_file(path, prefix, "matrix.mtx")
    genesfile = check_file(path, prefix, "genes.tsv")
    barcodesfile = check_file(path, prefix, "barcodes.tsv")
    adata = read_mtx(mtxfile).T  # transpose
    genes = pd.read_csv(genesfile, header=None, sep='\t')
    gene_id = genes[0].values
    gene_symbol = genes[1].values
    if var_names == 'gene_symbols':
        var_names = genes[1].values
        # if make_unique:
        #    var_names = anndata.utils.make_index_unique(pd.Index(var_names))
        adata.var_names = var_names
        adata.var['gene_ids'] = genes[0].values
    elif var_names == 'gene_ids':
        adata.var_names = genes[0].values
        adata.var['gene_symbols'] = genes[1].values
    else:
        raise ValueError("`var_names` needs to be 'gene_symbols' or 'gene_ids'")
    if os.path.isfile(f"{path}/{prefix}features.tsv.gz"):
        adata.var['feature_types'] = genes[2].values

    adata.obs_names = pd.read_csv(barcodesfile, header=None)[0].values
    return adata
