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
import numpy as np
from scipy import sparse
import os
from ..log_manager import logger
import sys
from anndata import AnnData


def read_stereo_data(path, sep='\t', bin_size=100, is_sparse=True):
    """
    transform stereo data into Anndata
    :param path: input file path
    :param sep: separator string
    :param bin_size: size of bin to merge
    :param is_sparse: the matrix is sparse matrix if is_sparse is True else np.ndarray
    :return:
    """
    if not os.path.exists(path):
        logger.error(f"{path} is not exist!")
        raise FileExistsError(f"{path} is not exist!")
    df = pd.read_csv(path, sep=sep)
    df.dropna(inplace=True)
    if "MIDCounts" in df.columns:
        df.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    df.columns = list(df.columns[0:-1]) + ['UMICount']
    df['x1'] = (df['x'] / bin_size).astype(np.int32)
    df['y1'] = (df['y'] / bin_size).astype(np.int32)
    df['pos'] = df['x1'].astype(str) + "-" + df['y1'].astype(str)
    bin_data = df.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bin_data.index)
    genes = set(x[1] for x in bin_data.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]
    logger.info(f'the martrix has {len(cells)} bins, and {len(genes)} genes.')
    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
        sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
    logger.info(f'the size of matrix is {sys.getsizeof(exp_matrix) / 1073741824} G.')
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    # pos[:, 1] = pos[:, 1] * -1
    adata.obsm['spatial'] = pos
    return adata


def read_h5ad(path: str) -> AnnData:
    pass
