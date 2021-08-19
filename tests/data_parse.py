#!/usr/bin/env python3
# coding: utf-8
"""
@author:qiuping1
@file:data_parse.py
@time:2020/12/30
change log: 2021/01/12   增加数据qc相关代码，并调式
python ./data_parse.py --input_path ../data/01.LiverCancer/DP8400012941BR_E4/DP8400012941BR_E4.txt --out_dir ../data/E4/ --read_raw  --bin_size 200
python ./data_parse.py --input_path ../data/E4/raw_andata.bin200.h5ad --out_dir ../data/E4/ --run_filter --normalize  --bin_size 200 --max_gene_cnt 7000 --min_genes 200 --min_cells 3 --max_mt 15
"""

import pandas as pd
import scanpy as sc
import numpy as np
import argparse
from scipy import sparse
import sys


def read_raw_file_bak(inpath, step):
    """
    读取原始bin1的数据，返回andata对象
    :param inpath: 输入文件路径
    :param step: 合并的bin大小
    :param output: andata存储路径
    :return: andata
    """
    df = pd.read_csv(inpath, sep='\t')
    df.dropna(inplace=True)
    df.columns = list(df.columns[0:-1]) + ['UMICount']
    df['x1'] = (df['x'] / step).astype(np.int32)
    df['y1'] = (df['y'] / step).astype(np.int32)
    df['pos'] = df['x1'].astype(str) + "-" + df['y1'].astype(str)
    g = df.groupby(['geneID', 'pos'])['UMICount'].sum()
    g = g.to_frame().reset_index()
    # 每个gene至少在3个bin里面捕获到
    # g = g[g.groupby('geneID')['geneID'].transform('size') > 2]
    g = g.pivot(index='pos', columns='geneID', values='UMICount').fillna(0)
    # 每个bin至少捕获到50个gene
    # g = g.loc[:, g.sum() >= 50]
    adata = sc.AnnData(g)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    pos[:, 1] = pos[:, 1] * -1
    adata.obsm['spatial'] = pos
    return adata


def read_raw_file(inpath, step):
    df = pd.read_csv(inpath, sep='\t')
    df.dropna(inplace=True)
    df.columns = list(df.columns[0:-1]) + ['UMICount']
    df['x1'] = (df['x'] / step).astype(np.int32)
    df['y1'] = (df['y'] / step).astype(np.int32)
    df['pos'] = df['x1'].astype(str) + "-" + df['y1'].astype(str)
    bindf = df.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bindf.index)
    genes = set(x[1] for x in bindf.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bindf.index]
    cols = [genesdic[x[1]] for x in bindf.index]
    print(f'the martrix has {len(cells)} bins, and {len(genes)} genes.')
    expMtx = sparse.csr_matrix((bindf.values, (rows, cols))).toarray()
    print(f'the size of matrix is {sys.getsizeof(expMtx) / 1073741824} G.')
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = sc.AnnData(X=expMtx, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    pos[:, 1] = pos[:, 1] * -1
    adata.obsm['spatial'] = pos
    return adata


def cal_data_distribution(adata):
    """
    计算数据的分布，主要包含total count，n_gene_by_count, mt gene
    :param adata: 经过基础过滤后的andata
    :return:
    """
    sc.pp.calculate_qc_metrics(adata, inplace=True)  # 统计qc指标
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  + adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)  # 统计线粒体基因分布
    return adata


def data_filter(adata, max_gene_cnt, max_mt, min_genes=50, min_cells=3):
    """
    数据过滤
    :param adata:
    :param max_gene_cnt: 单个bin至多包含的gene种类数量
    :param max_mt: 单个bin至多包含的线粒体gene比例
    :param min_genes: 每个bin至少包含的基因数
    :param min_cells: 每个gene至少出现在的bin数量
    :return:
    """
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs.n_genes_by_counts < max_gene_cnt, :]
    adata = adata[adata.obs.pct_counts_mt < max_mt, :]
    return adata


def data_filter_with_cluster(adata, groups, method, action='save'):
    """
    :param adata: andata对象
    :param groups: 分组类别列表
    :param action: save|move 选择保留还是移除groups
    :return:
    """
    if action == 'save':
        new_adata = adata[adata.obs[method].isin(groups), :]
    else:
        new_adata = adata[~adata.obs[method].isin(groups), :]
    return new_adata


def bin1_filter_with_cluster(raw_file, bin_size, cluster_andata, cluster_method, groups, output):
    df = pd.read_csv(raw_file, sep='\t')
    df.dropna(inplace=True)
    df.columns = list(df.columns[0:-1]) + ['UMICount']
    df['x1'] = (df['x'] / bin_size).astype(np.int32)
    df['y1'] = (df['y'] / bin_size).astype(np.int32)
    df['pos'] = df['x1'].astype(str) + "-" + df['y1'].astype(str)
    df.set_index(['pos'], inplace=True)
    cluster = cluster_andata.obs[cluster_method][~cluster_andata.obs[cluster_method].isin(groups)].to_frame()
    result = cluster.join(df)[['geneID', 'x', 'y', 'UMICount']]
    result.to_csv(output, sep='\t', index=False)


def data_normalize(adata, scale=False):
    """
    数据标准化
    :param adata:
    :param scale:
    :return:
    """
    sc.pp.normalize_total(adata, target_sum=1e4)  # TPM
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata)
    if scale:
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
        sc.pp.scale(adata, max_value=10)
    return adata


def save_adata(output, adata):
    """
    保存数据
    :param output: 输出路径
    :param adata: andata对象
    :return:
    """
    sc.write(output, adata)


def read_h5ad_file(inpath):
    """
    读取andata的h5ad文件
    :param inpath: 输入路径
    :return: andata
    """
    return sc.read_h5ad(inpath)


if __name__ == '__main__':
    from datetime import datetime
    import os
    import matplotlib
    import warnings

    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='the output folder of result ')
    parser.add_argument('--input_path', required=True, help='the input file. the raw file (bin1) if --read_raw is set '
                                                            'else h5ad of andata')
    parser.add_argument('--bin_size', required=True, type=int, help='the size of bin')
    parser.add_argument('--min_genes', type=int, default=50, help='the min genes of a cell')
    parser.add_argument('--min_cells', type=int, default=3, help='the min cells of a gene')
    parser.add_argument('--max_gene_cnt', type=int, default=3000, help='the max gene counts when --run_filter is set')
    parser.add_argument('--max_mt', type=int, default=10, help='the max counts of mt gene when --run_filter is set')
    parser.add_argument('--read_raw', action='store_true', help='whether to read the raw file')
    parser.add_argument('--run_filter', action='store_true', help='Whether to run filter')
    parser.add_argument('--normalize', action='store_true', help='Whether to run normalize')
    parser.add_argument('--scale', action='store_true', help='Whether to run scale')
    opt = parser.parse_args()
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    if opt.read_raw:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} read raw file {opt.input_path} ...')
        andata = read_raw_file(opt.input_path, opt.bin_size)
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} cal data distribution ...')
        andata = cal_data_distribution(andata)
        save_adata(output=os.path.join(opt.out_dir, f'raw_andata.bin{opt.bin_size}.h5ad'),  adata=andata)
    if opt.run_filter:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} start to run filter ')
        print(f'max_gene_cnt: {opt.max_gene_cnt}; max_mt: {opt.max_mt}')
        andata = read_h5ad_file(opt.input_path)
        andata = data_filter(andata, max_gene_cnt=opt.max_gene_cnt, max_mt=opt.max_mt,
                             min_cells=opt.min_cells, min_genes=opt.min_genes)
        output = os.path.join(opt.out_dir, f'qc_andata.bin{opt.bin_size}.h5ad')
        if opt.normalize:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} start to run normalize ')
            andata = data_normalize(andata, scale=opt.scale)
            output = os.path.join(opt.out_dir, f'qc_with_normalize.bin{opt.bin_size}.h5ad')
            plot_figures.save_fig(output=os.path.join(opt.out_dir, f'normalize_scatter.bin{opt.bin_size}.jpg'))
        save_adata(output=output, adata=andata)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Done ...')
