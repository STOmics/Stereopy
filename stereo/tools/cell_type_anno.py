#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:cell_type_anno.py
@time:2021/03/09
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool
import traceback
from ..log_manager import logger


class CellTypeAnno(object):
    def __init__(self, adata, ref_path=None, cores=1, keep_zeros=True, use_rf=True,
                 sample_rate=0.8, n_estimators=20, strategy='1'):
        self.data = adata
        self.ref_path = ref_path
        self.n_jobs = cores
        self.keep_zeros = keep_zeros
        self.use_rf = use_rf
        self.sample_rate = sample_rate
        self.n_estimators = n_estimators
        self.strategy = strategy

    def parse_ref_data(self):
        logger.info(f'loading ref data')
        ref_db = pd.read_csv(self.ref_path, index_col=0, header=0)
        ref_db = ref_db.fillna(0)
        # remove duplicate indices
        ref_db = ref_db[~ref_db.index.duplicated(keep='first')]
        logger.info('reference dataset shape: %s genes, %s samples' % ref_db.shape)
        return ref_db

    def random_choose_genes(df, sample_rate):
        sample_cnt = pd.Series(np.int32(df.sum(axis=0) * sample_rate), index=df.columns)
        gene_rate = df / df.sum(axis=0)
        sample_df = gene_rate.apply(lambda x: choose_gene(x, sample_cnt), axis=0)
        sample_df.fillna(0, inplace=True)
        return sample_df

    def choose_gene(x, num):
        gene_list = list(x.index)
        p = x.values
        res = np.random.choice(a=gene_list, size=num[x.name], p=p)
        res = np.unique(res, return_counts=True)
        res = pd.Series(data=res[1], index=res[0], name=x.name)
        return res


def annotation(refDB, testDB, keepZeros, method):
    """
    注释
    :param refDB: 参考数据dataframe
    :param testDB: 输入数据dataframe
    :param keepZeros: 是否保持参考数据的所有基因，建议为True
    :param method: 相关系数的计算方法：pearson 或 spearman
    :return:
    """
    # find common genes between test data and ref data
    testRows = set(testDB.index)
    refRows = set(refDB.index)
    if keepZeros:
        commonRows = list(refRows)
    else:
        commonRows = list(refRows.intersection(testRows))
    # only keep non-zero genes
    testDB = testDB.reindex(commonRows, fill_value=0.0)
    testrefDB = refDB.reindex(commonRows, fill_value=0.0)
    if method == 'pearson':
        corr_score = pearson_corr(testrefDB, testDB)
    else:
        corr_score = spearmanr_corr(testrefDB, testDB)
    corr_score = corr_score.fillna(0)
    return corr_score


def get_top_k_corr(score, map_path, output, k=10):
    """
    获取top 1的cell type
    :param score: 相关性矩阵
    :param map_path: sample与cell type的mapping文件
    :param output: 输出路径
    :param k: 相关分数从大到小取 top k
    :return:
    """
    m, n = score.shape
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(score, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    top_values = score[rows, topk_indices].reshape(-1)
    samples = score.columns[topk_indices.reshape(-1)]
    cell = score.index.repeat(k)
    map_df = pd.read_csv(map_path, index_col=0, header=0, sep=',')
    cell_type = map_df.loc[samples, 'cell type']
    df = pd.DataFrame({'cell': cell, 'cell type': cell_type, 'corr_score': top_values})
    group_df = df.groupby(['cell', 'cell type']).mean().reset_index() \
        .groupby('cell').apply(lambda x: x[x['corr_score'] == x['corr_score'].max()])
    group_df.to_csv(output, index=False)
    return group_df


def get_top_corr(score, map_df, output):
    """
    获取top 1的cell type
    :param score: 相关性矩阵
    :param map_path: sample与cell type的mapping文件
    :param output: 输出路径
    :return:
    """
    max_index = score.values.argmax(axis=1)
    max_value = score.values.max(axis=1)
    samples = score.columns[max_index]
    cell_type = map_df.loc[samples, 'cell type']
    cell = score.index
    df = pd.DataFrame({'cell': cell, 'cell type': cell_type, 'corr_score': max_value, 'corr_sample': samples})
    df.to_csv(output, index=False)


def split_dataframe(df, split_num):
    """
    按列分割dataframe
    :param df:
    :param split_num:
    :return:
    """
    datas = []
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} input data:  {df.shape[0]} genes, {df.shape[1]} cells.')
    if split_num > 1:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} split to {split_num} matrixs')
        step_size = int(df.shape[1]/split_num) + 1
        for i in range(split_num):
            start = i * step_size
            end = start + step_size if start + step_size < df.shape[1] else df.shape[1]
            datas.append(df.iloc[:, start: end])
    else:
        datas.append(df)
    return datas


def concat_top_corr_files(files, output_dir, prefix):
    """
    连接多个子进程的top相关性结果
    :param output_dir:
    :param method:
    :return:
    """
    df = pd.read_csv(files[0])
    for f in files[1:]:
        df1 = pd.read_csv(f)
        df = df.append(df1)
    df.to_csv(os.path.join(output_dir, f'{prefix}_top_annotation.csv'), index=False)


def merge_subsample_result(input_dir, prefix, sample_num, output_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if prefix in f]
    df = pd.read_csv(files[0])
    for f in files[1:]:
        df1 = pd.read_csv(f)
        df = df.append(df1)
    type_cnt = df.groupby(['cell', 'cell type']).count()[['corr_score']]
    type_cnt['type_rate'] = type_cnt / sample_num
    type_cnt.columns = ['type_cnt', 'type_rate']
    type_cnt.reset_index(inplace=True)
    score_mean = df.groupby(['cell', 'cell type']).mean()[['corr_score']]
    score_mean.columns = ['score_mean']
    score_mean.reset_index(inplace=True)
    df = score_mean.merge(type_cnt, on=['cell', 'cell type'])
    df.to_csv(os.path.join(output_dir, 'all_annotation.csv'), index=False)
    df = df[(df.groupby('cell')['type_cnt'].transform('max') == df['type_cnt']) & (
                df.groupby('cell')['score_mean'].transform('max') == df['score_mean'])]
    df.to_csv(os.path.join(output_dir, 'top_annotation.csv'), index=False)


def merge_subsample_result_filter(input_dir, prefix, output_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if prefix in f]
    df = pd.read_csv(files[0])
    for f in files[1:]:
        df_1 = pd.read_csv(f)
        df = df.append(df_1)
    score_mean = df.groupby(['cell', 'cell type']).mean()[['corr_score']]
    score_mean.columns = ['score_mean']
    score_mean.reset_index(inplace=True)
    tmp = score_mean.merge(df, on=['cell', 'cell type'])
    tmp = tmp[tmp['score_mean'] <= tmp['corr_score']]
    type_cnt = tmp.groupby(['cell', 'cell type']).count()[['score_mean']].reset_index()
    type_sum = tmp.groupby(['cell']).count()[['corr_score']].reset_index()
    score_mean = tmp.groupby(['cell', 'cell type']).mean()[['corr_score']]
    score_mean.columns = ['score_mean']
    score_mean.reset_index(inplace=True)
    type_rate = type_cnt.merge(type_sum, on=['cell'])
    type_rate['type_rate'] = type_rate['score_mean'] / type_rate['corr_score']
    type_rate.columns = ['cell', 'cell type', 'type_cnt', 'type_cnt_sum', 'type_rate']
    df = score_mean.merge(type_rate, on=['cell', 'cell type'])
    df.to_csv(os.path.join(output_dir, 'all_annotation.csv'), index=False)
    df = df[df.groupby('cell')['type_cnt'].transform('max') == df['type_cnt']]
    df = df[df.groupby('cell')['score_mean'].transform('max') == df['score_mean']]
    df.to_csv(os.path.join(output_dir, 'top_annotation.csv'), index=False)





def sub_process_run(sub_df, ref_df, map_df, keep_zeros, method, output, sub_index, sample_rate):
    """
    子进程注释函数
    :param ref_data:
    :param input_data:
    :param keep_zeros:
    :param method:
    :param sample_map:
    :param output:
    :param sub_index:
    :return:
    """
    sub_df = random_choose_genes(sub_df, sample_rate)
    nor_x = sub_df.values * 10000 / sub_df.values.sum(axis=0)[np.newaxis, :]
    nor_x = np.log1p(nor_x, out=nor_x)
    sub_df = pd.DataFrame(nor_x, columns=sub_df.columns, index=sub_df.index)
    corr_df = annotation(ref_df, sub_df, keep_zeros, method)
    all_out = os.path.join(output, f'{sub_index}.all_{method}_corr.csv')
    top_out = os.path.join(output, f'{sub_index}.top_{method}_corr.csv')
    corr_df.to_csv(all_out)
    get_top_corr(corr_df, map_df, top_out)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} subsample {sub_index} DONE!')


def sub_print_error(value):
    """
    子进程打印错误信息
    :param value:
    :return:
    """
    print("error: ", value)
    print("========")
    print(traceback.format_exc())
    raise


def main(test_path, refDir, keepZeros, output, method, process, sample_rate, sample_num, split_num, top_strategy):
    """
    主函数入口
    :param test_path: 输入文件路径
    :param refDir: 注释参考数据文件夹路径
    :param keepZeros: 是否使用所有参考数据的gene
    :param output: 输出文件夹
    :param method: 相关性系数计算方法
    :param process: 多进程数量
    :return:
    """
    ref_sample = os.path.join(refDir, '9606_symbol.csv')
    sample_map = os.path.join(refDir, '9606_map.csv')
    ref_df = parse_ref_data(ref_sample)
    map_df = pd.read_csv(sample_map, index_col=0, header=0, sep=',')
    andata = data_parse.read_h5ad_file(test_path)
    df = andata.to_df().transpose()
    cells = df.shape[1]
    split_num = int(cells / 2000) if split_num < 0 else split_num
    datas = split_dataframe(df, split_num=split_num)
    pool = Pool(process)
    tmp_output = os.path.join(output, 'tmp')
    if not os.path.exists(tmp_output):
        os.makedirs(tmp_output)
    for i in range(sample_num):
        for j in range(len(datas)):
            sub_index = f'subsample_{i}_{j}'
            pool.apply_async(sub_process_run, (datas[j], ref_df, map_df, keepZeros, method, tmp_output,
                                               sub_index, sample_rate), error_callback=sub_print_error)
    pool.close()
    pool.join()
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  start to merge top result ...')
    for i in range(sample_num):
        files = [os.path.join(tmp_output, f'subsample_{i}_{j}.top_{method}_corr.csv') for j in range(split_num)]
        index = f'subsample_{i}'
        concat_top_corr_files(files, tmp_output, index)
    if top_strategy == 1:
        merge_subsample_result(tmp_output, 'top_annotation.csv', sample_num, output)
    else:
        merge_subsample_result_filter(tmp_output, 'top_annotation.csv', output)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} DONE!')
