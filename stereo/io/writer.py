#!/usr/bin/env python3
# coding: utf-8
"""
@file: writer.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Nils Mechtel

change log:
    2021/07/05  create file.
    2022/02/09  save raw data and result
"""
from stereo.core.stereo_exp_data import StereoExpData
from stereo.log_manager import logger
from scipy.sparse import csr_matrix, issparse
import h5py
from stereo.io import h5ad
import pickle
import numpy as np
from copy import deepcopy


def write_h5ad(data, result=True, raw=True):
    """
    write the StereoExpData into h5ad file.
    :param data: the StereoExpData object.
    :param result: whether to save result and res_key
    :param raw: whether to save raw data
    :return:
    """
    if data.output is None:
        logger.error("The output path must be set before writing.")
    with h5py.File(data.output, mode='w') as f:
        h5ad.write(data.genes, f, 'genes')
        h5ad.write(data.cells, f, 'cells')
        h5ad.write(data.position, f, 'position')
        if issparse(data.exp_matrix):
            sp_format = 'csr' if isinstance(data.exp_matrix, csr_matrix) else 'csc'
            h5ad.write(data.exp_matrix, f, 'exp_matrix', sp_format)
        else:
            h5ad.write(data.exp_matrix, f, 'exp_matrix')
        h5ad.write(data.bin_type, f, 'bin_type')

        if raw is True:
            same_genes = np.array_equal(data.tl.raw.gene_names, data.gene_names)
            same_cells = np.array_equal(data.tl.raw.gene_names, data.gene_names)
            if not same_genes:
                # if raw genes differ from genes
                h5ad.write(data.tl.raw.genes, f, 'genes@raw')
            if not same_cells:
                # if raw cells differ from cells
                h5ad.write(data.tl.raw.cells, f, 'cells@raw')
            if not (same_genes | same_cells):
                # if either raw genes or raw cells are different
                h5ad.write(data.tl.raw.position, f, 'position@raw')
            # save raw exp_matrix
            if issparse(data.tl.raw.exp_matrix):
                sp_format = 'csr' if isinstance(data.tl.raw.exp_matrix, csr_matrix) else 'csc'
                h5ad.write(data.tl.raw.exp_matrix, f, 'exp_matrix@raw', sp_format)
            else:
                h5ad.write(data.tl.raw.exp_matrix, f, 'exp_matrix@raw')

        if result is True:
            # write key_record
            key_record = deepcopy(data.tl.key_record)
            supported_keys = ['hvg', 'pca', 'neighbors', 'umap', 'cluster', 'marker_genes'] # 'sct', 'spatial_hotspot'
            for analysis_key in data.tl.key_record.keys():
                if analysis_key not in supported_keys:
                    key_record.pop(analysis_key)
            h5ad.write_key_record(f, 'key_record', key_record)

            for analysis_key, res_keys in key_record.items():
                for res_key in res_keys:
                    # write result[res_key]
                    if analysis_key == 'hvg':
                        # interval to str
                        hvg_df = deepcopy(data.tl.result[res_key])
                        hvg_df.mean_bin = [str(interval) for interval in data.tl.result[res_key].mean_bin]
                        h5ad.write(hvg_df, f, f'{res_key}@hvg')  # -> dataframe
                    if analysis_key in ['pca', 'umap']:
                        h5ad.write(data.tl.result[res_key].values, f, f'{res_key}@{analysis_key}')  # -> array
                    if analysis_key == 'neighbors':
                        for neighbor_key, value in data.tl.result[res_key].items():
                            if issparse(value):
                                sp_format = 'csr' if isinstance(value, csr_matrix) else 'csc'
                                h5ad.write(value, f, f'{neighbor_key}@{res_key}@neighbors', sp_format)  # -> csr_matrix
                            else:
                                h5ad.write(value, f, f'{neighbor_key}@{res_key}@neighbors')  # -> Neighbors
                    if analysis_key == 'cluster':
                        h5ad.write(data.tl.result[res_key], f, f'{res_key}@cluster')  # -> dataframe
                    if analysis_key == 'marker_genes':
                        clusters = list(data.tl.result[res_key].keys())
                        h5ad.write(clusters, f, f'clusters_record@{res_key}@marker_genes')  # -> list
                        for cluster, df in data.tl.result[res_key].items():
                            h5ad.write(df, f, f'{cluster}@{res_key}@marker_genes')  # -> dataframe
                    if analysis_key == 'sct':
                        # tuple: (StereoExpData, dict-17 keys with different type)

                        # st_exp_data = data.tl.result[res_key][0]
                        # sct_dict = data.tl.result[res_key][1]
                        # if not np.array_equal(data.exp_matrix, st_exp_data.exp_matrix):
                        #     h5ad.write(st_exp_data.genes, f, f'genes@{res_key}@sct')
                        #     h5ad.write(st_exp_data.cells, f, f'cells@{res_key}@sct')
                        #     h5ad.write(st_exp_data.position, f, f'position@{res_key}@sct')
                        #     if issparse(st_exp_data.exp_matrix):
                        #         sp_format = 'csr' if isinstance(st_exp_data.exp_matrix, csr_matrix) else 'csc'
                        #         h5ad.write(st_exp_data.exp_matrix, f, f'exp_matrix@{res_key}@sct', sp_format)
                        #     else:
                        #         h5ad.write(st_exp_data.exp_matrix, f, f'exp_matrix@{res_key}@sct')
                        # h5ad.write_sct(f, f'sct_dict@{res_key}@sct', sct_dict)
                        pass
                    if analysis_key == 'spatial_hotspot':
                        # Hotspot object
                        pass


def write(data, output=None, output_type='h5ad', *args, **kwargs):
    """
    write the data as a h5ad file.

    :param: data: the StereoExpData object.
    :param: output: the output path. StereoExpData's output will be reset if the output is not None.
    :param: output_type: the output type. StereoExpData's output will be written in output_type.
    Default setting is h5ad.
    :return:
    """
    if not isinstance(data, StereoExpData):
        raise TypeError
    if output is not None:
        data.output = output
        if output_type == 'h5ad':
            write_h5ad(data, *args, **kwargs)


def save_pkl(obj, output):
    with open(output, "wb") as f:
        pickle.dump(obj, f)
    f.close()


def update_gef(data, gef_file, cluster_res_key):
    """
    add cluster result into gef file and update the gef file directly.

    :param data: SetreoExpData.
    :param gef_file: add cluster result into gef file.
    :param cluster_res_key: the key of cluster to get the result for group info.
    :return:
    """
    cluster = {}
    if cluster_res_key not in data.tl.result:
        raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
    clu_result = data.tl.result[cluster_res_key]
    for i,v in clu_result.iterrows():
        cluster[v['bins']] = int(v['group'])+1

    h5f = h5py.File(gef_file, 'r+')
    cell_names = np.bitwise_or(np.left_shift(h5f['cellBin']['cell']['x'].astype('uint64'), 32), h5f['cellBin']['cell']['y'])
    celltid = np.zeros(h5f['cellBin']['cell'].shape, dtype='uint16')
    n = 0
    for cell_name in cell_names:
        if cell_name in cluster:
            celltid[n] = cluster[cell_name]
        n += 1

    # h5f['cellBin']['cell']['cellTypeID'] = celltid
    h5f['cellBin']['cell']['clusterID'] = celltid