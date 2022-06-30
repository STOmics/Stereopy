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


def write_h5ad(data, use_raw=True, use_result=True, key_record=None, output=None):
    """
    write the StereoExpData into h5ad file.
    :param data: the StereoExpData object.
    :param use_raw: bool, whether to save raw data
    :param use_result: bool, whether to save result and res_key
    :param key_record: Dict. if None, it will save the result and res_key of data.tl.key_record.
    :param: output: the output path. StereoExpData's output will be reset if the output is not None.
    otherwise, it will save the result and res_key of this dict.

    :return:
    """
    if output is not None:
        data.output = output
    else:
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

        if use_raw is True:
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

        if use_result is True:
            # write key_record
            mykey_record = deepcopy(data.tl.key_record) if key_record is None else deepcopy(key_record)
            supported_keys = ['hvg', 'pca', 'neighbors', 'umap', 'cluster', 'marker_genes'] # 'sct', 'spatial_hotspot'
            for analysis_key in mykey_record.keys():
                if analysis_key not in supported_keys:
                    mykey_record.pop(analysis_key)
                    logger.info(f'key_name:{analysis_key} is not recongnized, try to select the name in {supported_keys} as your key_name.')
            h5ad.write_key_record(f, 'key_record', mykey_record)

            for analysis_key, res_keys in mykey_record.items():
                for res_key in res_keys:
                    # check
                    if res_key not in data.tl.result:
                        raise Exception(
                            f'{res_key} is not in the result, please check and run the coordinated func.')
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


def write_mid_gef(data, output):
    """
    write the StereoExpData into a gef file.

    :param data: StereoExpData object
    :param output: gef file.
    :return:
    """
    logger.info("The output standard gef file only contains one expression matrix with mid count."
                "Please make sure the expression matrix of StereoExpData object is mid count without normaliztion.")
    import numpy.lib.recfunctions as rfn
    final_exp = []  # [(x_1,y_1,umi_1),(x_2,y_2,umi_2)]
    final_gene = []  # [(A,offset,count)]
    exp_np = data.exp_matrix.toarray()

    for i in range(exp_np.shape[1]):
        gene_exp = exp_np[:, i]
        c_idx = np.nonzero(gene_exp)[0]  # idx for all cells
        zipped = np.concatenate((data.position[c_idx], gene_exp[c_idx].reshape(c_idx.shape[0], 1)), axis=1)
        for k in zipped:
            final_exp.append(k)

        ## count
        g_len = len(final_gene)
        last_offset = 0 if g_len == 0 else final_gene[g_len - 1][1]
        last_count = 0 if g_len == 0 else final_gene[g_len - 1][2]
        g_name = data.gene_names[i]
        offset = last_offset + last_count
        count = c_idx.shape[0]
        final_gene.append((g_name, offset, count))
    final_exp_np = rfn.unstructured_to_structured(np.array(final_exp, dtype=int),
                                                  np.dtype([('x', np.uint32), ('y', np.uint32), ('count', np.uint16)]))
    genetyp = np.dtype({'names': ['gene', 'offset', 'count'], 'formats': ['S32', np.uint32, np.uint32]})
    final_gene_np = np.array(final_gene, dtype=genetyp)
    h5f = h5py.File(output, "w")
    geneExp = h5f.create_group("geneExp")
    binsz = "bin" + str(data.bin_size)
    bing = geneExp.create_group(binsz)
    geneExp[binsz]["expression"] = final_exp_np  # np.arry([(10,20,2), (20,40,3)], dtype=exptype)
    geneExp[binsz]["gene"] = final_gene_np  # np.arry([("gene1",0,21), ("gene2",21,3)], dtype=genetype)
    if data.attr is not None:
        for key, value in data.attr.items():
            bing["expression"].attrs.create(key, value)
    h5f.attrs.create("version", 2)
    h5f.attrs.create("omics", 'Transcriptomics')
    h5f.close()


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