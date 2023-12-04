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
import pickle
from copy import deepcopy
from os import environ

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import (
    csr_matrix,
    issparse
)

from stereo.core.stereo_exp_data import StereoExpData
from stereo.io import h5ad
from stereo.log_manager import logger

environ['HDF5_USE_FILE_LOCKING'] = "FALSE"


def write_h5ad(
        data: StereoExpData,
        use_raw: bool = True,
        use_result: bool = True,
        key_record: dict = None,
        output: str = None,
        split_batches: bool = True):
    """
    Write the StereoExpData into a H5ad file.

    Parameters
    ---------------------
    data
        the input StereoExpData object.
    use_raw
        whether to save raw data.
    use_result
        whether to save `result` and `res_key`.
    key_record
        a dict includes selective `res_key` with the precondition that `use_result`
        is `True`, if None, it will save the `result` and `res_key` of
        `data.tl.key_record`,otherwise it will save the result and res_key of the dict.
    output
        the path to output file.
    split_batches
        Whether to save each batch to a single file if it is a merged data, default to True.
    Returns
    -------------------
    None
    """
    if data.merged and split_batches:
        from os import path
        from ..utils.data_helper import split
        data_list = split(data)
        batch = np.unique(data.cells.batch)
        if output is not None:
            name, ext = path.splitext(output)
        for bno, d in zip(batch, data_list):
            if output is not None:
                boutput = f"{name}-{d.sn}{ext}"
            else:
                boutput = None
            write_h5ad(d, use_raw=use_raw, use_result=use_result, key_record=key_record, output=boutput,
                       split_batches=False)
        return

    if output is not None:
        data.output = output
    else:
        if data.output is None:
            raise Exception("The output path must be set before writing.")

    with h5py.File(data.output, mode='w') as f:
        _write_one_h5ad(f, data, use_raw=use_raw, use_result=use_result, key_record=key_record)


def _write_one_h5ad(f, data: StereoExpData, use_raw=False, use_result=True, key_record=None):
    if data.attr is not None:
        for key, value in data.attr.items():
            f.attrs[key] = value
    if data.sn is not None:
        sn_list = []
        if isinstance(data.sn, str):
            sn_list = [['-1', data.sn]]
        else:
            for bno, sn in data.sn.items():
                if sn is None:
                    sn_list = []
                    break
                sn_list.append([bno, sn])
        if len(sn_list) > 0:
            sn_data = pd.DataFrame(sn_list, columns=['batch', 'sn'])
            h5ad.write(sn_data, f, 'sn', save_as_matrix=True)
    h5ad.write(data.genes, f, 'genes')
    h5ad.write(data.cells, f, 'cells')
    if data.position_z is None:
        position = data.position
    else:
        position = np.concatenate([data.position, data.position_z], axis=1)
    h5ad.write(position, f, 'position')
    if issparse(data.exp_matrix):
        sp_format = 'csr' if isinstance(data.exp_matrix, csr_matrix) else 'csc'
        h5ad.write(data.exp_matrix, f, 'exp_matrix', sp_format)
    else:
        h5ad.write(data.exp_matrix, f, 'exp_matrix')
    h5ad.write(data.bin_type, f, 'bin_type')
    h5ad.write(data.bin_size, f, 'bin_size')
    h5ad.write(data.merged, f, 'merged')

    if use_raw is True:
        same_genes = np.array_equal(data.tl.raw.gene_names, data.gene_names)
        same_cells = np.array_equal(data.tl.raw.cell_names, data.cell_names)
        if not same_genes:
            # if raw genes differ from genes
            h5ad.write(data.tl.raw.genes, f, 'genes@raw')
        if not same_cells:
            # if raw cells differ from cells
            h5ad.write(data.tl.raw.cells, f, 'cells@raw')
        if not (same_genes | same_cells):
            # if either raw genes or raw cells are different
            if data.tl.raw.position_z is None:
                position = data.tl.raw.position
            else:
                position = np.concatenate([data.tl.raw.position, data.tl.raw.position_z], axis=1)
            h5ad.write(position, f, 'position@raw')
        # save raw exp_matrix
        if issparse(data.tl.raw.exp_matrix):
            sp_format = 'csr' if isinstance(data.tl.raw.exp_matrix, csr_matrix) else 'csc'
            h5ad.write(data.tl.raw.exp_matrix, f, 'exp_matrix@raw', sp_format)
        else:
            h5ad.write(data.tl.raw.exp_matrix, f, 'exp_matrix@raw')

    if use_result is True:
        _write_one_h5ad_result(data, f, key_record)


def _write_one_h5ad_result(data, f, key_record):
    # write key_record
    mykey_record = deepcopy(data.tl.key_record) if key_record is None else deepcopy(key_record)
    mykey_record_keys = list(mykey_record.keys())
    # supported_keys = ['hvg', 'pca', 'neighbors', 'umap', 'cluster', 'marker_genes', 'cell_cell_communication '] # 'sct', 'spatial_hotspot' # noqa
    supported_keys = data.tl.key_record.keys()
    for analysis_key in mykey_record_keys:
        if analysis_key not in supported_keys:
            mykey_record.pop(analysis_key)
            logger.info(f'key_name:{analysis_key} is not recongnized, '
                        f'try to select the name in {supported_keys} as your key_name.')
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
                hvg_df: pd.DataFrame = deepcopy(data.tl.result[res_key])
                if 'mean_bin' in hvg_df.columns:
                    hvg_df.mean_bin = [str(interval) for interval in data.tl.result[res_key].mean_bin]
                h5ad.write(hvg_df, f, f'{res_key}@hvg')  # -> dataframe
            if analysis_key in ['pca', 'umap', 'totalVI', 'spatial_alignment_integration']:
                h5ad.write(data.tl.result[res_key].values, f, f'{res_key}@{analysis_key}')  # -> array
            if analysis_key == 'neighbors':
                for neighbor_key, value in data.tl.result[res_key].items():
                    if value is None:
                        continue
                    if issparse(value):
                        sp_format = 'csr' if isinstance(value, csr_matrix) else 'csc'
                        h5ad.write(value, f, f'{neighbor_key}@{res_key}@neighbors', sp_format)  # -> csr_matrix
                    else:
                        h5ad.write(value, f, f'{neighbor_key}@{res_key}@neighbors')  # -> Neighbors
            if analysis_key == 'cluster':
                h5ad.write(data.tl.result[res_key], f, f'{res_key}@cluster')  # -> dataframe
            if analysis_key == 'gene_exp_cluster':
                h5ad.write(data.tl.result[res_key], f, f'{res_key}@gene_exp_cluster', save_as_matrix=True)
            if analysis_key == 'marker_genes':
                clusters = list(data.tl.result[res_key].keys())
                h5ad.write(clusters, f, f'clusters_record@{res_key}@marker_genes')  # -> list
                for cluster, df in data.tl.result[res_key].items():
                    if cluster != 'parameters':
                        h5ad.write(df, f, f'{cluster}@{res_key}@marker_genes')  # -> dataframe
                    else:
                        name, value = [], []
                        for pname, pvalue in df.items():
                            name.append(pname)
                            value.append(pvalue)
                        parameters_df = pd.DataFrame({
                            'name': name,
                            'value': value
                        })
                        h5ad.write(parameters_df, f, f'{cluster}@{res_key}@marker_genes')  # -> dataframe
            if analysis_key == 'sct':
                h5ad.write(
                    csr_matrix(data.tl.result[res_key][0]['counts']), f, f'exp_matrix@{res_key}@sct_counts', 'csr'
                )
                h5ad.write(
                    csr_matrix(data.tl.result[res_key][0]['data']), f, f'exp_matrix@{res_key}@sct_data', 'csr'
                )
                h5ad.write(
                    csr_matrix(data.tl.result[res_key][0]['scale.data']), f, f'exp_matrix@{res_key}@sct_scale',
                    'csr'
                )
                h5ad.write(list(data.tl.result[res_key][1]['umi_genes']), f, f'genes@{res_key}@sct')
                h5ad.write(list(data.tl.result[res_key][1]['umi_cells']), f, f'cells@{res_key}@sct')
                h5ad.write(list(data.tl.result[res_key][1]['top_features']), f, f'genes@{res_key}@sct_top_features')
                h5ad.write(list(data.tl.result[res_key][0]['scale.data'].index), f,
                           f'genes@{res_key}@sct_scale_genename')
                # TODO ignored other result of the sct
            if analysis_key == 'spatial_hotspot':
                # Hotspot object
                pass
            if analysis_key == 'cell_cell_communication':
                for key, item in data.tl.result[res_key].items():
                    if key != 'parameters':
                        h5ad.write(item, f, f'{res_key}@{key}@cell_cell_communication',
                                   save_as_matrix=False)  # -> dataframe
                    else:
                        name, value = [], []
                        for pname, pvalue in item.items():
                            name.append(pname)
                            value.append(pvalue)
                        parameters_df = pd.DataFrame({
                            'name': name,
                            'value': value
                        })
                        h5ad.write(parameters_df, f, f'{res_key}@{key}@cell_cell_communication',
                                   save_as_matrix=False)  # -> dataframe
            if analysis_key == 'regulatory_network_inference':
                for key, item in data.tl.result[res_key].items():
                    if key == 'regulons':
                        h5ad.write(str(item), f, f'{res_key}@{key}@regulatory_network_inference')  # -> str
                    else:
                        h5ad.write(item, f, f'{res_key}@{key}@regulatory_network_inference',
                                   save_as_matrix=False)  # -> dataframe
            if analysis_key == 'co_occurrence':
                for key, item in data.tl.result[res_key].items():
                    h5ad.write(item, f, f'{res_key}@{key}@co_occurrence', save_as_matrix=True)


def write_h5ms(ms_data, output: str):
    """
    Save an object of MSData into a h5 file whose suffix is 'h5ms'.

    :param ms_data: The object of MSData to be saved.
    :param output: The path of file into which MSData is saved.
    """
    with h5py.File(output, mode='w') as f:
        f.create_group('sample')
        for idx, data in enumerate(ms_data._data_list):
            f['sample'].create_group(f'sample_{idx}')
            _write_one_h5ad(f['sample'][f'sample_{idx}'], data)
        if ms_data._merged_data:
            f.create_group('sample_merged')
            _write_one_h5ad(f['sample_merged'], ms_data._merged_data)
        h5ad.write_list(f, 'names', ms_data.names)
        h5ad.write_dataframe(f, 'obs', ms_data.obs)
        h5ad.write_dataframe(f, 'var', ms_data.var)
        h5ad.write(ms_data._var_type, f, 'var_type')
        h5ad.write(ms_data.relationship, f, 'relationship')
        if ms_data.tl.result:
            mss_f = f.create_group('mss')
            for key in ms_data.tl.result.keys():
                data = StereoExpData()
                data.tl.result = ms_data.tl.result[key]
                data.tl.key_record = ms_data.tl.key_record[key]
                mss_f.create_group(key)
                _write_one_h5ad_result(data, mss_f[key], data.tl.key_record)


def write_mid_gef(data: StereoExpData, output: str):
    """
    Write the StereoExpData object into a GEF (.h5) file.

    Parameters
    ---------------------
    data
        the input StereoExpData object.
    output
        the path to output file.

    Returns
    ---------------------
    None
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

        # count
        g_len = len(final_gene)
        last_offset = 0 if g_len == 0 else final_gene[g_len - 1][1]
        last_count = 0 if g_len == 0 else final_gene[g_len - 1][2]
        g_name = data.gene_names[i]
        offset = last_offset + last_count
        count = c_idx.shape[0]
        final_gene.append((g_name, offset, count))
    final_exp_np = rfn.unstructured_to_structured(
        np.array(final_exp, dtype=int), np.dtype([('x', np.uint32), ('y', np.uint32), ('count', np.uint16)]))
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


def update_gef(data: StereoExpData, gef_file: str, cluster_res_key: str):
    """
    Add cluster result into GEF (.h5) file and update the GEF file directly.

    Parameters
    -----------------
    data
        the input StereoExpData object.
    gef_file
        the path of the GEF file to add cluster result to.
    cluster_res_key
        the key to get cluster result from `data.tl.result`.

    Returns
    --------------
    None
    """
    cluster = {}
    cluster_idx = {}
    if cluster_res_key not in data.tl.result:
        raise Exception(f'{cluster_res_key} is not in the result, please check and run the func of cluster.')
    clu_result = data.tl.result[cluster_res_key]
    bins = clu_result['bins'].to_numpy().astype('U')
    groups = clu_result['group'].astype('category')
    groups_idx = groups.cat.codes.to_numpy()
    groups_code = groups.cat.categories.to_numpy()
    groups = groups.to_numpy()
    is_numeric = True
    for bin, c, cidx in zip(bins, groups, groups_idx):
        if not isinstance(c, str):
            cluster[bin] = c
            cluster_idx[bin] = cidx
        elif c.isdigit():
            cluster[bin] = int(c)
            cluster_idx[bin] = cidx
        else:
            cluster[bin] = c
            cluster_idx[bin] = cidx
            is_numeric = False

    with h5py.File(gef_file, 'r+') as h5f:
        cell_names = np.bitwise_or(np.left_shift(h5f['cellBin']['cell']['x'].astype('uint64'), 32),
                                   h5f['cellBin']['cell']['y'].astype('uint64')).astype('U')
        celltid = np.zeros(h5f['cellBin']['cell'].shape, dtype='uint16')

        for n, cell_name in enumerate(cell_names):
            if cell_name in cluster:
                if is_numeric:
                    celltid[n] = cluster[cell_name]
                else:
                    celltid[n] = cluster_idx[cell_name]

        if is_numeric:
            h5f['cellBin']['cell']['clusterID'] = celltid
        else:
            h5f['cellBin']['cell']['cellTypeID'] = celltid
            del h5f['cellBin']['cellTypeList']
            h5f['cellBin']['cellTypeList'] = groups_code
