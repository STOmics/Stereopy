"""
@file: reader.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Nils Mechtel

change log:
    2021/03/05  add read_stereo_data function , by Ping Qiu.
    2021/08/12  move read_txt functions from StereoExpData here. Add read_ann_h5ad,
    andata_to_stereo function by Yiran Wu.
    2021/08/20
    2022/02/09  read raw data and result
"""
from copy import deepcopy
from typing import Optional, Union, List
import re

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from typing_extensions import Literal

from stereo.core.cell import Cell
from stereo.core.constants import CHIP_RESOLUTION
from stereo.core.gene import Gene
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.core.stereo_exp_data import StereoExpData
from stereo.core.result import _BaseResult
from stereo.io import h5ad
from stereo.io.utils import(
    integrate_matrix_by_genes,
    transform_marker_genes_to_anndata,
    get_gem_comments
)
from stereo.log_manager import logger
from stereo.utils.read_write_utils import ReadWriteUtils


@ReadWriteUtils.check_file_exists
def read_gem(
        file_path: str,
        sep: str = '\t',
        bin_type: str = "bins",
        bin_size: int = 100,
        is_sparse: bool = True,
        center_coordinates: bool = False,
        gene_name_index: bool = False
):
    """
    Read the Stereo-seq GEM file, and generate the StereoExpData object.

    Parameters
    -------------
    file_path
        the path to input file.
    sep
        the separator string.
    bin_type
        the bin type includes `'bins'` or `'cell_bins'`, default to `'bins'`.
    bin_size
        the size of bin to merge, when `bin_type` is set to `'bins'`.
    is_sparse
        the expression matrix is sparse matrix, if `True`, otherwise `np.ndarray`.
    center_coordinates
        if set it to True, the coordinate of each bin will be the center of the bin,
        otherwise, the coordinate of each bin will be the left-top corner of the bin.
    gene_name_index
        In a v0.1 gem file, the column geneID is the gene name actually, but in a v0.2,
        geneID just a ID for genes and there is an additional column called geneName where is the gene name,
        When the version of gem file is v0.2, set `gene_name_index` to True to set column geneName as index, otherwise,
        set column geneID, if a v0.1 gem file, `gene_name_index` will be ignored and column geneID is set as index.

    Returns
    -------------
    An object of StereoExpData.
    """
    data = StereoExpData(file_path=file_path, file_format='gem', bin_type=bin_type, bin_size=bin_size)
    comments_lines, _ = get_gem_comments(str(data.file))
    # df = pd.read_csv(str(data.file), sep=sep, comment='#', header=0)
    df = pd.read_csv(str(data.file), sep=sep, header=comments_lines, engine='pyarrow')
    if 'MIDCounts' in df.columns:
        df.rename(columns={'MIDCounts': 'UMICount'}, inplace=True)
    elif 'MIDCount' in df.columns:
        df.rename(columns={'MIDCount': 'UMICount'}, inplace=True)
    if 'CellID' in df.columns:
        df.rename(columns={'CellID': 'cell_id'}, inplace=True)
    if 'label' in df.columns:
        df.rename(columns={'label': 'cell_id'}, inplace=True)
    dropna_subset = ['geneID', 'x', 'y', 'UMICount']
    if 'cell_id' in df.columns:
        dropna_subset.append('cell_id')
    if 'geneName' in df.columns:
        dropna_subset.append('geneName')

    df.dropna(
        subset=dropna_subset,
        axis=0,
        inplace=True
    )
    gdf = None
    if data.bin_type == 'cell_bins':
        gdf = parse_cell_bin_coor(df)
    else:
        if center_coordinates:
            df = parse_bin_coor(df, bin_size)
        else:
            df = parse_bin_coor_no_offset(df, bin_size)
    
    if gene_name_index and 'geneName' in df.columns:
        df['geneID'] = df['geneName']

    cells = df['cell_id'].unique()
    genes = df['geneID'].unique()
    cells_dict = dict(zip(cells, range(0, len(cells))))
    genes_dict = dict(zip(genes, range(0, len(genes))))
    rows = df['cell_id'].map(cells_dict)
    cols = df['geneID'].map(genes_dict)
    # logger.info(f'the martrix has {len(cells)} cells, and {len(genes)} genes.')
    exp_matrix = csr_matrix((df['UMICount'], (rows, cols)), shape=(cells.shape[0], genes.shape[0]), dtype=np.int32)
    data.cells = Cell(cell_name=cells)
    data.genes = Gene(gene_name=genes)

    if not gene_name_index and 'geneName' in df.columns:
        gene_names = df.groupby(by='geneID').aggregate({'geneName': lambda n: np.unique(n)[0]})['geneName']
        data.genes['real_gene_name'] = gene_names

    data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
    if data.bin_type == 'bins':
        # data.position = df.loc[:, ['x_center', 'y_center']].drop_duplicates().values
        data.position = df.loc[:, ['bin_x', 'bin_y']].drop_duplicates().values
    else:
        data.position = gdf.loc[cells][['x_center', 'y_center']].values
        data.cells.cell_point = gdf.loc[cells]['cell_point'].values
    data.position = data.position.astype(np.uint32)
    data.offset_x = df['x'].min()
    data.offset_y = df['y'].min()
    resolution = 500
    for chip_name in CHIP_RESOLUTION.keys():
        if data.sn[0:len(chip_name)] == chip_name:
            resolution = CHIP_RESOLUTION[chip_name]
            break
    data.attr = {
        'minX': df['x'].min(),
        'minY': df['y'].min(),
        'maxX': df['x'].max(),
        'maxY': df['y'].max(),
        'minExp': data.exp_matrix.min(),  # noqa
        'maxExp': data.exp_matrix.max(),  # noqa
        'resolution': resolution,
    }
    data.center_coordinates = center_coordinates
    logger.info(f'the martrix has {data.cell_names.size} cells, and {data.gene_names.size} genes.')
    return data


def parse_bin_coor(df, bin_size):
    """
    merge bins to a bin unit according to the bin size, also calculate the center coordinate of bin unit,
    and generate cell id of bin unit using the coordinate after merged.

    :param df: a dataframe of the bin file.
    :param bin_size: the size of bin to merge.
    :return:
    """
    x_min = df['x'].min()
    y_min = df['y'].min()
    df['bin_x'] = merge_bin_coor(df['x'].values, x_min, bin_size)
    df['bin_y'] = merge_bin_coor(df['y'].values, y_min, bin_size)
    df['cell_id'] = df['bin_x'].astype(str) + '_' + df['bin_y'].astype(str)
    df['bin_x'] = get_bin_center(df['bin_x'], x_min, bin_size)
    df['bin_y'] = get_bin_center(df['bin_y'], y_min, bin_size)
    return df


def parse_bin_coor_no_offset(df: pd.DataFrame, bin_size: int):
    df['bin_x'] = ((df['x'] // bin_size) * bin_size).astype(np.uint64)
    df['bin_y'] = ((df['y'] // bin_size) * bin_size).astype(np.uint64)
    df['cell_id'] = np.bitwise_or(
        np.left_shift(df['bin_x'], 32),
        df['bin_y']
    )
    df.sort_values(by=['geneID', 'cell_id'], inplace=True)
    df['cell_id'] = df['cell_id'].astype('U')
    return df


def parse_cell_bin_coor(df):
    gdf = df.groupby('cell_id').apply(lambda x: make_multipoint(x))
    return gdf


def make_multipoint(x):
    p = [Point(i) for i in zip(x['x'], x['y'])]
    mlp = MultiPoint(p).convex_hull
    x_center = mlp.centroid.x
    y_center = mlp.centroid.y
    return pd.Series({'cell_point': mlp, 'x_center': x_center, 'y_center': y_center})


def merge_bin_coor(coor: np.ndarray, coor_min: int, bin_size: int):
    return np.floor((coor - coor_min) / bin_size).astype(np.int32)


def get_bin_center(bin_coor: np.ndarray, coor_min: int, bin_size: int):
    return bin_coor * bin_size + coor_min + int(bin_size / 2)


def to_interval(interval_string: str):
    if interval_string.lower() == 'nan':
        return np.NaN
    [left, right] = interval_string[1:-1].split(', ')
    interval = pd.Interval(float(left), float(right))
    return interval


@ReadWriteUtils.check_file_exists
def read_stereo_h5ad(
        file_path: str,
        use_raw: bool = True,
        use_result: bool = True,
        bin_type: str = None,
        bin_size: int = None
):
    """
    Read the H5ad file, and generate the StereoExpData object.

    Parameters
    ----------------------
    file_path
        the path to input H5ad file.
    use_raw
        whether to read data of `self.raw`.
    use_result
        whether to read `result` and `res_key`.
    bin_type
        the bin type includes `'bins'` or `'cell_bins'`.
    bin_size
        the size of bin to merge, when `bin_type` is set to `'bins'`.

    Returns
    --------------------
    An object of StereoExpData.
    """

    data = StereoExpData(file_path=file_path)
    if not data.file.exists():
        logger.error('the input file is not exists, please check!')
        raise FileExistsError('the input file is not exists, please check!')
    with h5py.File(data.file, mode='r') as f:
        data = _read_stereo_h5ad_from_group(f, data, use_raw, use_result, bin_type, bin_size)
    return data


def _read_stereo_h5ad_from_group(f: Union[h5py.File, h5py.Group], data: StereoExpData, use_raw, use_result, bin_type=None, bin_size=None):
    # read data
    data.bin_type = bin_type if bin_type is not None else 'bins'
    data.bin_size = bin_size if bin_size is not None else 1
    not_data_attr_keys = {'bin_type', 'bin_size', 'merged'}
    if f.attrs is not None:
        data.attr = {}
        for key, value in f.attrs.items():
            if key not in not_data_attr_keys:
                data.attr[key] = value
            else:
                setattr(data, key, value)
    for k in f.keys():
        if k == 'cells':
            data.cells = h5ad.read_group(f[k])
        elif k == 'genes':
            data.genes = h5ad.read_group(f[k])
            if 'mean_bin' in data.genes:
                data.genes['mean_bin'] = [to_interval(interval_string) for interval_string in data.genes['mean_bin']]
        elif k == 'position':
            position = h5ad.read_dataset(f[k])
            data.position = position[:, [0, 1]]
            if position.shape[1] >= 3:
                data.position_z = position[:, [2]]
        elif k == 'bin_type':
            data.bin_type = h5ad.read_dataset(f[k])
        elif k == 'bin_size':
            data.bin_size = h5ad.read_dataset(f[k])
        elif k == 'merged':
            data.merged = h5ad.read_dataset(f[k])
        elif k == 'exp_matrix':
            if isinstance(f[k], h5py.Group):
                data.exp_matrix = h5ad.read_group(f[k])
            else:
                data.exp_matrix = h5ad.read_dataset(f[k])
        elif k == 'sn':
            sn_data = h5ad.read_group(f[k])
            if sn_data.shape[0] == 1:
                data.sn = str(sn_data['sn'][0])
            else:
                data.sn = {}
                for _, row in sn_data.iterrows():
                    batch, sn = row[0], row[1]
                    data.sn[str(batch)] = str(sn)

    # read raw
    if use_raw is True and 'exp_matrix@raw' in f.keys():
        data.tl.raw = StereoExpData()
        if isinstance(f['exp_matrix@raw'], h5py.Group):
            data.tl.raw.exp_matrix = h5ad.read_group(f['exp_matrix@raw'])
        else:
            data.tl.raw.exp_matrix = h5ad.read_dataset(f['exp_matrix@raw'])
        if 'cells@raw' in f.keys():
            data.tl.raw.cells = h5ad.read_group(f['cells@raw'])
        else:
            data.tl.raw.cells = deepcopy(data.cells)
        if 'genes@raw' in f.keys():
            data.tl.raw.genes = h5ad.read_group(f['genes@raw'])
        else:
            data.tl.raw.genes = deepcopy(data.genes)
        if 'position@raw' in f.keys():
            position = h5ad.read_dataset(f['position@raw'])
            data.tl.raw.position = position[:, [0, 1]]
            if position.shape[1] >= 3:
                data.tl.raw.position_z = position[:, [2]]
        else:
            data.tl.raw.position = deepcopy(data.position)

    # read key_record and result
    if use_result is True and 'key_record' in f.keys():
        h5ad.read_key_record(f['key_record'], data.tl.key_record)
        _read_stereo_h5_result(data.tl.key_record, data, f)
    return data


def _read_stereo_h5_result(key_record: dict, data: StereoExpData, f: Union[h5py.File, h5py.Group]):
    import ast
    from ..utils.pipeline_utils import cell_cluster_to_gene_exp_cluster
    key_record = deepcopy(key_record)
    for analysis_key in list(key_record.keys()):
        res_keys = key_record[analysis_key]
        for res_key in res_keys:
            if analysis_key == 'hvg':
                # hvg_df = h5ad.read_group(f[f'{res_key}@hvg'])
                # # str to interval
                # if 'mean_bin' in hvg_df.columns:
                #     hvg_df['mean_bin'] = [to_interval(interval_string) for interval_string in hvg_df['mean_bin']]
                # data.tl.result[res_key] = hvg_df
                hvg_columns = h5ad.read_dataset(f[f'{res_key}@hvg'])
                dict.setdefault(data.tl.result, res_key, list(hvg_columns))
            if analysis_key in ['pca', 'umap', 'totalVI', 'spatial_alignment_integration']:
                data.tl.result[res_key] = pd.DataFrame(h5ad.read_dataset(f[f'{res_key}@{analysis_key}']))
                if analysis_key == 'pca':
                    variance_ratio_key = f'{res_key}_variance_ratio'
                    if f'{variance_ratio_key}@{analysis_key}_variance_ratio' in f.keys():
                        data.tl.result[variance_ratio_key] = h5ad.read_dataset(f[f'{variance_ratio_key}@{analysis_key}_variance_ratio'])  # noqa
            if analysis_key == 'neighbors':
                data.tl.result[res_key] = {
                    # 'neighbor': h5ad.read_group(f[f'neighbor@{res_key}@neighbors']),
                    'neighbor': None,
                    'connectivities': h5ad.read_group(f[f'connectivities@{res_key}@neighbors']),
                    'nn_dist': h5ad.read_group(f[f'nn_dist@{res_key}@neighbors'])
                }
                if f'neighbor@{res_key}@neighbors' in f:
                    data.tl.result[res_key]['neighbor'] = h5ad.read_group(f[f'neighbor@{res_key}@neighbors'])
                if f'n_neighbors@{res_key}@neighbors' in f:
                    data.tl.result[res_key]['n_neighbors'] = h5ad.read_dataset(f[f'n_neighbors@{res_key}@neighbors'])
                if f'method@{res_key}@neighbors' in f:
                    data.tl.result[res_key]['method'] = h5ad.read_dataset(f[f'method@{res_key}@neighbors'])
                if f'metric@{res_key}@neighbors' in f:
                    data.tl.result[res_key]['metric'] = h5ad.read_dataset(f[f'metric@{res_key}@neighbors'])
            if analysis_key == 'cluster':
                if f'{res_key}@cluster' in f:
                    data.tl.result[res_key] = h5ad.read_group(f[f'{res_key}@cluster'])
                gene_cluster_res_key = f'gene_exp_{res_key}'
                if ('gene_exp_cluster' not in data.tl.key_record) or (
                        gene_cluster_res_key not in data.tl.key_record['gene_exp_cluster']):
                    gene_cluster_res = cell_cluster_to_gene_exp_cluster(data, res_key)
                    if gene_cluster_res is not False:
                        data.tl.result[gene_cluster_res_key] = gene_cluster_res
                        data.tl.reset_key_record('gene_exp_cluster', gene_cluster_res_key)
            if analysis_key == 'sct':
                data.tl.result[res_key] = [
                    {
                        'counts': h5ad.read_group(f[f'exp_matrix@{res_key}@sct_counts']),
                        'data': h5ad.read_group(f[f'exp_matrix@{res_key}@sct_data']),
                        'scale.data': h5ad.read_group(f[f'exp_matrix@{res_key}@sct_scale']),
                    },
                    {
                        'top_features': h5ad.read_dataset(f[f'genes@{res_key}@sct_top_features']),
                        'umi_genes': h5ad.read_dataset(f[f'genes@{res_key}@sct']),
                        'umi_cells': h5ad.read_dataset(f[f'cells@{res_key}@sct']),
                    }
                ]
                data.tl.result[res_key][0]['scale.data'] = pd.DataFrame(
                    data.tl.result[res_key][0]['scale.data'].toarray(),
                    columns=data.tl.result[res_key][1]['umi_cells'],
                    index=h5ad.read_dataset(f[f'genes@{res_key}@sct_scale_genename']),
                )
            if analysis_key == 'gene_exp_cluster':
                data.tl.result[res_key] = h5ad.read_group(f[f'{res_key}@gene_exp_cluster'])
            if analysis_key == 'marker_genes':
                clusters = h5ad.read_dataset(f[f'clusters_record@{res_key}@marker_genes'])
                data.tl.result[res_key] = {}
                for cluster in clusters:
                    cluster_key = f'{cluster}@{res_key}@marker_genes'
                    if cluster != 'parameters':
                        data.tl.result[res_key][cluster] = h5ad.read_group(f[cluster_key])
                    else:
                        parameters_df: pd.DataFrame = h5ad.read_group(f[cluster_key])
                        data.tl.result[res_key]['parameters'] = {}
                        for _, row in parameters_df.iterrows():
                            name = row['name']
                            value = row['value']
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            data.tl.result[res_key]['parameters'][name] = value
            if analysis_key == 'cell_cell_communication':
                data.tl.result[res_key] = {}
                for key in ['means', 'significant_means', 'deconvoluted', 'pvalues']:
                    full_key = f'{res_key}@{key}@cell_cell_communication'
                    if full_key in f.keys():
                        data.tl.result[res_key][key] = h5ad.read_group(f[full_key])
                parameters_df: pd.DataFrame = h5ad.read_group(f[f'{res_key}@parameters@cell_cell_communication'])
                data.tl.result[res_key]['parameters'] = {}
                for i, row in parameters_df.iterrows():
                    name = row['name']
                    value = row['value']
                    data.tl.result[res_key]['parameters'][name] = value
            if analysis_key == 'regulatory_network_inference':
                data.tl.result[res_key] = {}
                for key in ['regulons', 'auc_matrix', 'adjacencies']:
                    full_key = f'{res_key}@{key}@regulatory_network_inference'
                    if full_key in f.keys():
                        if key == 'regulons':
                            data.tl.result[res_key][key] = ast.literal_eval(h5ad.read_dataset(f[full_key]))
                        else:
                            data.tl.result[res_key][key] = h5ad.read_group(f[full_key])
            if analysis_key in ['co_occurrence']:
                data.tl.result[res_key] = {}
                for full_key in f.keys():
                    if not full_key.endswith(analysis_key):
                        continue
                    data_key = full_key.split('@')[1]
                    data.tl.result[res_key][data_key] = h5ad.read_group(f[full_key])

def _read_anndata_from_group(f: h5py.Group) -> AnnBasedStereoExpData:
    from distutils.version import StrictVersion
    from anndata import __version__ as anndata_version

    if StrictVersion(anndata_version) < StrictVersion('0.8.0'):
        from anndata._io.utils import read_attribute as read_elem
    else:
        from anndata._io.specs.registry import read_elem
    adata = AnnData(
        **{k: read_elem(f[k]) for k in f.keys()}
    )
    data = AnnBasedStereoExpData(based_ann_data=adata)
    # if 'key_record' in adata.uns:
    #     data.tl.key_record = {k: list(v) for k, v in adata.uns['key_record'].items()}
    #     del adata.uns['key_record']
    data.merged = f.attrs.get('merged', False)
    data.spatial_key = f.attrs.get('spatial_key', 'spatial')
    return data

@ReadWriteUtils.check_file_exists
def read_h5ms(file_path, use_raw=True, use_result=True):
    """
    Load a h5ms file as an object of MSData

    :param file_path: The path of h5ms file to be loaded.
    :param use_raw: Whether to load the raw data of each StereoExpData in MSData, defaults to True.
    :param use_result: Whether to load the analysis results which had been saved in h5ms file, defaults to True.

    :return: An object of MSData
    """
    from stereo.core.ms_data import MSData
    with h5py.File(file_path, mode='r') as f:
        # ms_data = MSData()
        data_list = []
        merged_data = None
        names = []
        var_type = None
        relationship = None
        scopes_data = {}
        result_keys = {}
        # result = {}
        for k in f.keys():
            if k == 'sample':
                slice_keys = list(f[k].keys())
                slice_keys.sort(key=lambda k: int(k.split('_')[1]))
                for one_slice_key in slice_keys:
                    # data = _read_stereo_h5ad_from_group(f[k][one_slice_key], StereoExpData(), use_raw, use_result)
                    encoding_type = f[k][one_slice_key].attrs.get('encoding-type', 'stereo_exp_data')
                    if encoding_type == 'anndata':
                        data = _read_anndata_from_group(f[k][one_slice_key])
                    else:
                        data = _read_stereo_h5ad_from_group(f[k][one_slice_key], StereoExpData(), use_raw, use_result)
                    data_list.append(data)
            elif k == 'sample_merged':
                for mk in f[k].keys():
                    # scope_data = _read_stereo_h5ad_from_group(f[k][mk], StereoExpData(), use_raw, use_result)
                    encoding_type = f[k][mk].attrs.get('encoding-type', 'stereo_exp_data')
                    if encoding_type == 'anndata':
                        scope_data = _read_anndata_from_group(f[k][mk])
                    else:
                        scope_data = _read_stereo_h5ad_from_group(f[k][mk], StereoExpData(), use_raw, use_result)
                    scopes_data[mk] = scope_data
                    if f[k][mk].attrs is not None:
                        merged_from_all = f[k][mk].attrs.get('merged_from_all', False)
                        if merged_from_all:
                            merged_data = scope_data
                # merged_data = StereoExpData()
                # merged_data = _read_stereo_h5ad_from_group(f[k], merged_data, use_raw, use_result)  # noqa
            elif k == 'names':
                names = h5ad.read_dataset(f[k])
                if isinstance(names, np.ndarray):
                    names = names.tolist()
            elif k == 'var_type':
                var_type = h5ad.read_dataset(f[k])
            elif k == 'relationship':
                relationship = h5ad.read_dataset(f[k])
            elif k == 'result_keys':
                for rk in f[k].keys():
                    result_keys[rk] = list(h5ad.read_dataset(f[k][rk]))
            # elif k == 'mss':
            #     for key in f['mss'].keys():
            #         data = StereoExpData()
            #         data.tl.result = {}
            #         h5ad.read_key_record(f['mss'][key]['key_record'], data.tl.key_record)
            #         _read_stereo_h5_result(data.tl.key_record, data, f['mss'][key])
            #         result[key] = data.tl.result
            else:
                logger.warn(f"{k} not in rules, did not read from h5ms")

        ms_data = MSData(
            _data_list=data_list,
            _names=names,
            _var_type=var_type,
            _relationship=relationship
        )
        ms_data.merged_data = merged_data
        # ms_data.tl.result = result
        ms_data.scopes_data = scopes_data
        ms_data.tl.result_keys = result_keys

        return ms_data


@ReadWriteUtils.check_file_exists
def read_seurat_h5ad(
        file_path: str,
        use_raw: bool = False
):
    """
    Read the H5ad file in Anndata format of Seurat, and generate the StereoExpData object.

    Parameters
    ------------------
    file_path
        the path of input H5ad file.
    use_raw
        whether to read data of `self.raw`.

    Returns
    ----------------------
    An object of StereoExpData.
    """
    data = StereoExpData(file_path=file_path)

    # basic
    # attributes = ["obsm", "varm", "obsp", "varp", "uns", "layers"]
    # df_attributes = ["obs", "var"]

    with h5py.File(data.file, mode='r') as f:

        if 'raw' not in f.keys():
            use_raw = False

        for k in f.keys():
            if k == "raw" or k.startswith("raw."):
                continue
            if k == "X":
                if isinstance(f[k], h5py.Dataset):
                    data.exp_matrix = h5ad.read_dense_as_sparse(f[k], csr_matrix, 10000)
                else:
                    data.exp_matrix = h5ad.read_group(f[k])
            elif k == "obs":
                cells_df = h5ad.read_dataframe(f[k])
                data.cells.cell_name = cells_df.index.values
                data.cells.total_counts = cells_df['total_counts'] if 'total_counts' in cells_df.keys() else None
                data.cells.pct_counts_mt = cells_df['pct_counts_mt'] if 'pct_counts_mt' in cells_df.keys() else None
                data.cells.n_genes_by_counts = cells_df[
                    'n_genes_by_counts'] if 'n_genes_by_counts' in cells_df.keys() else None
                data.position = cells_df[['x', 'y']].to_numpy(dtype=np.uint32)
                for cluster_key in f['obs']['__categories'].keys():
                    if cluster_key == 'orig.ident':
                        continue
                    data.tl.result[cluster_key] = pd.DataFrame(
                        {'bins': data.cells.cell_name, 'group': cells_df[cluster_key].values})
                    data.tl.key_record['cluster'].append(cluster_key)
            elif k == "var":
                genes_df = h5ad.read_dataframe(f[k])
                data.genes.gene_name = genes_df.index.values
                if 'highly_variable' in genes_df:
                    data.tl.result['highly_variable_genes'] = pd.DataFrame({
                        'means': genes_df.means.values,
                        'dispersions': genes_df.dispersions.values,
                        'dispersions_norm': genes_df.dispersions_norm.values,
                        'highly_variable': genes_df.highly_variable.values == 1
                    }, index=genes_df.index.values)
                    data.tl.key_record['hvg'].append('highly_variable_genes')
            elif k == 'obsm':
                for key in f['obsm'].keys():
                    if key == 'X_spatial':
                        continue
                    if key == 'X_pca':
                        data.tl.result['pca'] = pd.DataFrame(h5ad.read_dataset(f['obsm']['X_pca']))
                        data.tl.key_record['pca'].append('pca')
                    elif key == 'X_umap':
                        data.tl.result['umap'] = pd.DataFrame(h5ad.read_dataset(f['obsm']['X_umap']))
                        data.tl.key_record['umap'].append('umap')
            else:  # Base case
                pass
        if use_raw:
            data.tl.raw = StereoExpData()
            if isinstance(f['raw']['X'], h5py.Dataset):
                data.tl.raw.exp_matrix = h5ad.read_dense_as_sparse(f['raw']['X'], csr_matrix, 10000)
            else:
                data.tl.raw.exp_matrix = h5ad.read_group(f['raw']['X'])
            if 'obs' in f['raw']:
                cells_df = h5ad.read_dataframe(f[k])
                data.tl.raw.cells.cell_name = cells_df.index.values
                data.position = cells_df[['x', 'y']].to_numpy(dtype=np.uint32)
            else:
                data.tl.raw.cells.cell_name = data.cells.cell_name.copy()
                data.tl.raw.position = data.position.copy()
            if 'var' in f['raw']:
                genes_df = h5ad.read_dataframe(f['raw']['var'])
                data.tl.raw.genes.gene_name = genes_df.index.values
            else:
                data.tl.raw.genes.gene_name = data.genes.gene_name.copy()
    return data


@ReadWriteUtils.check_file_exists
def read_ann_h5ad(
        file_path: str,
        spatial_key: Optional[str] = "spatial",
        bin_type: str = None,
        bin_size: int = None,
        resolution: Optional[int] = 500
):
    """
    Read the H5ad file in Anndata format of Scanpy, and generate the StereoExpData object.

    Parameters
    ------------------
    file_path
        the path to input H5ad file.
    spatial_key
        use `.obsm['spatial_key']` as coordiante information.
    bin_type
        the bin type includes `'bins'` or `'cell_bins'`, default to `'bins'`.
    bin_size
        the size of bin to merge, when `bin_type` is set to `'bins'`.
    resolution
        the resolution of chip, default 500nm.
    Returns
    ---------------
    An object of StereoExpData.

    """
    data = StereoExpData(file_path=file_path, bin_type=bin_type, bin_size=bin_size)

    # basic
    # attributes = ["obsm", "varm", "obsp", "varp", "uns", "layers"]
    # df_attributes = ["obs", "var"]

    with h5py.File(data.file, mode='r') as f:

        for k in f.keys():
            if k == "raw" or k.startswith("raw."):
                continue
            if k == "X":
                if isinstance(f[k], h5py.Group):
                    data.exp_matrix = h5ad.read_group(f[k])
                else:
                    data.exp_matrix = h5ad.read_dataset(f[k])

            elif k == "raw":
                assert False, "unexpected raw format"
            elif k == "obs":
                cells_df = h5ad.read_dataframe(f[k])
                data.cells.cell_name = cells_df.index.values
                data.cells.total_counts = cells_df['total_counts'] if 'total_counts' in cells_df.keys() else None
                data.cells.pct_counts_mt = cells_df['pct_counts_mt'] if 'pct_counts_mt' in cells_df.keys() else None
                data.cells.n_genes_by_counts = cells_df[
                    'n_genes_by_counts'] if 'n_genes_by_counts' in cells_df.keys() else None
            elif k == "var":
                genes_df = h5ad.read_dataframe(f[k])
                data.genes.gene_name = genes_df.index.values
            elif k == 'obsm':
                if spatial_key is not None:
                    if isinstance(f[k], h5py.Group):
                        position = h5ad.read_group(f[k])[spatial_key]
                    else:
                        position = h5ad.read_dataset(f[k])[spatial_key]
                    data.position = position[:, [0, 1]]
                    if position.shape[1] >= 3:
                        data.position_z = position[:, [2]]
            elif k == 'uns':
                uns = h5ad.read_group(f[k])
                if 'bin_type' in uns:
                    bin_type = uns['bin_type']
                if 'bin_size' in uns:
                    bin_size = uns['bin_size']
                if 'resolution' in uns:
                    resolution = uns['resolution']
                if 'sn' in uns:
                    sn_data = uns['sn']
                    if sn_data.shape[0] == 1:
                        data.sn = str(sn_data['sn'][0])
                    else:
                        data.sn = {}
                        for _, row in sn_data.iterrows():
                            batch, sn = row[0], row[1]
                            data.sn[str(batch)] = str(sn)
            else:  # Base case
                pass

    data.bin_type = bin_type
    data.bin_size = bin_size
    data.attr = {'resolution': resolution}

    return data


# @ReadWriteUtils.check_file_exists
def read_h5ad(
        file_path: str = None,
        anndata: AnnData = None,
        flavor: str = 'scanpy',
        bin_type: str = None,
        bin_size: int = None,
        spatial_key: str = 'spatial',
        **kwargs
) -> Union[StereoExpData, AnnBasedStereoExpData]:
    """
    Read a h5ad file or load a AnnData object

    Parameters
    ------------------
    file_path
        the path of the h5ad file.
    anndata
        the object of AnnData to be loaded, only available while `flavor` is `'scanpy'`.
        `file_path` and `anndata` only can input one of them.
    flavor
        the format of the h5ad file, defaults to `'scanpy'`.
        `scanpy`: AnnData format of scanpy
        `stereopy`: h5 format of stereo
    bin_type
        the bin type includes `'bins'` and `'cell_bins'`.
    bin_size
        the size of bin to merge, when `bin_type` is set to `'bins'`.
    spatial_key
        the key of spatial information in AnnData.obsm, default to `'spatial'`.
        Only available while `flavor` is `'scanpy'`.
    Returns
    ---------------
    An object of StereoExpData while `flavor` is `'stereopy'` or an object of AnnBasedStereoExpData while `flavor` is `'scanpy'`

    """
    flavor = flavor.lower()

    if flavor == 'stereopy':
        if file_path is None:
            raise ValueError("The 'file_path' must be input.")
        if kwargs is None:
            kwargs = {}
        kwargs['bin_type'] = bin_type
        kwargs['bin_size'] = bin_size
        return read_stereo_h5ad(file_path, **kwargs)
    elif flavor == 'scanpy':
        if file_path is None and anndata is None:
            raise Exception("Must to input the 'file_path' or 'anndata'.")

        if file_path is not None and anndata is not None:
            raise Exception("'file_path' and 'anndata' only can input one of them")
        return AnnBasedStereoExpData(h5ad_file_path=file_path, based_ann_data=anndata, bin_type=bin_type,
                                     bin_size=bin_size, spatial_key=spatial_key, **kwargs)
    else:
        raise ValueError("Invalid value for 'flavor'")


def anndata_to_stereo(
        andata: AnnData,
        use_raw: bool = False,
        spatial_key: Optional[str] = None,
        resolution: Optional[int] = 500
):
    """
    Transform the Anndata object into StereoExpData format.

    Parameters
    -----------------------
    andata
        the input Anndata object.
    use_raw
        use `anndata.raw.X` if True, otherwise `anndata.X`.
    spatial_key
        use `.obsm['spatial_key']` as coordiante information.
    resolution
        the resolution of chip, default 500nm.
    Returns
    ---------------------
    An object of StereoExpData.
    """
    # data matrix,including X,raw,layer
    data = StereoExpData()
    data.exp_matrix = andata.raw.X if use_raw else andata.X
    # obs -> cell
    data.cells.cell_name = np.array(andata.obs_names)
    data.cells.n_genes_by_counts = andata.obs[
        'n_genes_by_counts'] if 'n_genes_by_counts' in andata.obs.columns.tolist() else None
    data.cells.total_counts = andata.obs['total_counts'] if 'total_counts' in andata.obs.columns.tolist() else None
    data.cells.pct_counts_mt = andata.obs['pct_counts_mt'] if 'pct_counts_mt' in andata.obs.columns.tolist() else None
    # var
    data.genes.gene_name = np.array(andata.var_names)
    data.genes.n_cells = andata.var['n_cells'] if 'n_cells' in andata.var.columns.tolist() else None
    data.genes.n_counts = andata.var['n_counts'] if 'n_counts' in andata.var.columns.tolist() else None
    # position
    if spatial_key is not None:
        position = andata.obsm[spatial_key]
        data.position = position[:, [0, 1]]
        if position.shape[1] >= 3:
            data.position_z = position[:, [2]]
    if 'bin_type' in andata.uns:
        data.bin_type = andata.uns['bin_type']
    if 'bin_size' in andata.uns:
        data.bin_size = andata.uns['bin_size']
    if 'resolution' in andata.uns:
        resolution = andata.uns['resolution']
    data.attr = {'resolution': resolution}
    return data


def stereo_to_anndata(
        data: StereoExpData,
        flavor: Literal['scanpy', 'seurat'] = 'scanpy',
        sample_id: str = "sample",
        reindex: bool = False,
        output: str = None,
        base_adata: AnnData = None,
        split_batches: bool = True,
        compression: Optional[Literal["gzip", "lzf"]] = 'gzip'
) -> AnnData:
    """
    Transform the StereoExpData object into Anndata format.

    Parameters
    -----------------------
    data
        the input StereoExpData object.
    flavor
        if you want to convert the output file into h5ad of Seurat, please set `'seurat'`.
    sample_id
        the sample name which will be set as `orig.ident` in obs.
    reindex
        if `True`, the cell index will be reindexed as `{sample_id}:{position_x}_{position_y}` format.
    output
        the path to output h5ad file.
    base_adata
        the input Anndata object.
    split_batches
        Whether to save each batch to a single file if it is a merged data, default to True.
    compression:
        The compression method to be used when saving data as a h5ad file, None means uncompressed, default to gzip.
    Returns
    -----------------
    An object of Anndata.
    """
    if data.merged and split_batches:
        from os import path
        from ..utils.data_helper import split
        data_list = split(data)
        batch = np.unique(data.cells.batch)
        adata_list = []
        if output is not None:
            name, ext = path.splitext(output)
        for bno, d in zip(batch, data_list):
            if output is not None:
                boutput = f"{name}-{d.sn}{ext}"
            else:
                boutput = None
            adata = stereo_to_anndata(d, flavor=flavor, sample_id=sample_id, reindex=reindex, output=boutput,
                                      split_batches=False)
            adata_list.append(adata)
        return adata_list

    from scipy.sparse import issparse

    if isinstance(data, AnnBasedStereoExpData) and base_adata is None:
        base_adata = data._ann_data.copy()

    if base_adata is None:
        adata = AnnData(shape=data.exp_matrix.shape, dtype=np.float64, obs=data.cells.to_df(), var=data.genes.to_df())
        adata.X = data.exp_matrix
    else:
        adata = base_adata

    # sample id
    logger.info(f"Adding {sample_id} in adata.obs['orig.ident'].")
    adata.obs['orig.ident'] = pd.Categorical([sample_id] * adata.obs.shape[0], categories=[sample_id])
    if (data.position is not None) and ('spatial' not in adata.obsm):
        logger.info("Adding data.position as adata.obsm['spatial'] .")
        if data.position_z is not None:
            adata.obsm['spatial'] = np.concatenate([data.position, data.position_z], axis=1)
        else:
            adata.obsm['spatial'] = data.position
        logger.info("Adding data.position as adata.obs['x'] and adata.obs['y'] .")
        cell_names_index = data.cell_names.astype('str')
        adata.obs['x'] = pd.DataFrame(data.position[:, 0], index=cell_names_index)
        adata.obs['y'] = pd.DataFrame(data.position[:, 1], index=cell_names_index)
        if data.position_z is not None:
            adata.obs['z'] = pd.DataFrame(data.position_z, index=cell_names_index)

    if flavor != 'seurat':
        if data.bin_type is not None:
            adata.uns['bin_type'] = data.bin_type
        if data.bin_size is not None:
            adata.uns['bin_size'] = 1 if data.bin_type == 'cell_bins' else data.bin_size
        if data.attr is not None and 'resolution' in data.attr:
            adata.uns['resolution'] = data.attr['resolution']
        if data.bin_type == 'cell_bins' and data.cells.cell_border is not None:
            adata.obsm['cell_border'] = data.cells.cell_border
        if 'key_record' not in adata.uns:
            adata.uns['key_record'] = deepcopy(data.tl.key_record)
        adata.uns['merged'] = data.merged

    if data.sn is not None:
        if isinstance(data.sn, str):
            sn_list = [['-1', data.sn]]
        else:
            sn_list = []
            for bno, sn in data.sn.items():
                sn_list.append([bno, sn])
        adata.uns['sn'] = pd.DataFrame(sn_list, columns=['batch', 'sn'])

    for key in data.tl.key_record.keys():
        if data.tl.key_record[key]:
            if key == 'hvg':
                res_key = data.tl.key_record[key][-1]
                logger.info(f"Adding data.tl.result['{res_key}'] into adata.var .")
                adata.uns[key] = {'params': {}, 'source': 'stereopy', 'method': key}
                for i in data.tl.result[res_key]:
                    adata.var[i] = data.tl.result[res_key][i]
                    if 'mean_bin' in adata.var.columns:
                        adata.var.drop(columns='mean_bin', inplace=True)
            elif key == 'sct':
                res_key = data.tl.key_record[key][-1]
                zero_index_data = data.tl.result[res_key][0]
                one_index_data = data.tl.result[res_key][1]
                logger.info(f"Adding data.tl.result['{res_key}'] into adata.uns['sct_'] .")
                adata.uns['sct_counts'] = csr_matrix(zero_index_data['counts'].T)
                adata.uns['sct_data'] = csr_matrix(zero_index_data['data'].T)
                adata.uns['sct_scale'] = csr_matrix(zero_index_data['scale.data'].T.to_numpy())
                adata.uns['sct_scale_genename'] = list(zero_index_data['scale.data'].index)
                adata.uns['sct_top_features'] = list(one_index_data['top_features'])
                adata.uns['sct_cellname'] = list(one_index_data['umi_cells'].astype('str'))
                adata.uns['sct_genename'] = list(one_index_data['umi_genes'])
            elif key in ['pca', 'umap', 'tsne', 'totalVI', 'spatial_alignment_integration']:
                # pca :we do not keep variance and PCs(for varm which will be into feature.finding in pca of seurat.)
                res_key = data.tl.key_record[key][-1]
                sc_key = f'X_{key}'
                logger.info(f"Adding data.tl.result['{res_key}'] into adata.obsm['{sc_key}'] .")
                adata.obsm[sc_key] = data.tl.result[res_key].values
                if key == 'pca':
                    variance_ratio_key = f'{res_key}_variance_ratio'
                    if variance_ratio_key in data.tl.result:
                        logger.info(f"Adding data.tl.result['{variance_ratio_key}'] into adata.uns['{key}_variance_ratio'] .")
                        adata.uns[variance_ratio_key] = data.tl.result[variance_ratio_key]
            elif key == 'neighbors':
                # neighbor :seurat use uns for conversion to @graph slot, but scanpy canceled neighbors of uns at present. # noqa
                # so this part could not be converted into seurat straightly.
                for res_key in data.tl.key_record[key]:
                    sc_con = 'connectivities' if res_key == 'neighbors' else f'{res_key}_connectivities'
                    sc_dis = 'distances' if res_key == 'neighbors' else f'{res_key}_distances'
                    logger.info(f"Adding data.tl.result['{res_key}']['connectivities'] into adata.obsp['{sc_con}'] .")
                    logger.info(f"Adding data.tl.result['{res_key}']['nn_dist'] into adata.obsp['{sc_dis}'] .")
                    adata.obsp[sc_con] = data.tl.result[res_key]['connectivities']
                    adata.obsp[sc_dis] = data.tl.result[res_key]['nn_dist']
                    logger.info(f"Adding info into adata.uns['{res_key}'].")
                    adata.uns[res_key] = {}
                    adata.uns[res_key]['connectivities_key'] = sc_con
                    adata.uns[res_key]['distance_key'] = sc_dis
                    params = {}
                    if 'n_neighbors' in data.tl.result[res_key]:
                        params['n_neighbors'] = data.tl.result[res_key]['n_neighbors']
                    if 'method' in data.tl.result[res_key]:
                        params['method'] = data.tl.result[res_key]['method']
                    if 'metric' in data.tl.result[res_key]:
                        params['metric'] = data.tl.result[res_key]['metric']
                    if len(params) > 0:
                        adata.uns[res_key]['params'] = params
            elif key == 'cluster':
                cell_name_index = data.cells.cell_name.astype('str')
                for res_key in data.tl.key_record[key]:
                    logger.info(f"Adding data.tl.result['{res_key}'] into adata.obs['{res_key}'] .")
                    adata.obs[res_key] = pd.DataFrame(data.tl.result[res_key]['group'].values, index=cell_name_index)
            elif key in ('gene_exp_cluster', 'cell_cell_communication'):
                for res_key in data.tl.key_record[key]:
                    # logger.info(f"Adding data.tl.result['{res_key}'] into adata.uns['{key}@{res_key}']")
                    # adata.uns[f"{key}@{res_key}"] = data.tl.result[res_key]
                    logger.info(f"Adding data.tl.result['{res_key}'] into adata.uns['{res_key}']")
                    adata.uns[res_key] = data.tl.result[res_key]
            # elif key == 'regulatory_network_inference':
            #     for res_key in data.tl.key_record[key]:
            #         logger.info(f"Adding data.tl.result['{res_key}'] into adata.uns['{res_key}'] .")
            #         regulon_key = f'{res_key}_regulons'
            #         res_key_data = data.tl.result[res_key]
            #         adata.uns[regulon_key] = res_key_data['regulons']
            #         auc_matrix_key = f'{res_key}_auc_matrix'
            #         adata.uns[auc_matrix_key] = res_key_data['auc_matrix']
            #         adjacencies_key = f'{res_key}_adjacencies'
            #         adata.uns[adjacencies_key] = res_key_data['adjacencies']
            elif key in ('co_occurrence', 'regulatory_network_inference'):
                for res_key in data.tl.key_record[key]:
                    logger.info(f"Adding data.tl.result['{res_key}'] into adata.uns['{res_key}'] .")
                    adata.uns[res_key] = data.tl.result[res_key]
            elif key == 'marker_genes':
                for res_key in data.tl.key_record[key]:
                    uns_key = _BaseResult.RENAME_DICT.get(res_key, res_key)
                    adata.uns[uns_key] = transform_marker_genes_to_anndata(data.tl.result[res_key])
            elif key == 'spatial_hotspot':
                for res_key in data.tl.key_record[key]:
                    if res_key in adata.uns:
                        del adata.uns[res_key]
                if 'key_record' in adata.uns:
                    adata.uns['key_record']['spatial_hotspot'] = []
            else:
                continue

    if data.tl.raw is not None:
        if flavor == 'seurat':
            # keep same shape between @counts and @data for seurat,because somtimes dim of sct are not the same.
            logger.info("Adding data.tl.raw.exp_matrix as adata.uns['raw_counts'] .")
            adata.uns['raw_counts'] = data.tl.raw.exp_matrix if issparse(data.tl.raw.exp_matrix) \
                else csr_matrix(data.tl.raw.exp_matrix)
            list_cell_names = data.tl.raw.cell_names.astype(str)
            adata.uns['raw_cellname'] = list(list_cell_names)
            adata.uns['raw_genename'] = list(data.tl.raw.gene_names)
            if data.tl.raw.position is not None and reindex:
                logger.info("Reindex as adata.uns['raw_cellname'] .")
                raw_sample = pd.DataFrame(['sample'] * data.tl.raw.cell_names.shape[0], index=list_cell_names)
                raw_x = pd.DataFrame(data.tl.raw.position[:, 0].astype(str), index=list_cell_names)
                raw_y = pd.DataFrame(data.tl.raw.position[:, 1].astype(str), index=list_cell_names)
                new_ix = np.array(raw_sample + "_" + raw_x + "_" + raw_y).tolist()
                adata.uns['raw_cellname'] = new_ix
        else:
            logger.info("Adding data.tl.raw.exp_matrix as adata.raw .")
            raw_exp = data.tl.raw.exp_matrix
            raw_genes = data.tl.raw.genes.to_df()
            raw_genes.dropna(axis=1, how='all', inplace=True)
            raw_adata = AnnData(shape=raw_exp.shape, var=raw_genes, dtype=np.float64)
            raw_adata.X = raw_exp
            adata.raw = raw_adata

    if reindex:
        logger.info("Reindex adata.X .")
        new_ix = (adata.obs['orig.ident'].astype(str) + ":" + adata.obs['x'].astype(str) + "_" +
                  adata.obs['y'].astype(str)).tolist()
        adata.obs.index = new_ix
        if 'sct_cellname' in adata.uns.keys():
            logger.info("Reindex as adata.uns['sct_cellname'] .")
            adata.uns['sct_cellname'] = new_ix

    if flavor == 'seurat':
        logger.info("Rename QC info.")
        adata.obs.rename(columns={'total_counts': "nCount_Spatial", "n_genes_by_counts": "nFeature_Spatial",
                                  "pct_counts_mt": 'percent.mito'}, inplace=True)

    logger.info("Finished conversion to anndata.")

    if output is not None:
        adata.write_h5ad(output, compression=compression)
        logger.info(f"Finished output to {output}")

    return adata


# def check_file(path, prefix, suffix):
#     filename = f"{path}/{prefix}{suffix}"
#     if os.path.isfile(filename):
#         return filename
#     elif suffix in {"matrix.mtx", "barcodes.tsv"} and os.path.isfile(f"{filename}.gz"):
#         return f'{filename}.gz'
#     elif suffix == "genes.tsv" and os.path.isfile(f'{path}/{prefix}features.tsv.gz'):
#         return f'{path}/{prefix}features.tsv.gz'
#     else:
#         # logger.error(f"{path} is not exist!")
#         # raise FileExistsError(f"can not find {path}/{prefix}{suffix}(or with .gz)!")
#         raise ValueError(f"can not find {filename}(or with .gz).")

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
@ReadWriteUtils.check_file_exists
def read_gef(
        file_path: str,
        bin_type: str = "bins",
        bin_size: int = 100,
        is_sparse: bool = True,
        gene_list: Optional[list] = None,
        region: Optional[list] = None,
        gene_name_index: Optional[bool] = False,
        num_threads: int = -1 
):
    """
    Read the GEF (.h5) file, and generate the StereoExpData object.

    Parameters
    ---------------
    file_path
        the path to input file.
    bin_type
        the bin type includes `'bins'` or `'cell_bins'`.
    bin_size
        the size of bin to merge, which only takes effect when the `bin_type` is set as `'bins'`.
    is_sparse
        the matrix is sparse matrix, if `True`, otherwise `np.ndarray`.
    gene_list
        select targeted data based on the gene list.
    region
        restrict data to the region condition, like [minX, maxX, minY, maxY].
    gene_name_index
        `True` to set gene name as index if the version of gef file is 4 or greater,
        otherwise to set gene id, if the version is 3 or less, `gene_name_index` would
        be forced to `True` because there is no gene id in this case.
    num_threads
        the number of threads to read the data, only available when `bin_type` is `'bins'`.
        -1 means to use all the cores of the machine.

    Returns
    ------------------------
    An object of StereoExpData.
    """
    logger.info('read_gef begin ...')
    from gefpy.utils import gef_is_cell_bin
    is_cell_bin = gef_is_cell_bin(file_path)
    if bin_type == 'cell_bins':
        if not is_cell_bin:
            raise Exception('This file is not the type of CellBin.')

        data = StereoExpData(file_path=file_path, file_format='gef', bin_type=bin_type, bin_size=bin_size)
        from gefpy.cgef_reader_cy import CgefR
        gef = CgefR(file_path, True)
        cellborders_coord_list, coord_count_per_cell = gef.get_cellborders([])
        border_points_count_per_cell = int(coord_count_per_cell / 2)
        cell_borders = cellborders_coord_list.reshape((-1, border_points_count_per_cell, 2))
        if gene_list is not None or region is not None:
            if gene_list is None:
                gene_list = []
            if region is None:
                region = []
            uniq_cell, gene_names, count, cell_ind, gene_ind, dnb_cnt, cell_area, gene_id = gef.get_filtered_data(region, gene_list)
            gene_num = gene_names.size
            cell_num = uniq_cell.size
            if cell_num == 0 or gene_num == 0:
                raise Exception('Can not find the data based on the gene list or region.')
            exp_matrix = csr_matrix((count, (cell_ind, gene_ind)), shape=(cell_num, gene_num), dtype=np.uint32)
            position = np.array(
                list((zip(np.right_shift(uniq_cell, 32), np.bitwise_and(uniq_cell, 0xffffffff))))).astype('uint32')

            data.position = position
            # logger.info(f'the matrix has {cell_num} cells, and {gene_num} genes.')

            uniq_cell_borders = cell_borders[np.in1d(gef.get_cell_names(), uniq_cell)]
            data.cells = Cell(cell_name=uniq_cell, cell_border=uniq_cell_borders)
            data.cells['dnbCount'] = dnb_cnt
            data.cells['area'] = cell_area

            if len(gene_id[0]) == 0:
                gene_name_index = True
            if gene_name_index:
                if len(gene_id[0]) > 0:
                    exp_matrix, gene_names = integrate_matrix_by_genes(gene_names, cell_num,
                                                            exp_matrix.data, exp_matrix.indices, exp_matrix.indptr)
                data.genes = Gene(gene_name=gene_names)
            else:
                data.genes = Gene(gene_name=gene_id)
                data.genes['real_gene_name'] = gene_names

            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
        else:
            # from gefpy.cell_exp_reader import CellExpReader
            # cell_bin_gef = CellExpReader(file_path)
            # data.position = cell_bin_gef.positions
            # logger.info(f'the matrix has {cell_bin_gef.cell_num} cells, and {cell_bin_gef.gene_num} genes.')
            # exp_matrix = csr_matrix((cell_bin_gef.count, (cell_bin_gef.rows, cell_bin_gef.cols)),
            #                         shape=(cell_bin_gef.cell_num, cell_bin_gef.gene_num), dtype=np.uint32)
            # data.cells = Cell(cell_name=cell_bin_gef.cells, cell_border=cell_borders)
            # data.cells['dnbCount'] = cell_bin_gef.dnbCount
            # data.cells['area'] = cell_bin_gef.area
            # data.genes = Gene(gene_name=cell_bin_gef.genes)
            # data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
            cell_names = gef.get_cell_names()
            cell_num = gef.get_cell_num()
            gene_names, gene_id = gef.get_gene_names()
            gene_names = gene_names.astype('U')
            gene_id = gene_id.astype('U')
            gene_num = gef.get_gene_num()
            # logger.info(f'the matrix has {cell_num} cells, and {gene_num} genes.')
            indices, indptr, count = gef.get_sparse_matrix_indices(order='cell')
            exp_matrix = csr_matrix((count, indices, indptr), shape=(cell_num, gene_num), dtype=np.uint32)
            data.cells = Cell(cell_name=cell_names, cell_border=cell_borders)
            cells = gef.get_cells()
            data.cells['dnbCount'] = cells['dnbCount']
            data.cells['area'] = cells['area']
            data.position = np.zeros(shape=(cell_num, 2), dtype=np.uint32)
            data.position[:, 0] = cells['x']
            data.position[:, 1] = cells['y']
            if len(gene_id[0]) == 0:
                gene_name_index = True
            if gene_name_index:
                if len(gene_id[0]) > 0:
                    exp_matrix, gene_names = integrate_matrix_by_genes(gene_names, cell_num, count, indices, indptr)
                data.genes = Gene(gene_name=gene_names)
            else:
                data.genes = Gene(gene_name=gene_id)
                data.genes['real_gene_name'] = gene_names

            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
        data.attr = {
            'resolution': read_gef_info(file_path)['resolution']
        }
        logger.info(f'the matrix has {data.cell_names.size} cells, and {data.gene_names.size} genes.')
        gef.cgef_close()
        return data
    else:
        if is_cell_bin:
            raise Exception('This file is not the type of SquareBin.')

        from gefpy.bgef_reader_cy import BgefR
        if num_threads <= 0:
            from multiprocessing import cpu_count
            num_threads = cpu_count()
        gef = BgefR(file_path, bin_size, num_threads, True)

        data = StereoExpData(file_path=file_path, file_format='gef', bin_type=bin_type, bin_size=bin_size)
        data.offset_x, data.offset_y = gef.get_offset()
        gef_attr = gef.get_exp_attr()
        data.attr = {
            'minX': gef_attr[0],
            'minY': gef_attr[1],
            'maxX': gef_attr[2],
            'maxY': gef_attr[3],
            'maxExp': gef_attr[4],
            'resolution': gef_attr[5],
        }

        if gene_list is not None or region is not None:
            if gene_list is None:
                gene_list = []
            if region is None:
                region = []
            uniq_cell, gene_names, count, cell_ind, gene_ind, gene_id = gef.get_filtered_data(region, gene_list)
            cell_num = uniq_cell.size
            gene_num = gene_names.size
            if cell_num == 0 or gene_num == 0:
                raise Exception('Can not find the data based on the gene list or region.')
            data.position = np.array(
                list((zip(np.right_shift(uniq_cell, 32), np.bitwise_and(uniq_cell, 0xffffffff))))).astype('uint32')
            data.cells = Cell(cell_name=uniq_cell)

            exp_matrix = csr_matrix((count, (cell_ind, gene_ind)), shape=(cell_num, gene_num), dtype=np.uint32)
            if len(gene_id[0]) == 0:
                gene_name_index = True
            if gene_name_index:
                if len(gene_id[0]) > 0:
                    exp_matrix, gene_names = integrate_matrix_by_genes(gene_names, cell_num,
                                                            exp_matrix.data, exp_matrix.indices, exp_matrix.indptr)
                data.genes = Gene(gene_name=gene_names)
            else:
                data.genes = Gene(gene_name=gene_id)
                data.genes['real_gene_name'] = gene_names

            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
        else:
            # gene_num = gef.get_gene_num()
            # uniq_cells, rows, count = gef.get_exp_data()
            # cell_num = len(uniq_cells)
            # logger.info(f'the matrix has {cell_num} cells, and {gene_num} genes.')
            # cols, uniq_genes, _ = gef.get_gene_data()
            # data.position = np.array(list(
            #     (zip(np.right_shift(uniq_cells, 32), np.bitwise_and(uniq_cells, 0xffffffff))))).astype('uint32')
            # exp_matrix = csr_matrix((count, (rows, cols)), shape=(cell_num, gene_num), dtype=np.uint32)
            # data.cells = Cell(cell_name=uniq_cells)
            # data.genes = Gene(gene_name=uniq_genes)
            # data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
            cell_names = gef.get_cell_names()
            cell_num = gef.get_cell_num()
            gene_names, gene_id = gef.get_gene_names()
            gene_num = gef.get_gene_num()
            data.position = np.array(list(
                (zip(np.right_shift(cell_names, 32), np.bitwise_and(cell_names, 0xffffffff))))).astype('uint32')
            data.cells = Cell(cell_name=cell_names)
            if len(gene_id[0]) == 0: # an old version gef file, no gene id
                gene_name_index = True
            
            cell_ind, gene_ind, count = gef.get_sparse_matrix_indices2()
            exp_matrix = csr_matrix((count, (cell_ind, gene_ind)), shape=(cell_num, gene_num), dtype=np.uint32)
            if gene_name_index:
                if len(gene_id[0]) > 0:
                    exp_matrix, gene_names = integrate_matrix_by_genes(gene_names, cell_num,
                                                            exp_matrix.data, exp_matrix.indices, exp_matrix.indptr)
                data.genes = Gene(gene_name=gene_names)
            else:
                data.genes = Gene(gene_name=gene_id)
                data.genes['real_gene_name'] = gene_names
            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
            
        logger.info(f'the matrix has {data.cell_names.size} cells, and {data.gene_names.size} genes.')
    logger.info('read_gef end.')

    return data


@ReadWriteUtils.check_file_exists
def read_gef_info(file_path: str):
    """
    Read the property information of the GEF `(.h5)` file.

    Parameters
    -------------
    file_path
        the path to input file.

    Returns
    --------------------
    An attribute dictionary.

    """
    from gefpy.utils import gef_is_cell_bin

    bin_type = gef_is_cell_bin(file_path)

    h5_file = h5py.File(file_path, 'r')
    info_dict = {}

    if not bin_type:
        logger.info('This is GEF file which contains traditional bin infomation.')
        logger.info('bin_type: bins')

        info_dict['bin_list'] = list(h5_file['geneExp'].keys())
        logger.info('Bin size list: {0}'.format(info_dict['bin_list']))

        if type(h5_file['geneExp']['bin1']['expression'].attrs['resolution']) is np.ndarray:
            info_dict['resolution'] = h5_file['geneExp']['bin1']['expression'].attrs['resolution'][0]
        else:
            info_dict['resolution'] = h5_file['geneExp']['bin1']['expression'].attrs['resolution']
        logger.info('Resolution: {0}'.format(info_dict['resolution']))

        info_dict['gene_count'] = h5_file['geneExp']['bin1']['gene'].shape[0]
        logger.info('Gene count: {0}'.format(info_dict['gene_count']))

        maxX = h5_file['geneExp']['bin1']['expression'].attrs['maxX'][0]
        minX = h5_file['geneExp']['bin1']['expression'].attrs['minX'][0]

        maxY = h5_file['geneExp']['bin1']['expression'].attrs['maxY'][0]
        minY = h5_file['geneExp']['bin1']['expression'].attrs['minY'][0]

        info_dict['offsetX'] = minX
        logger.info('offsetX: {0}'.format(info_dict['offsetX']))

        info_dict['offsetY'] = minY
        logger.info('offsetY: {0}'.format(info_dict['offsetY']))

        info_dict['width'] = maxX - minX
        logger.info('Width: {0}'.format(info_dict['width']))

        info_dict['height'] = maxY - minY
        logger.info('Height: {0}'.format(info_dict['height']))

        info_dict['maxExp'] = h5_file['geneExp']['bin1']['expression'].attrs['maxExp'][0]
        logger.info('Max Exp: {0}'.format(info_dict['maxExp']))

    else:
        logger.info('This is GEF file which contains cell bin infomation.')
        logger.info('bin_type: cell_bins')

        from gefpy.cgef_reader_cy import CgefR
        cgef = CgefR(file_path)

        info_dict['cell_num'] = cgef.get_cell_num()
        logger.info('Number of cells: {0}'.format(info_dict['cell_num']))

        info_dict['gene_num'] = cgef.get_gene_num()
        logger.info('Number of gene: {0}'.format(info_dict['gene_num']))

        info_dict['resolution'] = h5_file.attrs['resolution'][0]
        logger.info('Resolution: {0}'.format(info_dict['resolution']))

        info_dict['offsetX'] = h5_file.attrs['offsetX'][0]
        logger.info('offsetX: {0}'.format(info_dict['offsetX']))

        info_dict['offsetY'] = h5_file.attrs['offsetY'][0]
        logger.info('offsetY: {0}'.format(info_dict['offsetY']))

        info_dict['averageGeneCount'] = h5_file['cellBin']['cell'].attrs['averageGeneCount'][0]
        logger.info('Average number of genes: {0}'.format(info_dict['averageGeneCount']))

        info_dict['maxGeneCount'] = h5_file['cellBin']['cell'].attrs['maxGeneCount'][0]
        logger.info('Maximum number of genes: {0}'.format(info_dict['maxGeneCount']))

        info_dict['averageExpCount'] = h5_file['cellBin']['cell'].attrs['averageExpCount'][0]
        logger.info('Average expression: {0}'.format(info_dict['averageExpCount']))

        info_dict['maxExpCount'] = h5_file['cellBin']['cell'].attrs['maxExpCount'][0]
        logger.info('Maximum expression: {0}'.format(info_dict['maxExpCount']))

    return info_dict

@ReadWriteUtils.check_file_exists
def mudata_to_msdata(
    file_path: str = None,
    sample_names: Optional[Union[np.ndarray, List[str], None]] = None,
    scope_names: Optional[Union[np.ndarray, List[str], None]] = None,
    entire_merged_data_name: Optional[str] = None
):
    """
    Read a h5mu file and convert it to a MSData object.

    :param file_path: The path of the MuData file, defaults to None
    :param sample_names: The names of single samples that are saved in the MuData object, defaults to None,
                            if None, the names starting with 'sample_' will be used.
    :param scope_names: The names of merged samples that are saved in the MuData object, defaults to None,
                            if None, the names like 'scope_[0,1,2]' will be used.
    :param entire_merged_data_name: The name of the merged sample which is merged from all samples, default to None,
                            if None, use the one like 'scope_[0,1,2]' whose square brackets contain index sequence of all samples.

    :return: The MSData object.

    .. note::

        You need to install the mudata package before using this function:

            pip install mudata
    """
    try:
        from mudata import read_h5mu
    except ImportError:
        raise ImportError("Please install mudata first: `pip install mudata`.")
    from stereo.core.ms_data import MSData
    
    mudata = read_h5mu(file_path)

    mod_keys = list(mudata.mod.keys())
    if sample_names is None:
        sample_names = []
        left_mod_keys = []
        for k in mod_keys:
            match = re.match(r'^sample_\d+$', k)
            if match:
                sample_names.append(k)
            else:
                left_mod_keys.append(k)
        sample_names.sort(key=lambda x: int(x.split('_')[1]))
        mod_keys = left_mod_keys
    
    data_list = [AnnBasedStereoExpData(based_ann_data=mudata[n]) for n in sample_names if n in mudata.mod]
    if len(data_list) == 0:
        raise ValueError("No sample data found in the MuData object.")
    if 'names' in mudata.uns:
        names = list(mudata.uns['names'])
    else:
        names = sample_names
    
    var_type = mudata.uns.get('var_type', 'intersect')
    relationship = mudata.uns.get('relationship', 'other')
    relationship_info = mudata.uns.get('relationship_info', {})

    ms_data = MSData(
        _data_list=data_list,
        _names=names,
        _var_type=var_type,
        _relationship=relationship,
        _relationship_info=relationship_info
    )

    if entire_merged_data_name is None:
        entire_merged_data_name = ms_data.generate_scope_key(ms_data.names)
    entire_merged_data = None
    
    if scope_names is None:
        scope_names = []
        left_mod_keys = []
        for k in mod_keys:
            match = re.match(r'^scope_\[\d+(,\d+)*\]$', k)
            if match:
                scope_names.append(k)
            else:
                left_mod_keys.append(k)
        mod_keys = left_mod_keys
    
    scopes_data = {
        n: AnnBasedStereoExpData(based_ann_data=mudata[n]) for n in scope_names if n in mudata.mod
    }
    for k in scopes_data.keys():
        if k == entire_merged_data_name:
            entire_merged_data = scopes_data[k]
            if not re.match(r'^scope_\[\d+(,\d+)*\]$', k):
                del scopes_data[k]
                entire_merged_data_name = ms_data.generate_scope_key(ms_data.names)
                scopes_data[entire_merged_data_name] = entire_merged_data
            break
    if len(scopes_data) > 0:
        ms_data.scopes_data = scopes_data
    ms_data.merged_data = entire_merged_data

    if 'result_keys' in mudata.uns:
        for n, k in mudata.uns['result_keys'].items():
            if n not in ms_data.scopes_data:
                continue
            ms_data.tl.result_keys[n] = list(k)
    
    del mudata

    return ms_data
