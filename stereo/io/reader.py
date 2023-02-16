#!/usr/bin/env python3
# coding: utf-8

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
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from shapely.geometry import Point, MultiPoint

from stereo.io import h5ad
from stereo.core.cell import Cell
from stereo.core.gene import Gene
from stereo.core.stereo_exp_data import StereoExpData
from stereo.utils.read_write_utils import ReadWriteUtils
from stereo.log_manager import logger


@ReadWriteUtils.check_file_exists
def read_gem(file_path, sep='\t', bin_type="bins", bin_size=100, is_sparse=True):
    """
    read the stereo-seq file, and generate the object of StereoExpData.

    :param file_path: input file
    :param sep: separator string
    :param bin_type: the type of bin, if file format is stereo-seq file. `bins` or `cell_bins`.
    :param bin_size: the size of bin to merge. The parameter only takes effect
                     when the value of data.bin_type is 'bins'.
    :param is_sparse: the matrix is sparse matrix if is_sparse is True else np.ndarray

    :return: an object of StereoExpData.
    """
    data = StereoExpData(file_path=file_path, bin_type=bin_type, bin_size=bin_size)
    df = pd.read_csv(str(data.file), sep=sep, comment='#', header=0)
    if 'MIDCounts' in df.columns:
        df.rename(columns={'MIDCounts': 'UMICount'}, inplace=True)
    elif 'MIDCount' in df.columns:
        df.rename(columns={'MIDCount': 'UMICount'}, inplace=True)
    if 'CellID' in df.columns:
        df.rename(columns={'CellID': 'cell_id'}, inplace=True)
    df.dropna(inplace=True)
    gdf = None
    if data.bin_type == 'cell_bins':
        df.rename(columns={'label': 'cell_id'}, inplace=True)
        gdf = parse_cell_bin_coor(df)
    else:
        df = parse_bin_coor(df, bin_size)
    cells = df['cell_id'].unique()
    genes = df['geneID'].unique()
    cells_dict = dict(zip(cells, range(0, len(cells))))
    genes_dict = dict(zip(genes, range(0, len(genes))))
    rows = df['cell_id'].map(cells_dict)
    cols = df['geneID'].map(genes_dict)
    logger.info(f'the martrix has {len(cells)} cells, and {len(genes)} genes.')
    exp_matrix = csr_matrix((df['UMICount'], (rows, cols)), shape=(cells.shape[0], genes.shape[0]), dtype=np.int32)
    data.cells = Cell(cell_name=cells)
    data.genes = Gene(gene_name=genes)
    data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
    if data.bin_type == 'bins':
        data.position = df.loc[:, ['x_center', 'y_center']].drop_duplicates().values
    else:
        data.position = gdf.loc[cells][['x_center', 'y_center']].values
        data.cells.cell_point = gdf.loc[cells]['cell_point'].values
    data.offset_x = df['x'].min()
    data.offset_y = df['y'].min()
    data.attr = {
        'minX': df['x'].min(),
        'minY': df['y'].min(),
        'maxX': df['x'].max(),
        'maxY': df['y'].max(),
        'minExp': data.exp_matrix.toarray().min() if is_sparse else data.exp_matrix.min(),
        'maxExp': data.exp_matrix.toarray().max() if is_sparse else data.exp_matrix.min(),
        'resolution': 0,
    }
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
    df['x_center'] = get_bin_center(df['bin_x'], x_min, bin_size)
    df['y_center'] = get_bin_center(df['bin_y'], y_min, bin_size)
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
    return np.floor((coor - coor_min) / bin_size).astype(np.int)


def get_bin_center(bin_coor: np.ndarray, coor_min: int, bin_size: int):
    return bin_coor * bin_size + coor_min + int(bin_size / 2)


def to_interval(interval_string):
    [left, right] = interval_string[1:-1].split(', ')
    interval = pd.Interval(float(left), float(right))
    return interval

@ReadWriteUtils.check_file_exists
def read_stereo_h5ad(file_path, use_raw=True, use_result=True, ):
    """
    read the h5ad file, and generate the object of StereoExpData.

    :param file_path: the path of input file.
    :param use_raw: whether to save raw data
    :param use_result: whether to save result and res_key

    :return:
    """
    from ..utils.pipeline_utils import cell_cluster_to_gene_exp_cluster
    data = StereoExpData(file_path=file_path)
    if not data.file.exists():
        logger.error('the input file is not exists, please check!')
        raise FileExistsError('the input file is not exists, please check!')
    with h5py.File(data.file, mode='r') as f:
        # read data
        for k in f.keys():
            if k == 'cells':
                data.cells = h5ad.read_group(f[k])
            elif k == 'genes':
                data.genes = h5ad.read_group(f[k])
            elif k == 'position':
                data.position = h5ad.read_dataset(f[k])
            elif k == 'bin_type':
                data.bin_type = h5ad.read_dataset(f[k])
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
                data.tl.raw.position = h5ad.read_dataset(f['position@raw'])
            else:
                data.tl.raw.position = deepcopy(data.position)

        # read key_record and result
        if use_result is True and 'key_record' in f.keys():
            h5ad.read_key_record(f['key_record'], data.tl.key_record)
            for analysis_key, res_keys in data.tl.key_record.items():
                for res_key in res_keys:
                    if analysis_key == 'hvg':
                        hvg_df = h5ad.read_group(f[f'{res_key}@hvg'])
                        # str to interval
                        hvg_df['mean_bin'] = [to_interval(interval_string) for interval_string in hvg_df['mean_bin']]
                        data.tl.result[res_key] = hvg_df
                    if analysis_key in ['pca', 'umap']:
                        data.tl.result[res_key] = pd.DataFrame(h5ad.read_dataset(f[f'{res_key}@{analysis_key}']))
                    if analysis_key == 'neighbors':
                        data.tl.result[res_key] = {
                            'neighbor': h5ad.read_group(f[f'neighbor@{res_key}@neighbors']),
                            'connectivities': h5ad.read_group(f[f'connectivities@{res_key}@neighbors']),
                            'nn_dist': h5ad.read_group(f[f'nn_dist@{res_key}@neighbors'])
                        }
                    if analysis_key == 'cluster':
                        data.tl.result[res_key] = h5ad.read_group(f[f'{res_key}@cluster'])
                        gene_cluster_res_key = f'gene_exp_{res_key}'
                        if ('gene_exp_cluster' not in data.tl.key_record) or (gene_cluster_res_key not in data.tl.key_record['gene_exp_cluster']):
                            data.tl.result[gene_cluster_res_key] = cell_cluster_to_gene_exp_cluster(data.tl, res_key)
                            data.tl.reset_key_record('gene_exp_cluster', gene_cluster_res_key)
                    if analysis_key == 'gene_exp_cluster':
                        data.tl.result[res_key] = h5ad.read_group(f[f'{res_key}@gene_exp_cluster'])
                    if analysis_key == 'marker_genes':
                        clusters = h5ad.read_dataset(f[f'clusters_record@{res_key}@marker_genes'])
                        data.tl.result[res_key] = {}
                        for cluster in clusters:
                            cluster_key = f'{cluster}@{res_key}@marker_genes'
                            data.tl.result[res_key][cluster] = h5ad.read_group(f[cluster_key])
    return data

@ReadWriteUtils.check_file_exists
def read_seurat_h5ad(file_path, use_raw=False):
    """
    read the h5ad file in Anndata format of seurat, and generate the object of StereoExpData.

    :param file_path: h5ad file path.
    :return: StereoExpData obj.
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
                data.cells.n_genes_by_counts = cells_df['n_genes_by_counts'] if 'n_genes_by_counts' in cells_df.keys() else None
                data.position = cells_df[['x', 'y']].to_numpy(dtype=np.uint32)
                for cluster_key in f['obs']['__categories'].keys():
                    if cluster_key == 'orig.ident':
                        continue
                    data.tl.result[cluster_key] = pd.DataFrame({'bins': data.cells.cell_name, 'group': cells_df[cluster_key].values}) 
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
def read_ann_h5ad(file_path, spatial_key: Optional[str] = None):
    """
    read the h5ad file in Anndata format, and generate the object of StereoExpData.

    :param file_path: h5ad file path.
    :param spatial_key: use .obsm[`'spatial_key'`] as position.
    :return: StereoExpData obj.
    """
    data = StereoExpData(file_path=file_path)

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
                data.cells.n_genes_by_counts = cells_df['n_genes_by_counts'] if 'n_genes_by_counts' in cells_df.keys() else None
            elif k == "var":
                genes_df = h5ad.read_dataframe(f[k])
                data.genes.gene_name = genes_df.index.values
                # data.genes.n_cells = genes_df['n_cells']
                # data.genes.n_counts = genes_df['n_counts']
            elif k == 'obsm':
                if spatial_key is not None:
                    if isinstance(f[k], h5py.Group):
                        data.position = h5ad.read_group(f[k])[spatial_key]
                    else:
                        data.position = h5ad.read_dataset(f[k])[spatial_key]
            else:  # Base case
                pass
    return data


def anndata_to_stereo(andata: AnnData, use_raw=False, spatial_key: Optional[str] = None):
    """
    transform the Anndata object into StereoExpData object.

    :param andata: input Anndata object,
    :param use_raw: use andata.raw.X if True else andata.X. Default is False.
    :param spatial_key: use .obsm[`'spatial_key'`] as position.
    :return: StereoExpData obj.
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
    data.position = andata.obsm[spatial_key] if spatial_key is not None else None
    return data


def stereo_to_anndata(data: StereoExpData, flavor='scanpy', sample_id="sample", reindex=False, output=None, split_batches=True):
    """
    transform the StereoExpData object into Anndata object.

    :param data: StereoExpData object
    :param flavor: 'scanpy' or 'seurat'.
    if you want to convert the output_h5ad into h5seurat for seurat, please set 'seurat'.
    :param sample_id: sample name, which will be set as 'orig.ident' in obs.
    :param reindex: if True, the cell index will be reindex as "{sample_id}:{position_x}_{position_y}" format.
    :param output: path of output_file(.h5ad).
    :return: Anndata object
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
            adata = stereo_to_anndata(d, flavor=flavor, sample_id=sample_id, reindex=reindex, output=boutput, split_batches=False)
            adata_list.append(adata)
        return adata_list

    from scipy.sparse import issparse

    if data.tl.raw is None:
        logger.error('convert to AnnData should have raw data')
        raise Exception

    exp = data.tl.raw.exp_matrix if issparse(data.tl.raw.exp_matrix) else csr_matrix(data.tl.raw.exp_matrix)
    cells = data.tl.raw.cells.to_df()
    cells.dropna(axis=1, how='all', inplace=True)
    genes = data.tl.raw.genes.to_df()
    genes.dropna(axis=1, how='all', inplace=True)

    adata = AnnData(X=exp,
                    dtype=np.float64,
                    obs=cells,
                    var=genes,
                    # uns={'neighbors': {'connectivities_key': 'None','distance_key': 'None'}},
                    )
    ##sample id
    logger.info(f"Adding {sample_id} in adata.obs['orig.ident'].")
    adata.obs['orig.ident'] = pd.Categorical([sample_id] * adata.obs.shape[0], categories=[sample_id])
    if data.position is not None:
        logger.info(f"Adding data.position as adata.obsm['spatial'] .")
        adata.obsm['spatial'] = data.position
        # adata.obsm['X_spatial'] = data.position
        logger.info(f"Adding data.position as adata.obs['x'] and adata.obs['y'] .")
        adata.obs['x'] = pd.DataFrame(data.position[:, 0], index=data.cell_names.astype('str'))
        adata.obs['y'] = pd.DataFrame(data.position[:, 1], index=data.cell_names.astype('str'))
    
    if data.sn is not None:
        if isinstance(data.sn, str):
            sn_list = [['-1', data.sn]]
        else:
            sn_list = []
            for bno, sn in data.sn.items():
                sn_list.append([bno, sn])
        adata.uns['sn'] = pd.DataFrame(sn_list, columns=['batch', 'sn'])

    for key in data.tl.key_record.keys():
        if len(data.tl.key_record[key]) > 0:
            if key == 'hvg':
                res_key = data.tl.key_record[key][-1]
                logger.info(f"Adding data.tl.result['{res_key}'] in adata.var .")
                adata.uns[key] = {'params': {}, 'source': 'stereopy', 'method': key}
                for i in data.tl.result[res_key]:
                    if i == 'mean_bin':
                        continue
                    adata.var[i] = data.tl.result[res_key][i]
            elif key == 'sct':
                res_key = data.tl.key_record[key][-1]
                # adata.uns[res_key] = {}
                logger.info(f"Adding data.tl.result['{res_key}'] in adata.uns['sct_'] .")
                adata.uns['sct_counts'] = csr_matrix(data.tl.result[res_key][0]['counts'].T)
                adata.uns['sct_data'] = csr_matrix(data.tl.result[res_key][0]['data'].T)
                adata.uns['sct_scale'] = csr_matrix(data.tl.result[res_key][0]['scale.data'].T.to_numpy())
                adata.uns['sct_top_features'] = list(data.tl.result[res_key][0]['scale.data'].index)
                adata.uns['sct_cellname'] = list(data.tl.result[res_key][1]['umi_cells'].astype('str'))
                adata.uns['sct_genename'] = list(data.tl.result[res_key][1]['umi_genes'])
            elif key in ['pca', 'umap', 'tsne']:
                # pca :we do not keep variance and PCs(for varm which will be into feature.finding in pca of seurat.)
                res_key = data.tl.key_record[key][-1]
                sc_key = f'X_{key}'
                logger.info(f"Adding data.tl.result['{res_key}'] in adata.obsm['{sc_key}'] .")
                adata.obsm[sc_key] = data.tl.result[res_key].values
            elif key == 'neighbors':
                # neighbor :seurat use uns for conversion to @graph slot, but scanpy canceled neighbors of uns at present.
                # so this part could not be converted into seurat straightly.
                for res_key in data.tl.key_record[key]:
                    sc_con = 'connectivities' if res_key == 'neighbors' else f'{res_key}_connectivities'
                    sc_dis = 'distances' if res_key == 'neighbors' else f'{res_key}_distances'
                    logger.info(f"Adding data.tl.result['{res_key}']['connectivities'] in adata.obsp['{sc_con}'] .")
                    logger.info(f"Adding data.tl.result['{res_key}']['nn_dist'] in adata.obsp['{sc_dis}'] .")
                    adata.obsp[sc_con] = data.tl.result[res_key]['connectivities']
                    adata.obsp[sc_dis] = data.tl.result[res_key]['nn_dist']
                    logger.info(f"Adding info in adata.uns['{res_key}'].")
                    adata.uns[res_key] = {}
                    adata.uns[res_key]['connectivities_key'] = sc_con
                    adata.uns[res_key]['distance_key'] = sc_dis
                    # adata.uns[res_key]['connectivities'] = data.tl.result[res_key]['connectivities']
                    # adata.uns[res_key]['distances'] = data.tl.result[res_key]['nn_dist']
            elif key == 'cluster':
                for res_key in data.tl.key_record[key]:
                    logger.info(f"Adding data.tl.result['{res_key}'] in adata.obs['{res_key}'] .")
                    adata.obs[res_key] = pd.DataFrame(data.tl.result[res_key]['group'].values,
                                                      index=data.cells.cell_name.astype('str'))
            elif key == 'gene_exp_cluster':
                for res_key in data.tl.key_record[key]: 
                    adata.uns[res_key] = data.tl.result[res_key]
            else:
                continue

    if data.tl.raw is not None:
        if flavor == 'seurat':
            # keep same shape between @counts and @data for seurat,because somtimes dim of sct are not the same.
            logger.info(f"Adding data.tl.raw.exp_matrix as adata.uns['raw_counts'] .")
            adata.uns['raw_counts'] = data.tl.raw.exp_matrix if issparse(data.tl.raw.exp_matrix) \
                else csr_matrix(data.tl.raw.exp_matrix)
            adata.uns['raw_cellname'] = list(data.tl.raw.cell_names.astype(str))
            adata.uns['raw_genename'] = list(data.tl.raw.gene_names)
            if data.tl.raw.position is not None and reindex:
                logger.info(f"Reindex as adata.uns['raw_cellname'] .")
                raw_sample = pd.DataFrame(['sample'] * data.tl.raw.cell_names.shape[0],
                                          index=data.tl.raw.cell_names.astype('str'))
                raw_x = pd.DataFrame(data.tl.raw.position[:, 0].astype(str), index=data.tl.raw.cell_names.astype('str'))
                raw_y = pd.DataFrame(data.tl.raw.position[:, 1].astype(str), index=data.tl.raw.cell_names.astype('str'))
                new_ix = np.array(raw_sample + "_" + raw_x + "_" + raw_y).tolist()
                adata.uns['raw_cellname'] = new_ix
        else:
            logger.info(f"Adding data.tl.raw.exp_matrix as adata.raw .")
            raw_exp = data.tl.raw.exp_matrix
            raw_genes = data.tl.raw.genes.to_df()
            raw_genes.dropna(axis=1, how='all', inplace=True)
            raw_adata = AnnData(X=raw_exp, var=raw_genes, dtype=np.float64, )
            adata.raw = raw_adata

    if reindex:
        logger.info(f"Reindex adata.X .")
        new_ix = (adata.obs['orig.ident'].astype(str) + ":" + adata.obs['x'].astype(str) + "_" +
                  adata.obs['y'].astype(str)).tolist()
        adata.obs.index = new_ix
        if 'sct_cellname' in adata.uns.keys():
            logger.info(f"Reindex as adata.uns['sct_cellname'] .")
            adata.uns['sct_cellname'] = new_ix

    if flavor == 'seurat':
        logger.info(f"Rename QC info.")
        adata.obs.rename(columns={'total_counts': "nCount_Spatial", "n_genes_by_counts": "nFeature_Spatial",
                                  "pct_counts_mt": 'percent.mito'}, inplace=True)
        # if 'X_pca' not in list(adata.obsm.keys()):
        # logger.info(f"Creating fake info. Please ignore X_ignore in your data.")
        # adata.obsm['X_ignore'] = np.zeros((adata.obs.shape[0], 2))

    # logger.info(f"Adding data.attr in adata.uns.")
    # if data.offset_x is not None:
    #     adata.uns['offset_x'] = data.offset_x
    # if data.offset_y is not None:
    #     adata.uns['offset_y'] = data.offset_y
    # if data.attr is not None:
    #     for key, value in data.attr.items():
    #         adata.uns[key] = value

    logger.info(f"Finished conversion to anndata.")

    if output is not None:
        adata.write_h5ad(output)
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
def read_gef(file_path: str, bin_type="bins", bin_size=100, is_sparse=True, gene_list: list = None,
             region: list = None):
    """
    read the gef(.h5) file, and generate the object of StereoExpData.

    :param file_path: input file
    :param bin_type: bin_type , bins or cell_bins
    :param bin_size: the size of bin to merge. The parameter only takes effect
                  when the value of data.bin_type is 'bins'.
    :param is_sparse: the matrix is sparse matrix if is_sparse is True else np.ndarray
    :param gene_list: restrict to this gene list
    :param region: restrict to this region, [minX, maxX, minY, maxY]

    :return: an object of StereoExpData.
    """
    logger.info(f'read_gef begin ...')
    if bin_type == 'cell_bins':
        data = StereoExpData(file_path=file_path, bin_type=bin_type, bin_size=bin_size)
        from gefpy.cgef_reader_cy import CgefR
        gef = CgefR(file_path)
        cellborders_coord_list, coord_count_per_cell = gef.get_cellborders([])
        border_points_count_per_cell = int(coord_count_per_cell / 2)
        cell_borders = cellborders_coord_list.reshape((-1, border_points_count_per_cell, 2))
        if gene_list is not None or region is not None:
            if gene_list is None:
                gene_list = []
            if region is None:
                region = []
            uniq_cell, gene_names, count, cell_ind, gene_ind = gef.get_filtered_data(region,gene_list)
            gene_num = gene_names.size
            cell_num = uniq_cell.size
            exp_matrix = csr_matrix((count, (cell_ind, gene_ind)), shape=(cell_num, gene_num), dtype=np.uint32)
            position = np.array(list((zip(np.right_shift(uniq_cell, 32), np.bitwise_and(uniq_cell, 0xffffffff))))).astype('uint32')

            data.position = position
            logger.info(f'the matrix has {cell_num} cells, and {gene_num} genes.')

            uniq_cell_borders = cell_borders[np.in1d(gef.get_cell_names(), uniq_cell)]
            data.cells = Cell(cell_name=uniq_cell, cell_border=uniq_cell_borders)
            data.genes = Gene(gene_name=gene_names)

            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
        else:
            from gefpy.cell_exp_reader import CellExpReader
            cell_bin_gef = CellExpReader(file_path)
            data.position = cell_bin_gef.positions
            logger.info(f'the matrix has {cell_bin_gef.cell_num} cells, and {cell_bin_gef.gene_num} genes.')
            exp_matrix = csr_matrix((cell_bin_gef.count, (cell_bin_gef.rows, cell_bin_gef.cols)), shape=(cell_bin_gef.cell_num, cell_bin_gef.gene_num), dtype=np.uint32)
            data.cells = Cell(cell_name=cell_bin_gef.cells, cell_border=cell_borders)
            data.genes = Gene(gene_name=cell_bin_gef.genes)
            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
    else:
        from gefpy.bgef_reader_cy import BgefR
        gef = BgefR(file_path, bin_size, 4)

        data = StereoExpData(file_path=file_path, bin_type=bin_type, bin_size=bin_size)
        data.offset_x, data.offset_y = gef.get_offset()
        gef_attr = gef.get_exp_attr()
        data.attr={
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
            uniq_cell, gene_names, count, cell_ind, gene_ind = gef.get_filtered_data(region, gene_list)
            cell_num = uniq_cell.size
            gene_num = gene_names.size
            logger.info(f'the matrix has {cell_num} cells, and {gene_num} genes.')
            exp_matrix = csr_matrix((count, (cell_ind, gene_ind)), shape=(cell_num, gene_num), dtype=np.uint32)
            position = np.array(list((zip(np.right_shift(uniq_cell, 32), np.bitwise_and(uniq_cell, 0xffffffff))))).astype('uint32')

            data.position = position
            data.cells = Cell(cell_name=uniq_cell)
            data.genes = Gene(gene_name=gene_names)

            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
        else:
            gene_num = gef.get_gene_num()
            uniq_cells, rows, count = gef.get_exp_data()
            cell_num = len(uniq_cells)
            logger.info(f'the matrix has {cell_num} cells, and {gene_num} genes.')
            cols, uniq_genes = gef.get_gene_data()
            data.position = np.array(list(
                (zip(np.right_shift(uniq_cells, 32), np.bitwise_and(uniq_cells, 0xffffffff))))).astype('uint32')
            exp_matrix = csr_matrix((count, (rows, cols)), shape=(cell_num, gene_num), dtype=np.uint32)
            data.cells = Cell(cell_name=uniq_cells)
            data.genes = Gene(gene_name=uniq_genes)
            data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
    logger.info(f'read_gef end.')

    return data

@ReadWriteUtils.check_file_exists
def read_gef_info(file_path: str):
    """
    read the infomation of gef(.h5) file.

    :param file_path: input file
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


# @ReadWriteUtils.check_file_exists
# def read_h5ad(file_path: str, flavor: str = 'scanpy'):
#     '''
#     :param file_path: h5ad file path.
#     :return: `StereoExpData`-like `AnnBasedStereoExpData` obj
#     '''
#     if flavor == 'scanpy':
#         from stereo.core.stereo_exp_data import AnnBasedStereoExpData
#         return AnnBasedStereoExpData(file_path)
#     elif flavor == 'seurat':
#         raise NotImplementedError
#     else:
#         raise Exception
