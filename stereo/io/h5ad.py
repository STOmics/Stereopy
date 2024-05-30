#!/usr/bin/env python3
# coding: utf-8
"""
@file: h5ad.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: wuyiran

change log:
    2021/06/18  create file.
    2022/02/09  write and read neighbors
"""
import h5py
import numpy as np
import pandas as pd
from typing import Union
from types import MappingProxyType
from pandas.api.types import is_categorical_dtype
from scipy import sparse
from packaging import version
from stereo.utils.spmatrix_helper import idx_chunks_along_axis
from functools import singledispatch
from stereo.core.gene import Gene
from stereo.core.cell import Cell
from stereo.algorithm.neighbors import Neighbors

H5PY_V3 = version.parse(h5py.__version__).major >= 3


@singledispatch
def write(v, f, k, *args, **kwargs):
    write_scalar(f, k, v)


@write.register(np.ndarray)
def _(v, f, k):
    write_array(f, k, v)


@write.register(list)
def _(v, f, k):
    write_list(f, k, v)


@write.register(pd.DataFrame)
def _(v, f, k, save_as_matrix=False):
    write_dataframe(f, k, v, save_as_matrix=save_as_matrix)


@write.register(sparse.spmatrix)
def _(v, f, k, sp_format):
    write_spmatrix(f, k, v, sp_format)


@write.register(Gene)
def _(v, f, k):
    write_genes(f, k, v)


@write.register(Cell)
def _(v, f, k):
    write_cells(f, k, v)


@write.register(Neighbors)
def _(v, f, k):
    write_neighbors(f, k, v)


def write_array(f: Union[h5py.File, h5py.Group], key, value, dataset_kwargs=MappingProxyType({})):
    # Convert unicode to fixed length strings
    if value.dtype.kind in {'U', 'O'}:
        value = value.astype(h5py.special_dtype(vlen=str))
    elif value.dtype.names is not None:
        value = _to_hdf5_vlen_strings(value)
    f.create_dataset(key, data=value, **dataset_kwargs)


def write_list(f, key, value, dataset_kwargs=MappingProxyType({})):
    write_array(f, key, np.array(value), dataset_kwargs=dataset_kwargs)


def write_scalar(f, key, value, dataset_kwargs=MappingProxyType({})):
    write_array(f, key, np.array(value), dataset_kwargs=dataset_kwargs)


def write_spmatrix(f, k, v, fmt: str, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    g.attrs['encoding-type'] = f'{fmt}_matrix'
    g.attrs['shape'] = v.shape
    # Allow resizing
    if 'maxshape' not in dataset_kwargs:
        dataset_kwargs = dict(maxshape=(None,), **dataset_kwargs)
    g.create_dataset('data', data=v.data, **dataset_kwargs)
    g.create_dataset('indices', data=v.indices, **dataset_kwargs)
    g.create_dataset('indptr', data=v.indptr, **dataset_kwargs)


# def write_genes(f, k, v: Gene, dataset_kwargs=MappingProxyType({})):
#     g = f.create_group(k)
#     g.attrs['encoding-type'] = 'gene'
#     write_array(g, 'gene_name', v.gene_name, dataset_kwargs)
#     if v.n_cells is not None:
#         write_array(g, 'n_cells', v.n_cells, dataset_kwargs)
#     if v.n_counts is not None:
#         write_array(g, 'n_counts', v.n_counts, dataset_kwargs)
#     if v.mean_umi is not None:
#         write_array(g, 'mean_umi', v.n_counts, dataset_kwargs)
#     if v.real_gene_name is not None:
#         write_array(g, 'real_gene_name', v.real_gene_name, dataset_kwargs)

def write_genes(f: h5py.File, k: str, v: Gene, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    g.attrs['encoding-type'] = 'gene'
    g.attrs['version'] = 'v2'
    write_dataframe(g, 'var', v.to_df(), dataset_kwargs)


# def write_cells(f, k, v: Cell, dataset_kwargs=MappingProxyType({})):
#     g = f.create_group(k)
#     g.attrs['encoding-type'] = 'cell'
#     write_array(g, 'cell_name', v.cell_name, dataset_kwargs)
#     if v.total_counts is not None:
#         write_array(g, 'total_counts', v.total_counts, dataset_kwargs)
#     if v.pct_counts_mt is not None:
#         write_array(g, 'pct_counts_mt', v.pct_counts_mt, dataset_kwargs)
#     if v.n_genes_by_counts is not None:
#         write_array(g, 'n_genes_by_counts', v.n_genes_by_counts, dataset_kwargs)
#     if v.batch is not None:
#         write_array(g, 'batch', v.batch, dataset_kwargs)
#     if v.cell_border is not None:
#         write_array(g, 'cell_border', v.cell_border, dataset_kwargs)

def write_cells(f: h5py.File, k: str, v: Cell, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    g.attrs['encoding-type'] = 'cell'
    g.attrs['version'] = 'v2'
    write_dataframe(g, 'obs', v.to_df(), dataset_kwargs)
    if v.cell_border is not None:
        write_array(g, 'cell_border', v.cell_border, dataset_kwargs)


def write_spmatrix_as_dense(f, key, value, dataset_kwargs=MappingProxyType({})):
    dset = f.create_dataset(key, shape=value.shape, dtype=value.dtype, **dataset_kwargs)
    compressed_axis = int(isinstance(value, sparse.csc_matrix))
    for idx in idx_chunks_along_axis(value.shape, compressed_axis, 1000):
        dset[idx] = value[idx].toarray()


def write_dataframe(f, key, df, dataset_kwargs=MappingProxyType({}), save_as_matrix=False):
    # Check arguments
    for reserved in ('__categories', '_index'):
        if reserved in df.columns:
            raise ValueError(f'{reserved!r} is a reserved name for dataframe columns.')

    col_names = [check_key(c) for c in df.columns]

    if df.index.name is not None:
        index_name = df.index.name
    else:
        index_name = '_index'
    index_name = check_key(index_name)

    group = f.create_group(key)
    group.attrs['encoding-type'] = 'dataframe'
    group.attrs['column-order'] = col_names
    group.attrs['_index'] = index_name
    group.attrs['save-as-matrix'] = save_as_matrix

    write_series(group, index_name, df.index, dataset_kwargs=dataset_kwargs)
    if save_as_matrix:
        write_array(group, 'values', df.values, dataset_kwargs=dataset_kwargs)
    else:
        for col_name, (_, series) in zip(col_names, df.items()):
            write_series(group, col_name, series, dataset_kwargs=dataset_kwargs)


def write_series(group, key, series, dataset_kwargs=MappingProxyType({})):
    # group here is an h5py type, otherwise categoricals won’t write
    if series.dtype == object:  # Assuming it’s string
        group.create_dataset(
            key,
            data=series.values,
            dtype=h5py.special_dtype(vlen=str),
            **dataset_kwargs,
        )
    elif is_categorical_dtype(series):
        # This should work for categorical Index and Series
        categorical: pd.Categorical = series.values
        categories: np.ndarray = categorical.categories.values
        codes: np.ndarray = categorical.codes
        category_key = f'__categories/{key}'

        write_array(group, category_key, categories, dataset_kwargs=dataset_kwargs)
        write_array(group, key, codes, dataset_kwargs=dataset_kwargs)

        group[key].attrs['categories'] = group[category_key].ref
        group[category_key].attrs['ordered'] = categorical.ordered
    else:
        write_array(group, key, series.values, dataset_kwargs=dataset_kwargs)


def write_key_record(f, key, key_record, dataset_kwargs=MappingProxyType({})):
    group = f.create_group(key)
    group.attrs['encoding-type'] = 'key_record'
    for key, res_keys in key_record.items():
        write_list(group, key, res_keys, dataset_kwargs=dataset_kwargs)


def write_neighbors(f, key, neighbors, dataset_kwargs=MappingProxyType({})):
    group = f.create_group(key)
    group.attrs['encoding-type'] = 'neighbors'
    group.attrs['n_neighbors'] = neighbors.n_neighbors
    group.attrs['n_pcs'] = neighbors.n_pcs
    group.attrs['metric'] = neighbors.metric
    group.attrs['method'] = neighbors.method
    group.attrs['knn'] = neighbors.knn
    group.attrs['random_state'] = neighbors.random_state
    write_array(group, 'x', neighbors.x, dataset_kwargs=dataset_kwargs)


def check_key(key):
    """
    Checks that passed value is a valid h5py key.
    Should convert it if there is an obvious conversion path, error otherwise.
    """
    typ = type(key)
    if issubclass(typ, str):
        return str(key)
    else:
        raise TypeError(f'{key} of type {typ} is an invalid key. Should be str.')


def _to_hdf5_vlen_strings(value: np.ndarray) -> np.ndarray:
    """
    This corrects compound dtypes to work with hdf5 files.

    :param value:
    :return:
    """
    new_dtype = []
    for dt_name, (dt_type, _) in value.dtype.fields.items():
        if dt_type.kind in ('U', 'O'):
            new_dtype.append((dt_name, h5py.special_dtype(vlen=str)))
        else:
            new_dtype.append((dt_name, dt_type))
    return value.astype(new_dtype)


def read_dataframe(group) -> pd.DataFrame:
    save_as_matrix = group.attrs.get('save-as-matrix', default=False)
    if save_as_matrix:
        columns = list(group.attrs['column-order'])
    else:
        columns = [c for c in group.attrs['column-order'] if isinstance(group[c], h5py.Dataset)]
    idx_key = group.attrs['_index']
    df = pd.DataFrame(
        {k: read_series(group[k]) for k in columns} if not save_as_matrix else read_dataset(group['values']),
        index=read_series(group[idx_key]),
        columns=list(columns),
    )
    if idx_key != '_index':
        df.index.name = idx_key
    return df


def read_spmatrix(group) -> sparse.spmatrix:
    shape = tuple(group.attrs['shape'])
    dtype = group['data'].dtype
    mtx = sparse.csr_matrix(shape, dtype=dtype) if group.attrs['encoding-type'] == 'csr_matrix' \
        else sparse.csc_matrix(shape, dtype=dtype)
    mtx.data = group['data'][...]
    mtx.indices = group['indices'][...]
    mtx.indptr = group['indptr'][...]
    return mtx


def read_genes(group: h5py.Group) -> Gene:
    version = group.attrs.get('version', 'v1')
    if version == 'v1':
        gene_name = group['gene_name'][...]
        gene = Gene(gene_name=gene_name)
        n_cells = group['n_cells'][...] if 'n_cells' in group.keys() else None
        n_counts = group['n_counts'][...] if 'n_counts' in group.keys() else None
        gene.n_cells = n_cells
        gene.n_counts = n_counts
    else:
        var = read_dataframe(group['var'])
        gene = Gene(var=var)
    return gene


def read_cells(group: h5py.Group) -> Cell:
    version = group.attrs.get('version', 'v1')
    if version == 'v1':
        cell_name = group['cell_name'][...]
        for i in range(cell_name.shape[0]):
            if type(cell_name[i]) is bytes:
                cell_name[i] = cell_name[i].decode()
        cell = Cell(cell_name=cell_name)
        total_counts = group['total_counts'][...] if 'total_counts' in group.keys() else None
        pct_counts_mt = group['pct_counts_mt'][...] if 'pct_counts_mt' in group.keys() else None
        n_genes_by_counts = group['n_genes_by_counts'][...] if 'n_genes_by_counts' in group.keys() else None
        cell.total_counts = total_counts
        cell.pct_counts_mt = pct_counts_mt
        cell.n_genes_by_counts = n_genes_by_counts
    else:
        obs = read_dataframe(group['obs'])
        cell = Cell(obs=obs)
        if 'cell_border' in group.keys():
            cell.cell_border = group['cell_border'][...]
    return cell


def read_series(dataset) -> Union[np.ndarray, pd.Categorical]:
    if 'categories' in dataset.attrs:
        categories = dataset.attrs['categories']
        ordered = False
        if isinstance(categories, h5py.Reference):
            categories_dset = dataset.parent[dataset.attrs['categories']]
            categories = read_dataset(categories_dset)
            ordered = bool(categories_dset.attrs.get('ordered', False))
        else:
            pass
        return pd.Categorical.from_codes(
            read_dataset(dataset), categories, ordered=ordered
        )
    else:
        return read_dataset(dataset)


def read_dataset(dataset: h5py.Dataset):
    if H5PY_V3:
        string_dtype = h5py.check_string_dtype(dataset.dtype)
        if (string_dtype is not None) and (string_dtype.encoding == 'utf-8'):
            dataset = dataset.asstr()
    value = dataset[()]
    if not hasattr(value, 'dtype'):
        return value
    elif isinstance(value.dtype, str):
        pass
    elif issubclass(value.dtype.type, np.string_):
        value = value.astype(str)
        # Backwards compat, old datasets have strings as one element 1d arrays
        if len(value) == 1:
            return value[0]
    if value.shape == ():
        value = value[()]
    return value


def read_neighbors(group) -> Neighbors:
    x = group['x'][...]
    n_neighbors = group.attrs['n_neighbors']
    n_pcs = int(group.attrs['n_pcs'])
    metric = group.attrs['metric']
    method = group.attrs['method']
    knn = bool(group.attrs['knn'])
    random_state = group.attrs['random_state']
    neighbor = Neighbors(
        x=x,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        method=method,
        knn=knn,
        random_state=random_state,
    )
    return neighbor


def read_key_record(group, key_record):
    for k in group.keys():
        key_record[k] = group[k][...].astype('U').tolist()


def read_group(group: h5py.Group) -> Union[dict, pd.DataFrame, sparse.spmatrix, Gene, Cell, Neighbors]:
    encoding_type = group.attrs.get('encoding-type')
    if encoding_type is None or encoding_type == 'dict':
        pass
    elif encoding_type == 'dataframe':
        return read_dataframe(group)
    elif encoding_type in {'csr_matrix', 'csc_matrix'}:
        return read_spmatrix(group)
    elif encoding_type == 'cell':
        return read_cells(group)
    elif encoding_type == 'gene':
        return read_genes(group)
    elif encoding_type == 'neighbors':
        return read_neighbors(group)
    else:
        raise ValueError(f'Unfamiliar `encoding-type`: {encoding_type}.')
    d = dict()
    for sub_key, sub_value in group.items():
        d[sub_key] = read_dataset(sub_value) if isinstance(sub_value, h5py.Dataset) else read_group(sub_value)
    return d


def read_dense_as_sparse(
        dataset: h5py.Dataset, sparse_format: sparse.spmatrix, axis_chunk: int
):
    if sparse_format == sparse.csr_matrix:
        return read_dense_as_csr(dataset, axis_chunk)
    elif sparse_format == sparse.csc_matrix:
        return read_dense_as_csc(dataset, axis_chunk)
    else:
        raise ValueError(f'Cannot read dense array as type: {sparse_format}')


def read_dense_as_csr(dataset, axis_chunk=6000):
    sub_matrices = []
    for idx in idx_chunks_along_axis(dataset.shape, 0, axis_chunk):
        dense_chunk = dataset[idx]
        sub_matrix = sparse.csr_matrix(dense_chunk)
        sub_matrices.append(sub_matrix)
    return sparse.vstack(sub_matrices, format='csr')


def read_dense_as_csc(dataset, axis_chunk=6000):
    sub_matrices = []
    for idx in idx_chunks_along_axis(dataset.shape, 1, axis_chunk):
        sub_matrix = sparse.csc_matrix(dataset[idx])
        sub_matrices.append(sub_matrix)
    return sparse.hstack(sub_matrices, format='csc')
