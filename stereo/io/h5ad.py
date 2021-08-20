#!/usr/bin/env python3
# coding: utf-8
"""
@file: h5ad.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/18  create file.
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
def _(v, f, k):
    write_dataframe(f, k, v)


@write.register(sparse.spmatrix)
def _(v, f, k, sp_format):
    write_spmatrix(f, k, v, sp_format)


@write.register(Gene)
def _(v, f, k):
    write_genes(f, k, v)


@write.register(Cell)
def _(v, f, k):
    write_cells(f, k, v)


def write_array(f, key, value, dataset_kwargs=MappingProxyType({})):
    # Convert unicode to fixed length strings
    if value.dtype.kind in {"U", "O"}:
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
    g.attrs["encoding-type"] = f"{fmt}_matrix"
    g.attrs["shape"] = v.shape
    # Allow resizing
    if "maxshape" not in dataset_kwargs:
        dataset_kwargs = dict(maxshape=(None,), **dataset_kwargs)
    g.create_dataset("data", data=v.data, **dataset_kwargs)
    g.create_dataset("indices", data=v.indices, **dataset_kwargs)
    g.create_dataset("indptr", data=v.indptr, **dataset_kwargs)


def write_genes(f, k, v, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    g.attrs["encoding-type"] = "gene"
    write_array(g, 'gene_name', v.gene_name, dataset_kwargs)


def write_cells(f, k, v, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    g.attrs["encoding-type"] = "cell"
    write_array(g, 'cell_name', v.cell_name, dataset_kwargs)


def write_spmatrix_as_dense(f, key, value, dataset_kwargs=MappingProxyType({})):
    dset = f.create_dataset(key, shape=value.shape, dtype=value.dtype, **dataset_kwargs)
    compressed_axis = int(isinstance(value, sparse.csc_matrix))
    for idx in idx_chunks_along_axis(value.shape, compressed_axis, 1000):
        dset[idx] = value[idx].toarray()


def write_dataframe(f, key, df, dataset_kwargs=MappingProxyType({})):
    # Check arguments
    for reserved in ("__categories", "_index"):
        if reserved in df.columns:
            raise ValueError(f"{reserved!r} is a reserved name for dataframe columns.")

    col_names = [check_key(c) for c in df.columns]

    if df.index.name is not None:
        index_name = df.index.name
    else:
        index_name = "_index"
    index_name = check_key(index_name)

    group = f.create_group(key)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["column-order"] = col_names
    group.attrs["_index"] = index_name

    write_series(group, index_name, df.index, dataset_kwargs=dataset_kwargs)
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
        category_key = f"__categories/{key}"

        write_array(group, category_key, categories, dataset_kwargs=dataset_kwargs)
        write_array(group, key, codes, dataset_kwargs=dataset_kwargs)

        group[key].attrs["categories"] = group[category_key].ref
        group[category_key].attrs["ordered"] = categorical.ordered
    else:
        write_array(group, key, series.values, dataset_kwargs=dataset_kwargs)


def check_key(key):
    """
    Checks that passed value is a valid h5py key.
    Should convert it if there is an obvious conversion path, error otherwise.
    """
    typ = type(key)
    if issubclass(typ, str):
        return str(key)
    else:
        raise TypeError(f"{key} of type {typ} is an invalid key. Should be str.")


def _to_hdf5_vlen_strings(value: np.ndarray) -> np.ndarray:
    """
    This corrects compound dtypes to work with hdf5 files.

    :param value:
    :return:
    """
    new_dtype = []
    for dt_name, (dt_type, _) in value.dtype.fields.items():
        if dt_type.kind in ("U", "O"):
            new_dtype.append((dt_name, h5py.special_dtype(vlen=str)))
        else:
            new_dtype.append((dt_name, dt_type))
    return value.astype(new_dtype)


def read_dataframe(group) -> pd.DataFrame:
    columns = list(group.attrs["column-order"])
    idx_key = group.attrs["_index"]
    df = pd.DataFrame(
        {k: read_series(group[k]) for k in columns},
        index=read_series(group[idx_key]),
        columns=list(columns),
    )
    if idx_key != "_index":
        df.index.name = idx_key
    return df


def read_spmatrix(group) -> sparse.spmatrix:
    shape = tuple(group.attrs["shape"])
    dtype = group["data"].dtype
    mtx = sparse.csr_matrix(shape, dtype=dtype) if group.attrs["encoding-type"] == "csr_matrix" \
        else sparse.csc_matrix(shape, dtype=dtype)
    mtx.data = group["data"][...]
    mtx.indices = group["indices"][...]
    mtx.indptr = group["indptr"][...]
    return mtx


def read_genes(group) -> Gene:
    gene_name = group["gene_name"][...]
    gene = Gene(gene_name=gene_name)
    return gene


def read_cells(group) -> Cell:
    cell_name = group["cell_name"][...]
    cell = Cell(cell_name=cell_name)
    return cell


def read_series(dataset) -> Union[np.ndarray, pd.Categorical]:
    if "categories" in dataset.attrs:
        categories = dataset.attrs["categories"]
        ordered = False
        if isinstance(categories, h5py.Reference):
            categories_dset = dataset.parent[dataset.attrs["categories"]]
            categories = read_dataset(categories_dset)
            ordered = bool(categories_dset.attrs.get("ordered", False))
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
        if (string_dtype is not None) and (string_dtype.encoding == "utf-8"):
            dataset = dataset.asstr()
    value = dataset[()]
    if not hasattr(value, "dtype"):
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


def read_group(group: h5py.Group) -> Union[dict, pd.DataFrame, sparse.spmatrix, Gene, Cell]:
    encoding_type = group.attrs.get("encoding-type")
    if encoding_type is None:
        pass
    elif encoding_type == "dataframe":
        return read_dataframe(group)
    elif encoding_type in {"csr_matrix", "csc_matrix"}:
        return read_spmatrix(group)
    elif encoding_type == "cell":
        return read_cells(group)
    elif encoding_type == "gene":
        return read_genes(group)
    else:
        raise ValueError(f"Unfamiliar `encoding-type`: {encoding_type}.")
    d = dict()
    for sub_key, sub_value in group.items():
        d[sub_key] = read_dataset(sub_value)
    return d


def read_dense_as_sparse(
    dataset: h5py.Dataset, sparse_format: sparse.spmatrix, axis_chunk: int
):
    if sparse_format == sparse.csr_matrix:
        return read_dense_as_csr(dataset, axis_chunk)
    elif sparse_format == sparse.csc_matrix:
        return read_dense_as_csc(dataset, axis_chunk)
    else:
        raise ValueError(f"Cannot read dense array as type: {sparse_format}")


def read_dense_as_csr(dataset, axis_chunk=6000):
    sub_matrices = []
    for idx in idx_chunks_along_axis(dataset.shape, 0, axis_chunk):
        dense_chunk = dataset[idx]
        sub_matrix = sparse.csr_matrix(dense_chunk)
        sub_matrices.append(sub_matrix)
    return sparse.vstack(sub_matrices, format="csr")


def read_dense_as_csc(dataset, axis_chunk=6000):
    sub_matrices = []
    for idx in idx_chunks_along_axis(dataset.shape, 1, axis_chunk):
        sub_matrix = sparse.csc_matrix(dataset[idx])
        sub_matrices.append(sub_matrix)
    return sparse.hstack(sub_matrices, format="csc")
