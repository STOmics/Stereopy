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
from typing import Literal
from types import MappingProxyType
from pandas.api.types import is_categorical_dtype
from scipy import sparse


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


def write_spmatrix(f, k, v, fmt: Literal["csr", "csc"], dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    g.attrs["encoding-type"] = f"{fmt}_matrix"
    g.attrs["shape"] = v.shape
    # Allow resizing
    if "maxshape" not in dataset_kwargs:
        dataset_kwargs = dict(maxshape=(None,), **dataset_kwargs)
    g.create_dataset("data", data=v.data, **dataset_kwargs)
    g.create_dataset("indices", data=v.indices, **dataset_kwargs)
    g.create_dataset("indptr", data=v.indptr, **dataset_kwargs)


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


def idx_chunks_along_axis(shape: tuple, axis: int, chunk_size: int):
    """
    Gives indexer tuples chunked along an axis.

    :param shape: Shape of array to be chunked
    :param axis: Axis to chunk along
    :param chunk_size: Size of chunk along axis
    :return: An iterator of tuples for indexing into an array of passed shape.
    """
    total = shape[axis]
    cur = 0
    mutable_idx = [slice(None) for i in range(len(shape))]
    while cur + chunk_size < total:
        mutable_idx[axis] = slice(cur, cur + chunk_size)
        yield tuple(mutable_idx)
        cur += chunk_size
    mutable_idx[axis] = slice(cur, None)
    yield tuple(mutable_idx)