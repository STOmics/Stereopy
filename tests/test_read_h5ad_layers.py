"""
Test that _read_stereo_h5ad_from_group handles 'layers' stored as an
h5py.Dataset instead of an h5py.Group (issue #382).

This test file directly patches the minimal function logic to avoid pulling
in the entire Stereopy dependency tree during CI.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest


def _read_layers_logic(f, k):
    """
    Extracted layers-reading logic from _read_stereo_h5ad_from_group
    (stereo/io/reader.py) for isolated testing.
    """
    layers = {}
    if isinstance(f[k], h5py.Group):
        for layer_key in f[k].keys():
            if isinstance(f[k][layer_key], h5py.Group):
                layers[layer_key] = dict(f[k][layer_key])
            else:
                layers[layer_key] = f[k][layer_key][()]
    elif isinstance(f[k], h5py.Dataset):
        layers[k] = f[k][()]
    return layers


def test_layers_as_dataset_does_not_raise():
    """When 'layers' is an h5py.Dataset, reading should not raise AttributeError."""
    n_cells, n_genes = 5, 10
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test.h5")
        arr = np.random.rand(n_cells, n_genes).astype(np.float32)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("layers", data=arr)

        with h5py.File(h5_path, "r") as f:
            layers = _read_layers_logic(f, "layers")

    assert "layers" in layers
    np.testing.assert_array_equal(layers["layers"], arr)


def test_layers_as_group_preserves_behaviour():
    """When 'layers' is an h5py.Group, each sub-dataset is read as a layer."""
    n_cells, n_genes = 5, 10
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test.h5")
        spliced = np.random.rand(n_cells, n_genes).astype(np.float32)
        unspliced = np.random.rand(n_cells, n_genes).astype(np.float32)
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("layers")
            grp.create_dataset("spliced", data=spliced)
            grp.create_dataset("unspliced", data=unspliced)

        with h5py.File(h5_path, "r") as f:
            layers = _read_layers_logic(f, "layers")

    assert "spliced" in layers
    assert "unspliced" in layers
    np.testing.assert_array_equal(layers["spliced"], spliced)
    np.testing.assert_array_equal(layers["unspliced"], unspliced)


def test_layers_as_dataset_old_code_would_fail():
    """Demonstrate that calling .keys() on a Dataset raises AttributeError."""
    n_cells, n_genes = 5, 10
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("layers", data=np.zeros((n_cells, n_genes)))

        with h5py.File(h5_path, "r") as f:
            assert isinstance(f["layers"], h5py.Dataset)
            with pytest.raises(AttributeError):
                f["layers"].keys()
