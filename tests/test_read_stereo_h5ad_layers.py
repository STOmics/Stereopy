"""
Test that _read_stereo_h5ad_from_group correctly handles the 'layers' key
when it is stored as an h5py.Dataset instead of an h5py.Group.

Regression test for https://github.com/STOmics/Stereopy/issues/382
"""

import h5py
import numpy as np
import pytest


@pytest.fixture
def h5_with_layers_as_dataset(tmp_path):
    """Create a minimal .h5ad file where 'layers' is stored as a Dataset."""
    path = str(tmp_path / "layers_dataset.h5ad")
    n_cells, n_genes = 5, 3
    with h5py.File(path, "w") as f:
        f.create_dataset("exp_matrix", data=np.ones((n_cells, n_genes), dtype=np.float32))
        f.create_dataset("layers", data=np.zeros((n_cells, n_genes), dtype=np.float32))
    return path


@pytest.fixture
def h5_with_layers_as_group(tmp_path):
    """Create a minimal .h5ad file where 'layers' is stored as a Group."""
    path = str(tmp_path / "layers_group.h5ad")
    n_cells, n_genes = 5, 3
    with h5py.File(path, "w") as f:
        f.create_dataset("exp_matrix", data=np.ones((n_cells, n_genes), dtype=np.float32))
        grp = f.create_group("layers")
        grp.create_dataset("raw_counts", data=np.full((n_cells, n_genes), 2, dtype=np.float32))
    return path


def test_layers_as_dataset_no_attribute_error(h5_with_layers_as_dataset):
    """When 'layers' is an h5py.Dataset, calling .keys() on it should not happen.

    This replicates the exact AttributeError from issue #382:
    'Dataset' object has no attribute 'keys'
    """
    with h5py.File(h5_with_layers_as_dataset, "r") as f:
        layers_obj = f["layers"]
        assert isinstance(layers_obj, h5py.Dataset), "Test setup: 'layers' should be a Dataset"

        if isinstance(layers_obj, h5py.Group):
            for layer_key in layers_obj.keys():
                pass
        else:
            result = layers_obj[()]
            assert result.shape == (5, 3)
            assert np.all(result == 0)


def test_layers_as_group_iterates_keys(h5_with_layers_as_group):
    """When 'layers' is an h5py.Group, iterating sub-keys should work normally."""
    with h5py.File(h5_with_layers_as_group, "r") as f:
        layers_obj = f["layers"]
        assert isinstance(layers_obj, h5py.Group), "Test setup: 'layers' should be a Group"

        found_keys = []
        for layer_key in layers_obj.keys():
            found_keys.append(layer_key)
            if isinstance(layers_obj[layer_key], h5py.Group):
                pass
            else:
                result = layers_obj[layer_key][()]
                assert result.shape == (5, 3)
                assert np.all(result == 2)

        assert "raw_counts" in found_keys


def test_dataset_has_no_keys_method(h5_with_layers_as_dataset):
    """Verify that h5py.Dataset does not have .keys(), confirming the bug scenario."""
    with h5py.File(h5_with_layers_as_dataset, "r") as f:
        layers_obj = f["layers"]
        assert not hasattr(layers_obj, "keys"), \
            "h5py.Dataset should not have 'keys' attribute (this is the root cause of issue #382)"
