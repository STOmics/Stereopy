"""Tests for reading 'layers' key from Stereopy-format H5AD files.

Verifies that both h5py.Group and h5py.Dataset formats for the 'layers'
key are handled correctly in _read_stereo_h5ad_from_group (issue #382).
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from stereo.io.reader import _read_stereo_h5ad_from_group
from stereo.core.stereo_exp_data import StereoExpData


def _make_minimal_h5(path, layers_as_dataset=False):
    """Create a minimal Stereopy-format H5AD file with a 'layers' key.

    Parameters
    ----------
    path
        Path to write the HDF5 file.
    layers_as_dataset
        If True, store 'layers' as an h5py.Dataset (flat array).
        If False, store 'layers' as an h5py.Group with a sub-dataset.
    """
    n_cells, n_genes = 5, 10
    with h5py.File(path, 'w') as f:
        f.create_dataset('exp_matrix', data=np.ones((n_cells, n_genes), dtype=np.float32))
        cells_grp = f.create_group('cells')
        cells_grp.create_dataset('cell_name', data=np.array([b'c0', b'c1', b'c2', b'c3', b'c4']))
        genes_grp = f.create_group('genes')
        genes_grp.create_dataset('gene_name', data=np.array([
            b'g0', b'g1', b'g2', b'g3', b'g4', b'g5', b'g6', b'g7', b'g8', b'g9'
        ]))
        if layers_as_dataset:
            f.create_dataset('layers', data=np.zeros((n_cells, n_genes), dtype=np.float32))
        else:
            layers_grp = f.create_group('layers')
            layers_grp.create_dataset('raw_counts', data=np.full((n_cells, n_genes), 2.0, dtype=np.float32))


@pytest.fixture
def h5_with_layers_group(tmp_path):
    path = str(tmp_path / "layers_group.h5ad")
    _make_minimal_h5(path, layers_as_dataset=False)
    return path


@pytest.fixture
def h5_with_layers_dataset(tmp_path):
    path = str(tmp_path / "layers_dataset.h5ad")
    _make_minimal_h5(path, layers_as_dataset=True)
    return path


class TestReadH5adLayers:
    """Test that _read_stereo_h5ad_from_group handles layers as Group or Dataset."""

    def test_layers_as_group(self, h5_with_layers_group):
        """When 'layers' is an h5py.Group, each sub-key should be read into data.layers."""
        data = StereoExpData(file_path=h5_with_layers_group)
        with h5py.File(h5_with_layers_group, 'r') as f:
            _read_stereo_h5ad_from_group(f, data, use_raw=False, use_result=False)
        assert 'raw_counts' in data.layers
        assert data.layers['raw_counts'].shape == (5, 10)
        np.testing.assert_array_equal(data.layers['raw_counts'], np.full((5, 10), 2.0))

    def test_layers_as_dataset(self, h5_with_layers_dataset):
        """When 'layers' is an h5py.Dataset, it should be read directly without error."""
        data = StereoExpData(file_path=h5_with_layers_dataset)
        with h5py.File(h5_with_layers_dataset, 'r') as f:
            _read_stereo_h5ad_from_group(f, data, use_raw=False, use_result=False)
        assert 'layers' in data.layers
        assert data.layers['layers'].shape == (5, 10)
        np.testing.assert_array_equal(data.layers['layers'], np.zeros((5, 10)))

    def test_layers_as_dataset_no_attribute_error(self, h5_with_layers_dataset):
        """Regression test: reading a Dataset-type 'layers' must not raise AttributeError."""
        data = StereoExpData(file_path=h5_with_layers_dataset)
        with h5py.File(h5_with_layers_dataset, 'r') as f:
            try:
                _read_stereo_h5ad_from_group(f, data, use_raw=False, use_result=False)
            except AttributeError as e:
                pytest.fail(
                    "AttributeError raised when reading layers as Dataset: {}".format(e)
                )
