"""Tests for H5AD flavor auto-detection in read_h5ad."""
import os
import tempfile
import pytest
import h5py
import numpy as np


def _create_stereopy_h5ad(path):
    """Create a minimal Stereopy-format H5AD file."""
    with h5py.File(path, 'w') as f:
        f.attrs['bin_type'] = 'bins'
        f.attrs['bin_size'] = 100
        f.attrs['merged'] = False

        g = f.create_group('cells')
        g.attrs['encoding-type'] = 'cell'
        g.attrs['version'] = 'v2'
        obs_g = g.create_group('obs')
        obs_g.attrs['encoding-type'] = 'dataframe'
        obs_g.attrs['column-order'] = []
        obs_g.attrs['_index'] = '_index'
        obs_g.attrs['save-as-matrix'] = False
        obs_g.create_dataset('_index', data=np.array(['cell_0', 'cell_1'], dtype='S'))

        g2 = f.create_group('genes')
        g2.attrs['encoding-type'] = 'gene'
        g2.attrs['version'] = 'v2'
        var_g = g2.create_group('var')
        var_g.attrs['encoding-type'] = 'dataframe'
        var_g.attrs['column-order'] = []
        var_g.attrs['_index'] = '_index'
        var_g.attrs['save-as-matrix'] = False
        var_g.create_dataset('_index', data=np.array(['gene_0', 'gene_1'], dtype='S'))

        f.create_dataset('exp_matrix', data=np.array([[1, 2], [3, 4]]))
        f.create_dataset('position', data=np.array([[0, 0], [1, 1]], dtype=np.uint32))


def _create_scanpy_h5ad(path):
    """Create a minimal standard AnnData (scanpy-format) H5AD file."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('X', data=np.array([[1.0, 2.0], [3.0, 4.0]]))

        obs = f.create_group('obs')
        obs.attrs['encoding-type'] = 'dataframe'
        obs.attrs['column-order'] = []
        obs.attrs['_index'] = '_index'
        obs.create_dataset('_index', data=np.array(['cell_0', 'cell_1'], dtype='S'))

        var = f.create_group('var')
        var.attrs['encoding-type'] = 'dataframe'
        var.attrs['column-order'] = []
        var.attrs['_index'] = '_index'
        var.create_dataset('_index', data=np.array(['gene_0', 'gene_1'], dtype='S'))


class TestDetectH5adFlavor:

    def test_detect_stereopy_format(self, tmp_path):
        from stereo.io.reader import _detect_h5ad_flavor
        path = str(tmp_path / 'test_stereopy.h5ad')
        _create_stereopy_h5ad(path)
        assert _detect_h5ad_flavor(path) == 'stereopy'

    def test_detect_scanpy_format(self, tmp_path):
        from stereo.io.reader import _detect_h5ad_flavor
        path = str(tmp_path / 'test_scanpy.h5ad')
        _create_scanpy_h5ad(path)
        assert _detect_h5ad_flavor(path) == 'scanpy'

    def test_detect_nonexistent_file(self):
        from stereo.io.reader import _detect_h5ad_flavor
        assert _detect_h5ad_flavor('/nonexistent/file.h5ad') is None

    def test_detect_empty_file(self, tmp_path):
        from stereo.io.reader import _detect_h5ad_flavor
        path = str(tmp_path / 'empty.h5ad')
        with h5py.File(path, 'w') as f:
            pass
        assert _detect_h5ad_flavor(path) is None
