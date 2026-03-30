"""Test for issue #384: KeyError 'group' in stereo_to_anndata cluster handling."""
import pytest
import pandas as pd
import numpy as np


class FakeGenes:
    def __init__(self, gene_names):
        self._var = pd.DataFrame(index=gene_names)
        self._matrix = {}
        self._pairwise = {}

    def to_df(self):
        return self._var.copy()


class FakeCells:
    def __init__(self, cell_names):
        self.cell_name = np.array(cell_names)
        self._obs = pd.DataFrame(index=cell_names)
        self._matrix = {}
        self._pairwise = {}
        self.cell_border = None


class FakeResult(dict):
    pass


class FakeKeyRecord(dict):
    pass


class FakeTl:
    def __init__(self):
        self.result = FakeResult()
        self.key_record = FakeKeyRecord()


class FakeData:
    def __init__(self, cells, genes):
        self.cells = cells
        self.genes = genes
        self.tl = FakeTl()
        self.position = None
        self.position_z = None
        self.bin_type = None
        self.bin_size = None
        self.attr = None
        self.sn = None
        self.layers = {}
        self.merged = False
        self.position_offset = None
        self.position_min = None
        self.exp_matrix = np.zeros((len(cells.cell_name), len(genes._var)))
        self.cell_names = cells.cell_name


def test_cluster_result_with_group_column():
    """Normal case: cluster result is DataFrame with 'group' column."""
    cells = FakeCells(['cell_1', 'cell_2', 'cell_3'])
    genes = FakeGenes(['gene_1', 'gene_2'])
    data = FakeData(cells, genes)

    cluster_df = pd.DataFrame({
        'bins': ['cell_1', 'cell_2', 'cell_3'],
        'group': ['A', 'B', 'A']
    })
    data.tl.result['leiden'] = cluster_df
    data.tl.key_record['cluster'] = ['leiden']

    res = data.tl.result['leiden']
    assert isinstance(res, pd.DataFrame)
    assert 'group' in res.columns


def test_cluster_result_dict_without_group():
    """Bug case: cluster result is dict without 'group' key (e.g., metadata dict)."""
    cells = FakeCells(['cell_1', 'cell_2', 'cell_3'])
    genes = FakeGenes(['gene_1', 'gene_2'])
    data = FakeData(cells, genes)

    metadata = {'params': {}, 'source': 'stereopy', 'method': 'leiden'}
    data.tl.result['leiden'] = metadata
    data.tl.key_record['cluster'] = ['leiden']

    res = data.tl.result['leiden']
    assert isinstance(res, dict)
    assert 'group' not in res


def test_cluster_result_dict_with_group():
    """Edge case: cluster result is dict with 'group' key."""
    cells = FakeCells(['cell_1', 'cell_2', 'cell_3'])
    genes = FakeGenes(['gene_1', 'gene_2'])
    data = FakeData(cells, genes)

    result_dict = {
        'bins': np.array(['cell_1', 'cell_2', 'cell_3']),
        'group': np.array(['A', 'B', 'A'])
    }
    data.tl.result['leiden'] = result_dict
    data.tl.key_record['cluster'] = ['leiden']

    res = data.tl.result['leiden']
    assert isinstance(res, dict)
    assert 'group' in res
