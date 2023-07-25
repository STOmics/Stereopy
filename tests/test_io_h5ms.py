import copy
import pytest
import unittest

from stereo.core.ms_data import MSData
from stereo.io import read_gef
from stereo.io.reader import read_h5ms
from stereo.io.writer import write_h5ms
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, DEMO_DATA_URL


class TestIOH5ms(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp()

        self.file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)

        self.ms_data = MSData()
        self.ms_data += read_gef(self.file_path)
        self.ms_data += copy.deepcopy(self.ms_data[0])

        from stereo.core.ms_pipeline import slice_generator as sg
        self.ms_data.integrate()
        self.ms_data.tl.cal_qc(scope=sg[:], mode="integrate")
        self.ms_data.tl.filter_cells(scope=sg[:], mode="integrate", min_gene=200, min_n_genes_by_counts=3,
                                max_n_genes_by_counts=7000, pct_counts_mt=8, inplace=False)
        self.ms_data.tl.log1p(scope=sg[:], mode="integrate")
        self.ms_data.tl.normalize_total(scope=sg[:], mode="integrate", target_sum=1e4)
        self.ms_data.tl.pca(scope=sg[:1], mode="integrate", use_highly_genes=False, hvg_res_key='highly_variable_genes',
                       n_pcs=20, res_key='pca', svd_solver='arpack')
        self.ms_data.tl.neighbors(scope=sg[0:1], mode="integrate", pca_res_key='pca', n_pcs=30, res_key='neighbors',
                             n_jobs=8)


    def test_write_and_read(self):
        write_h5ms(self.ms_data, TEST_DATA_PATH + "SS200000135TL_D1.h5ms")
        ms_data = read_h5ms(TEST_DATA_PATH + "SS200000135TL_D1.h5ms")
        print(ms_data)
