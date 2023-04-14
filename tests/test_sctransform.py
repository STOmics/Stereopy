import unittest

from stereo.io import read_gef, write_h5ad, read_stereo_h5ad, stereo_to_anndata
from stereo.utils._download import _download

from settings import DEMO_DATA_URL, TEST_DATA_PATH


class TestSCTransform(unittest.TestCase):

    def setUp(self) -> None:
        file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        self.DATA = read_gef(file_path)

    def test_scTransform(self):
        self.DATA.tl.sctransform()

    def test_scTransform_result_write(self):
        self.DATA.tl.raw_checkpoint()
        self.DATA.tl.sctransform(do_correct_umi=True, filter_hvgs=False)
        write_h5ad(self.DATA, output=TEST_DATA_PATH + "135.sct.h5ad")
        data = read_stereo_h5ad(TEST_DATA_PATH + "135.sct.h5ad")
        data1 = stereo_to_anndata(self.DATA)

    def test_scTransform_result_write_1(self):
        self.DATA.tl.raw_checkpoint()
        self.DATA.tl.sctransform(do_correct_umi=False, filter_hvgs=True)
        write_h5ad(self.DATA, output=TEST_DATA_PATH + "135.sct.h5ad")
        data = read_stereo_h5ad(TEST_DATA_PATH + "135.sct.h5ad")
        data1 = stereo_to_anndata(self.DATA)

    def test_scTransform_result_write_2(self):
        self.DATA.tl.raw_checkpoint()
        self.DATA.tl.sctransform(do_correct_umi=True, filter_hvgs=True)
        write_h5ad(self.DATA, output=TEST_DATA_PATH + "135.sct.h5ad")
        data = read_stereo_h5ad(TEST_DATA_PATH + "135.sct.h5ad")
        data1 = stereo_to_anndata(self.DATA)

    def test_scTransform_result_write_3(self):
        self.DATA.tl.raw_checkpoint()
        self.DATA.tl.sctransform(do_correct_umi=True, filter_hvgs=True)
        write_h5ad(self.DATA, output=TEST_DATA_PATH + "135.sct.h5ad")
        data = read_stereo_h5ad(TEST_DATA_PATH + "135.sct.h5ad")
        data1 = stereo_to_anndata(self.DATA)