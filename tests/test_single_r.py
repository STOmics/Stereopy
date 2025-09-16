import unittest

from stereo.algorithm.single_r.single_r import SingleR
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, DEMO_TEST_URL, DEMO_REF_URL


class TestSingleR(unittest.TestCase, SingleR):

    def setUp(self) -> None:
        ref_file_path = _download(DEMO_REF_URL, dir_str=TEST_DATA_PATH)
        test_file_path = _download(DEMO_TEST_URL, dir_str=TEST_DATA_PATH)

        self.ref_data = AnnBasedStereoExpData(h5ad_file_path=ref_file_path)
        self.test_data = AnnBasedStereoExpData(h5ad_file_path=test_file_path)

    def test_example_single_r(self):
        self.test_data.tl.single_r(self.ref_data, ref_use_col="celltype")

    def test_example_single_r_cluster(self):
        self.test_data.tl.single_r(self.ref_data, ref_use_col="celltype", cluster_res_key='leiden')
