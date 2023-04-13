import unittest

from stereo.algorithm.single_r.single_r import SingleR
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.utils._download import _download


class TestSingleR(unittest.TestCase, SingleR):
    # TODO with no download url
    DEMO_REF_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                   'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                   'nodeId=8a80804386ed81950187315defb43bc2&code='

    DEMO_TEST_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                   'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                   'nodeId=8a80804386ed81950187315e260e3bc7&code='

    def setUp(self) -> None:
        ref_file_path = _download(TestSingleR.DEMO_REF_URL)
        test_file_path = _download(TestSingleR.DEMO_TEST_URL)

        self.ref_data = AnnBasedStereoExpData(h5ad_file_path=ref_file_path)
        self.test_data = AnnBasedStereoExpData(h5ad_file_path=test_file_path)


    def test_example_single_r(self):
        self.test_data.tl.single_r(self.ref_data, ref_use_col="celltype")

    def test_example_single_r_cluster(self):
        self.test_data.tl.single_r(self.ref_data, ref_use_col="celltype", cluster_res_key='leiden')