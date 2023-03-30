import unittest

from stereo.algorithm.single_r.single_r import SingleR
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.utils._download import _download


class TestSingleR(unittest.TestCase, SingleR):
    # TODO with no download url
    DEMO_REF_URL = ''
    DEMO_TEST_URL = ''

    def test_example_single_r(self):
        ref_file_path = _download(TestSingleR.DEMO_REF_URL, file_name='GSE84133_GSM2230761_mouse1_ref.h5ad')
        test_file_path = _download(TestSingleR.DEMO_TEST_URL, file_name='GSE84133_GSM2230762_mouse2_test.h5ad')

        ref_data = AnnBasedStereoExpData(h5ad_file_path=ref_file_path)
        ref_data._ann_data.obs['celltype'] = ref_data._ann_data.obsm['annotation_au']['celltype']

        test_data = AnnBasedStereoExpData(h5ad_file_path=test_file_path)
        test_data.tl.single_r(ref_data, ref_use_col="celltype")