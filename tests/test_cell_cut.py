import pytest
import unittest

from stereo.utils._download import _download

from settings import TEST_DATA_PATH, DEMO_CELL_SEGMENTATION_TIF_URL, DEMO_CELL_SEGMENTATION_V3_MODEL_URL, \
    TEST_IMAGE_PATH


class TestCellCut(unittest.TestCase):

    def setUp(self) -> None:
        demo_tif_path = _download(DEMO_CELL_SEGMENTATION_TIF_URL, dir_str=TEST_DATA_PATH)
        model_v3_path = _download(DEMO_CELL_SEGMENTATION_V3_MODEL_URL, dir_str=TEST_DATA_PATH)

        self.demo_tif_path = demo_tif_path
        self.model_v3_path = model_v3_path

    @pytest.mark.cell_cut_env
    def test_cell_cut(self):
        from stereo.image import cell_seg_v3
        cell_seg_v3(self.model_v3_path, self.demo_tif_path, TEST_IMAGE_PATH + "cell_cut_v3.tif")
