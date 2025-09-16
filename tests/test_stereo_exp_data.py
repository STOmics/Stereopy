import unittest

import numpy as np
import pandas as pd
import stereo as st
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, TEST_IMAGE_PATH, DEMO_DATA_URL, DEMO_135_CELL_BIN_GEF_URL


class TestStereoExpData(unittest.TestCase):

    def setUp(self) -> None:
        file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        self.gef_file = file_path
        file_path = _download(DEMO_135_CELL_BIN_GEF_URL, dir_str=TEST_DATA_PATH)
        self.cgef_file = file_path

    def test_sub_by_index(self):
        data = st.io.read_gef(self.gef_file)
        data.sub_by_index(cell_index=np.array([1, 2, 3, 4]))
        self.assertEqual(len(data.cell_names), 4)

    def test_sub_by_index_pd_index(self):
        data = st.io.read_gef(self.gef_file)
        data.sub_by_index(cell_index=pd.Index([1, 2, 3, 4]))
        self.assertEqual(len(data.cell_names), 4)

    def test_sub_by_name_cell(self):
        data = st.io.read_gef(self.gef_file)
        data = data.sub_by_name(cell_name=['36936718752000', '37795712221600', '38225208951600', '40802189322600'])
        self.assertEqual(len(data.cell_names), 4)

    def test_sub_by_name_gene(self):
        data = st.io.read_gef(self.gef_file)
        data = data.sub_by_name(gene_name=['CAAA01147332.1', 'CAAA01118383.1', 'Gm47283'])
        self.assertEqual(len(data.gene_names), 3)