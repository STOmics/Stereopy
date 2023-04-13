import unittest
import pytest

import stereo as st
from stereo.utils._download import _download
from settings import TEST_DATA_PATH, DEMO_DATA_URL, TEST_IMAGE_PATH


class TestSpatialHotspot(unittest.TestCase):

    def setUp(self) -> None:
        test_file_path = _download(DEMO_DATA_URL)

        self.test_data = st.io.read_gef(test_file_path)

    @pytest.mark.heavy
    def test_spatial_hotspot(self):
        self.test_data.tl.cal_qc()
        self.test_data.tl.filter_cells(
            min_gene=200, min_n_genes_by_counts=3, max_n_genes_by_counts=2500, pct_counts_mt=5, inplace=True
        )
        self.test_data.tl.raw_checkpoint()
        self.test_data.tl.normalize_total(target_sum=10000)
        self.test_data.tl.log1p()
        self.test_data.tl.highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000,
                                                res_key='highly_variable_genes')
        self.test_data.tl.scale()
        self.test_data.tl.spatial_hotspot(
            use_highly_genes=True,
            use_raw=True,
            hvg_res_key='highly_variable_genes',
            model='normal',
            n_neighbors=30,
            n_jobs=20,
            fdr_threshold=0.05,
            min_gene_threshold=10,
            res_key='spatial_hotspot',
        )
        self.test_data.plt.hotspot_local_correlations(out_path=TEST_IMAGE_PATH + 'hotspot_local_correlations.png')
        self.test_data.plt.hotspot_modules(out_path=TEST_IMAGE_PATH + 'hotspot_modules.png')
