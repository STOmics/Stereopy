import pytest
import unittest

import stereo as st
from stereo.utils._download import _download
from settings import TEST_DATA_PATH, DEMO_DATA_URL, DEMO_TFS_URL, DEMO_DATABASE_URL, DEMO_MOTIF_URL, TEST_IMAGE_PATH


class TestRegulatoryNetworkInference(unittest.TestCase):

    def setUp(self) -> None:
        self.tfs_fn = _download(DEMO_TFS_URL, dir_str=TEST_DATA_PATH)
        self.database_fn = _download(DEMO_DATABASE_URL, dir_str=TEST_DATA_PATH)
        self.motif_anno_fn = _download(DEMO_MOTIF_URL, dir_str=TEST_DATA_PATH)

        test_file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)

        self.test_data = st.io.read_gef(test_file_path, bin_size=600)

    @pytest.mark.heavy
    def test_example_regulatory_network_inference(self):
        self.test_data.tl.filter_cells(
            min_gene=20, min_n_genes_by_counts=3, pct_counts_mt=5, inplace=True
        )
        self.test_data.tl.raw_checkpoint()
        from multiprocessing import cpu_count
        self.test_data.tl.regulatory_network_inference(self.database_fn, self.motif_anno_fn, self.tfs_fn, save=True,
                                                       num_workers=int(cpu_count() / 2))

        # test auc_heatmap() method.  This method requires the use of the `test_data` object.  It also requires the use of the `test_data` object
        self.test_data.plt.auc_heatmap(network_res_key='regulatory_network_inference', width=28, height=28,
                                       out_path=TEST_IMAGE_PATH + "auc_heatmap.png")

        # test spatial_scatter_by_regulon() method.  This method requires the use of the `test_data` object. It also requires the use of the `test_data` object
        # self.test_data.plt.spatial_scatter_by_regulon(reg_name='Thra(+)', network_res_key='regulatory_network_inference', dot_size=2, out_path=TEST_IMAGE_PATH + "spatial_scatter_by_regulon.png")

        # normalization
        self.test_data.tl.normalize_total(target_sum=10000)
        self.test_data.tl.log1p()
        self.test_data.tl.scale(max_value=10, zero_center=True)

        # pca, neighbors, leiden, and pcoa methods require the `test_data` object.  They also require the `test_data` object.  They are tested independently
        self.test_data.tl.pca(use_highly_genes=False, n_pcs=30, res_key='pca')
        self.test_data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors')
        self.test_data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')

        # test grn_dotplot() method.  This method requires the use of the `test_data` object.  It also requires the use of the `test_data` object
        self.test_data.plt.grn_dotplot(self.test_data.tl.result['leiden'],
                                       network_res_key='regulatory_network_inference',
                                       out_path=TEST_IMAGE_PATH + "grn_dotplot.png")

        # test auc_heatmap_by_group() method.  This method requires the use of the `test_data` object.  It also requires the use of the `test_data` object
        self.test_data.plt.auc_heatmap_by_group(network_res_key='regulatory_network_inference',
                                                celltype_res_key='leiden', top_n_feature=5,
                                                out_path=TEST_IMAGE_PATH + "auc_heatmap_by_group.png")
