import unittest

import stereo as st
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, TEST_IMAGE_PATH, DEMO_DATA_URL


class TestMarkerGenes(unittest.TestCase):
    data = None
    gef_file = None

    @classmethod
    def setUpClass(cls) -> None:
        file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        cls.gef_file = file_path
        cls.data = st.io.read_gef(cls.gef_file)
        cls.data.tl.cal_qc()
        cls.data.tl.raw_checkpoint()
        cls.data.tl.normalize_total(target_sum=1e4)
        cls.data.tl.log1p()
        cls.data.tl.highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes',
                                          n_top_genes=None)
        cls.data.tl.scale()
        cls.data.tl.pca(use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca',
                        svd_solver='arpack')
        cls.data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=8)
        cls.data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')

    def setUp(self) -> None:
        self.gef_file = self.__class__.gef_file
        self.data = self.__class__.data

    def test_find_marker_genes_default(self):
        """
        Default Parameters
        ------------------

        cluster_res_key,
        method: str = 't_test', # t_test or wilcoxon_test
        case_groups: Union[str, np.ndarray, list] = 'all',
        control_groups: Union[str, np.ndarray, list] = 'rest',
        corr_method: str = 'benjamini-hochberg', # 'bonferroni', 'benjamini-hochberg'
        use_raw: bool = True,
        use_highly_genes: bool = True,
        hvg_res_key: Optional[str] = 'highly_variable_genes',
        res_key: str = 'marker_genes',
        output: Optional[str] = None,
        sort_by='scores',
        n_genes: Union[str, int] = 'all'
        """
        self.data.tl.find_marker_genes(cluster_res_key='leiden')

    def test_find_marker_genes_method_wilcoxon_test(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', method="wilcoxon_test")

    def test_find_marker_genes_not_use_raw(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', use_raw=False)

    def test_find_marker_genes_corr_method_bonferroni(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', corr_method="bonferroni")

    def test_find_marker_genes_sort_by_log2fc(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', sort_by="log2fc")

    def test_find_marker_genes_n_genes_none(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', n_genes='auto')

    def test_find_marker_genes_n_genes_num(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', n_genes=23)

    def test_find_marker_genes_sort_by_log2fc_n_genes_none(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', sort_by="log2fc", n_genes='auto')

    def test_find_marker_genes_method_wilcoxon_test_n_genes_none(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', method="wilcoxon_test", n_genes='auto')

    def test_find_marker_genes_method_wilcoxon_test_sort_by_log2fc_n_genes_none(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', method="wilcoxon_test", sort_by="log2fc",
                                       n_genes='auto')

    def test_find_marker_genes_output(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', output=TEST_DATA_PATH + "/marker_genes1.csv")
        self.data.plt.marker_genes_scatter(res_key='marker_genes', markers_num=5,
                                           out_path=TEST_IMAGE_PATH + "marker_genes_scatter.png")

    def test_find_marker_genes_sort_by_log2fc_output(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', sort_by="log2fc",
                                       output=TEST_DATA_PATH + "/marker_genes2.csv")
        self.data.plt.marker_gene_volcano(group_name='2.vs.rest', vlines=False,
                                          out_path=TEST_IMAGE_PATH + "marker_gene_volcano.png")

    def test_find_marker_genes_sort_by_log2fc_n_genes_none_output(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', sort_by="log2fc", n_genes='auto',
                                       output=TEST_DATA_PATH + "/marker_genes3.csv")
        self.data.plt.marker_genes_text(res_key='marker_genes', markers_num=10, sort_key='scores',
                                        out_path=TEST_IMAGE_PATH + "marker_genes.png")

    def test_filter_marker_genes_sort_by_log2fc_n_genes_none_output(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', sort_by="log2fc", n_genes='auto',
                                       output=TEST_DATA_PATH + "/marker_genes4.csv")
        self.data.tl.filter_marker_genes(min_fold_change=1, min_in_group_fraction=0.001, max_out_group_fraction=0.1,
                                         output=TEST_DATA_PATH + "filter_genes4.csv")
        self.data.plt.marker_genes_text(res_key='marker_genes', markers_num=10, sort_key='scores',
                                        out_path=TEST_IMAGE_PATH + "marker_genes_filtered.png")

    def test_filter_plot_scatter(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', sort_by="log2fc", n_genes='auto',
                                       output=TEST_DATA_PATH + "/marker_genes4.csv")
        self.data.tl.filter_marker_genes(min_fold_change=1, min_in_group_fraction=0.001, max_out_group_fraction=0.1,
                                         output=TEST_DATA_PATH + "filter_genes4.csv")
        self.data.plt.marker_genes_scatter(res_key='marker_genes', markers_num=5,
                                           out_path=TEST_IMAGE_PATH + "marker_genes_scatter_filtered.png")

    def test_filter_plot_volcano(self):
        self.data.tl.find_marker_genes(cluster_res_key='leiden', sort_by="log2fc", n_genes='auto',
                                       output=TEST_DATA_PATH + "marker_genes4.csv")
        self.data.tl.filter_marker_genes(min_fold_change=1, min_in_group_fraction=0.001, max_out_group_fraction=0.1,
                                         output=TEST_DATA_PATH + "filter_genes4.csv")
        self.data.plt.marker_gene_volcano(group_name='2.vs.rest', vlines=False,
                                          out_path=TEST_IMAGE_PATH + "marker_gene_volcano_filtered.png")
