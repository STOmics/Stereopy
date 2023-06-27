import pytest
import unittest

import stereo as st
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, TEST_IMAGE_PATH, DEMO_DATA_URL, DEMO_135_CELL_BIN_GEF_URL


class TestClustering(unittest.TestCase):

    def setUp(self) -> None:
        file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        self.gef_file = file_path
        file_path = _download(DEMO_135_CELL_BIN_GEF_URL, dir_str=TEST_DATA_PATH)
        self.cgef_file = file_path

    def test_main(self):
        data = st.io.read_gef(self.gef_file)
        data.tl.cal_qc()
        data.plt.violin(out_path=TEST_IMAGE_PATH + "violin.png")
        data.plt.spatial_scatter(out_path=TEST_IMAGE_PATH + "spatial_scatter.png")
        data.plt.genes_count(out_path=TEST_IMAGE_PATH + "genes_count.png")
        data.tl.raw_checkpoint()
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes', n_top_genes=None
        )
        data.plt.highly_variable_genes(res_key='highly_variable_genes', out_path=TEST_IMAGE_PATH + "highly_variable_genes.png")
        data.tl.scale(zero_center=False)
        data.tl.pca(
            use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca', svd_solver='arpack'
        )
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)
        data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap', init_pos='spectral')
        data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
        data.plt.umap(res_key='umap', cluster_key='leiden', out_path=TEST_IMAGE_PATH + "umap.png")
        data.plt.cluster_scatter(res_key='leiden', out_path=TEST_IMAGE_PATH + "leiden.png")
    
    def test_cellbins_main(self):
        data = st.io.read_gef(self.gef_file)
        data.tl.cal_qc()
        data.plt.violin(out_path=TEST_IMAGE_PATH + "violin.png")
        data.plt.spatial_scatter(out_path=TEST_IMAGE_PATH + "spatial_scatter.png")
        data.plt.genes_count(out_path=TEST_IMAGE_PATH + "genes_count.png")
        data.tl.filter_cells(
            min_gene=200,
            max_gene=2000,
            min_n_genes_by_counts=3,
            max_n_genes_by_counts=1000,
            pct_counts_mt=5,
            cell_list=['31692563683562', '31546534795506', '31804232833272'],
            inplace=False
        )
        data.tl.filter_genes(
            min_cell=100,
            max_cell=1000,
            mean_umi_gt=10,
            gene_list=['Gm1992', 'Xkr4', 'Gm37381'],
            inplace=False
        )
        data.tl.raw_checkpoint()
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes', n_top_genes=None
        )
        data.plt.highly_variable_genes(res_key='highly_variable_genes', out_path=TEST_IMAGE_PATH + "cellbins_highly_variable_genes.png")
        data.tl.scale(zero_center=False)
        data.tl.pca(
            use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca', svd_solver='arpack'
        )
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)
        data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap', init_pos='spectral')
        data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
        data.plt.umap(res_key='umap', cluster_key='leiden', out_path=TEST_IMAGE_PATH + "cellbins_umap.png")
        data.plt.cluster_scatter(res_key='leiden', out_path=TEST_IMAGE_PATH + "cellbins_leiden.png")

    @pytest.mark.gpu
    def test_clustering_gpu(self):
        data = st.io.read_gef(self.gef_file, bin_size=50)
        data.tl.cal_qc()
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes',
                                      n_top_genes=None)
        data.tl.pca(use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca',
                    svd_solver='arpack')
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=8, method='rapids')
        data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap', init_pos='spectral',
                     method='rapids')
        data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden', method='rapids')
        data.tl.louvain(neighbors_res_key='neighbors', res_key='louvain', flavor='rapids', use_weights=True)

    def test_self_key(self):
        data = st.io.read_gef(self.gef_file)
        data.tl.cal_qc()
        data.tl.raw_checkpoint()
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='self_highly_variable_genes', n_top_genes=None
        )
        data.plt.highly_variable_genes(res_key='self_highly_variable_genes', out_path=TEST_IMAGE_PATH + "self_highly_variable_genes.png")
        data.tl.scale(zero_center=False)
        data.tl.pca(
            use_highly_genes=True, hvg_res_key='self_highly_variable_genes', n_pcs=20, res_key='self_pca', svd_solver='arpack'
        )
        data.tl.neighbors(pca_res_key='self_pca', n_pcs=30, res_key='self_neighbors', n_jobs=2)
        data.tl.umap(pca_res_key='self_pca', neighbors_res_key='self_neighbors', res_key='self_umap', init_pos='spectral')
        data.tl.leiden(neighbors_res_key='self_neighbors', res_key='self_leiden')
        data.plt.umap(res_key='self_umap', cluster_key='self_leiden', out_path=TEST_IMAGE_PATH + "self_umap.png")
        data.plt.cluster_scatter(res_key='self_leiden', out_path=TEST_IMAGE_PATH + "self_leiden.png")