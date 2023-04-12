import unittest

import stereo as st
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, TEST_IMAGE_PATH, DEMO_DATA_URL


class TestClustering(unittest.TestCase):

    def setUp(self) -> None:
        file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        self.gef_file = file_path

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
        data.plt.highly_variable_genes(res_key='highly_variable_genes',
                                       out_path=TEST_IMAGE_PATH + "highly_variable_genes.png")
        data.tl.scale(zero_center=False)
        data.tl.pca(
            use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca', svd_solver='arpack'
        )
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)
        data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap', init_pos='spectral')
        data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
        data.plt.umap(res_key='umap', cluster_key='leiden', out_path=TEST_IMAGE_PATH + "umap.png")
        data.plt.cluster_scatter(res_key='leiden', out_path=TEST_IMAGE_PATH + "leiden.png")

    def test_clustering_gpu(self):
        try:
            import cudf
            import cugraph
            from cuml import UMAP
            from cuml.neighbors.nearest_neighbors import NearestNeighbors
        except ImportError as e:
            st.logger.info(f'this is not a GPU environment, got expection: {str(e)}')
            return

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
