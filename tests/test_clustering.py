import unittest
import stereo as st
from stereo.utils._download import _download


class TestClustering(unittest.TestCase):
    DEMO_DATA_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804a837dc46f018382c40ca51af0&code='

    def setUp(self) -> None:
        file_path = _download(TestClustering.DEMO_DATA_URL)
        self.gef_file = file_path

    def test_main(self):
        data = st.io.read_gef(self.gef_file)
        data.tl.cal_qc()
        data.plt.violin(out_path="./image_path/violin.png")
        data.plt.spatial_scatter(out_path="./image_path/spatial_scatter.png")
        data.plt.genes_count(out_path="./image_path/genes_count.png")
        data.tl.raw_checkpoint()
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes', n_top_genes=None
        )
        data.plt.highly_variable_genes(res_key='highly_variable_genes', out_path="./image_path/highly_variable_genes.png")
        data.tl.scale(zero_center=False)
        data.tl.pca(
            use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca', svd_solver='arpack'
        )
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)
        data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap', init_pos='spectral')
        data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
        data.plt.umap(res_key='umap', cluster_key='leiden', out_path="./image_path/umap.png")
        data.plt.cluster_scatter(res_key='leiden', out_path="./image_path/leiden.png")
