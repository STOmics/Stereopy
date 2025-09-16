import math
import unittest

import stereo as st
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, DEMO_DATA_URL


class TestH5adFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.data = st.io.read_gef(_download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH))

    def test_stereo_to_ann(self):
        self.data.tl.cal_qc()
        self.data.tl.raw_checkpoint()
        self.data.tl.normalize_total(target_sum=1e4)
        self.data.tl.log1p()
        self.data.tl.highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes',
                                           n_top_genes=None)
        self.data.tl.scale(zero_center=False)
        self.data.tl.pca(use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca', svd_solver='arpack')
        self.data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)
        self.data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap', init_pos='spectral')
        self.data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
        self.data.tl.louvain(neighbors_res_key='neighbors', res_key='louvain')
        self.data.tl.phenograph()
        self.data.tl.find_marker_genes(cluster_res_key='leiden')

        ann_data_1 = st.io.stereo_to_anndata(self.data, flavor='scanpy')
        self.assertEqual(ann_data_1.X.shape, self.data.exp_matrix.shape)
        # TODO sometimes raising exception
        # self.assertEqual(math.ceil(ann_data_1.X.sum()), self.data.exp_matrix.sum())
        self.assertEqual(ann_data_1.raw.X.shape, self.data.tl.raw.exp_matrix.shape)
        self.assertEqual(math.ceil(ann_data_1.raw.X.sum()), self.data.tl.raw.exp_matrix.sum())
        self.assertIn('hvg', ann_data_1.uns)
        for i in self.data.tl.result['highly_variable_genes']:
            if i == 'mean_bin':
                self.assertNotIn(i, ann_data_1.var.columns)
            else:
                self.assertIn(i, ann_data_1.var.columns)
        for x_key in {'X_pca', 'X_umap'}:
            self.assertIn(x_key, ann_data_1.obsm)
        for res_key in self.data.tl.key_record['neighbors']:
            sc_con = 'connectivities' if res_key == 'neighbors' else f'{res_key}_connectivities'
            sc_dis = 'distances' if res_key == 'neighbors' else f'{res_key}_distances'
            self.assertIn(sc_con, ann_data_1.obsp)
            self.assertIn(sc_dis, ann_data_1.obsp)
            self.assertIn(res_key, ann_data_1.uns)
        for res_key in self.data.tl.key_record['cluster']:
            self.assertIn(res_key, ann_data_1.obs.columns)

    def test_stereo_to_ann_sct(self):
        self.data.tl.cal_qc()
        self.data.tl.raw_checkpoint()
        self.data.tl.sctransform()

        ann_data_1 = st.io.stereo_to_anndata(self.data, flavor='scanpy')

        self.assertIn('sct_counts', ann_data_1.uns)
        self.assertIn('sct_data', ann_data_1.uns)
        self.assertIn('sct_scale', ann_data_1.uns)
        self.assertIn('sct_top_features', ann_data_1.uns)
        self.assertIn('sct_cellname', ann_data_1.uns)
        self.assertIn('sct_genename', ann_data_1.uns)
