import unittest

from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.io.reader import read_gef
from stereo.utils._download import _download
from settings import TEST_DATA_PATH, DEMO_DATA_URL, DEMO_H5AD_URL, TEST_IMAGE_PATH


class TestPlotClusterTraj(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp()

        self.file_gef_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        self.file_h5ad_path = _download(DEMO_H5AD_URL, dir_str=TEST_DATA_PATH)

    def _preprocess(self, data):
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes', n_top_genes=None
        )
        data.tl.scale(zero_center=False)
        data.tl.pca(use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca',
                    svd_solver='arpack')
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)
        data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
        data.cells['leiden'] = data.cells['leiden'].astype('category')
        data.tl.paga(groups='leiden')

    def test_plot_cluster_traj_h5ad(self):
        data = AnnBasedStereoExpData(self.file_h5ad_path)
        self._preprocess(data)
        data.plt.plot_cluster_traj(data.tl.result['paga']['connectivities_tree'].todense(), data.position[:, 0],
                                   data.position[:, 1], data.cells['leiden'].to_numpy(), lower_thresh_not_equal=0.95,
                                   count_thresh=100, eps_co=30, check_surr_co=20, type_traj='curve',
                                   out_path=TEST_IMAGE_PATH + 'test_plot_cluster_traj_gef.tif')

    def test_plot_cluster_traj_gef(self):
        data = read_gef(self.file_gef_path)
        self._preprocess(data)
        data.plt.plot_cluster_traj(data.tl.result['paga']['connectivities_tree'].todense(), data.position[:, 0],
                                   data.position[:, 1], data.cells['leiden'].to_numpy(), lower_thresh_not_equal=0.95,
                                   count_thresh=100, eps_co=30, check_surr_co=20, type_traj='curve',
                                   out_path=TEST_IMAGE_PATH + 'test_plot_cluster_traj_h5ad.tif', )
