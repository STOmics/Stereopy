import unittest

import numpy as np

from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.io.reader import read_gef
from stereo.utils._download import _download
from settings import TEST_DATA_PATH, DEMO_DATA_URL, DEMO_H5AD_URL, TEST_IMAGE_PATH


class TestPlotVec(unittest.TestCase):

    def setUp(self) -> None:
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

    def test_plot_vec_gef(self):
        data = read_gef(self.file_gef_path)
        self._preprocess(data)

        data.cells['leiden'] = data.cells['leiden'].astype('category')
        data.tl.result['iroot'] = np.flatnonzero(data.cells['leiden'] == '1')[100]

        data.tl.dpt(n_branchings=0)

        data.plt.plot_vec(
            data.position[:, 0],
            data.position[:, 1],
            data.cells['leiden'].to_numpy(),
            data.tl.result['dpt_pseudotime'],
            type='stream',
            line_width=0.5,
            background='field',
            num_pix=50,
            filter_type='gauss',
            sigma_val=1,
            radius_val=3,
            scatter_s=0.1,
            density=2,
            seed_val=0,
            num_legend_per_col=20,
            dpi_val=2000,
            out_path=TEST_IMAGE_PATH+"test_plot_vec_gef.tif"
        )

    def test_plot_time_scatter_gef(self):
        data = read_gef(self.file_gef_path)
        self._preprocess(data)

        data.cells['leiden'] = data.cells['leiden'].astype('category')
        data.tl.result['iroot'] = np.flatnonzero(data.cells['leiden'] == '1')[100]
        data.tl.dpt(n_branchings=0)

        data.plt.plot_time_scatter(out_path=TEST_IMAGE_PATH + "test_plot_time_scatter_gef.tif")

    def test_plot_time_scatter_h5ad(self):
        data = AnnBasedStereoExpData(self.file_h5ad_path)
        self._preprocess(data)

        data.cells['leiden'] = data.cells['leiden'].astype('category')
        data.tl.result['iroot'] = np.flatnonzero(data.cells['leiden'] == '1')[100]
        data.tl.dpt(n_branchings=0)

        data.plt.plot_time_scatter(out_path=TEST_IMAGE_PATH + "test_plot_time_scatter_h5ad.tif")

    def test_plot_vec_h5ad(self):
        data = AnnBasedStereoExpData(self.file_h5ad_path)
        self._preprocess(data)

        data.cells['leiden'] = data.cells['leiden'].astype('category')
        data.tl.result['iroot'] = np.flatnonzero(data.cells['leiden'] == '1')[100]

        data.tl.dpt(n_branchings=0)
        data.plt.plot_vec(
            data.position[:, 0],
            data.position[:, 1],
            data.cells['leiden'].to_numpy(),
            data.tl.result['dpt_pseudotime'],
            type='vec',
            line_width=1,
            background='scatter',
            num_pix=1,
            filter_type='gauss',
            sigma_val=1,
            radius_val=3,
            scatter_s=0.1,
            density=2,
            seed_val=0,
            num_legend_per_col=20,
            dpi_val=2000,
            out_path=TEST_IMAGE_PATH + "test_plot_vec_h5ad.tif"
        )
