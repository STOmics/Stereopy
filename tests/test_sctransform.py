import unittest

from stereo.io import read_gef, write_h5ad, read_stereo_h5ad, stereo_to_anndata
from stereo.utils._download import _download

from settings import DEMO_DATA_URL, TEST_DATA_PATH


class TestSCTransform(unittest.TestCase):

    def setUp(self) -> None:
        file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        self.DATA = read_gef(file_path)

    def test_scTransform(self):
        self.DATA.tl.sctransform()

    def test_scTransform_result_write(self):
        self.DATA.tl.raw_checkpoint()
        self.DATA.tl.sctransform(do_correct_umi=True, filter_hvgs=False)
        data = self.DATA

        raw_matrix = data.tl.raw.exp_matrix
        counts_matrix = data.tl.result['sctransform'][0]['counts'].T
        data_matrix = data.tl.result['sctransform'][0]['data'].T
        scale_matrix = data.tl.result['sctransform'][0]['scale.data'].to_numpy().T
        main_matrix = data.exp_matrix

        # check `main_matrix` is `scale_matrix`
        self.assertEqual(main_matrix.shape, scale_matrix.shape)
        self.assertAlmostEqual(main_matrix[1].sum(), scale_matrix[1].sum(), places=4)

        write_h5ad(self.DATA, output=TEST_DATA_PATH + "135.sct.h5ad")
        _ = read_stereo_h5ad(TEST_DATA_PATH + "135.sct.h5ad")

        ann_data = stereo_to_anndata(self.DATA)
        ann_raw_matrix = ann_data.raw.X
        ann_main_matrix = ann_data.X

        # check `ann_raw_matrix` is `raw_matrix`
        self.assertEqual(ann_raw_matrix.shape, raw_matrix.shape)
        self.assertAlmostEqual(ann_raw_matrix[1].sum(), raw_matrix[1].sum(), places=4)

        # check `ann_main_matrix` is `scale_matrix`
        self.assertEqual(ann_main_matrix.shape, scale_matrix.shape)
        self.assertAlmostEqual(ann_main_matrix[1].sum(), scale_matrix[1].sum(), places=4)

    def test_scTransform_result_write_1(self):
        self.DATA.tl.raw_checkpoint()
        self.DATA.tl.sctransform(do_correct_umi=False, filter_hvgs=True, var_features_n=2000)

        data = self.DATA
        raw_matrix = data.tl.raw.exp_matrix
        counts_matrix = data.tl.result['sctransform'][0]['counts'].T
        data_matrix = data.tl.result['sctransform'][0]['data'].T

        scale_matrix = data.tl.result['sctransform'][0]['scale.data'].to_numpy().T
        main_matrix = data.exp_matrix

        # check `main_matrix` is `scale_matrix`, and default var_features_n is 3000
        self.assertEqual(scale_matrix.shape[1], 2000)
        self.assertEqual(main_matrix.shape, scale_matrix.shape)
        self.assertAlmostEqual(main_matrix[1].sum(), scale_matrix[1].sum(), places=4)

        write_h5ad(self.DATA, output=TEST_DATA_PATH + "135.sct.h5ad")
        _ = read_stereo_h5ad(TEST_DATA_PATH + "135.sct.h5ad")

        ann_data = stereo_to_anndata(self.DATA)
        ann_raw_matrix = ann_data.raw.X
        ann_main_matrix = ann_data.X

        # check `ann_raw_matrix` is `scale_matrix`
        self.assertEqual(ann_raw_matrix.shape, raw_matrix.shape)
        self.assertAlmostEqual(ann_raw_matrix[1].sum(), raw_matrix[1].sum(), places=4)

        # check `ann_main_matrix` is `scale_matrix`
        self.assertEqual(ann_main_matrix.shape, scale_matrix.shape)
        self.assertAlmostEqual(ann_main_matrix[1].sum(), scale_matrix[1].sum(), places=4)

    def test_scTransform_result_write_2(self):
        self.DATA.tl.raw_checkpoint()
        self.DATA.tl.sctransform(do_correct_umi=True, filter_hvgs=True)

        data = self.DATA
        raw_matrix = data.tl.raw.exp_matrix
        counts_matrix = data.tl.result['sctransform'][0]['counts'].T
        data_matrix = data.tl.result['sctransform'][0]['data'].T

        scale_matrix = data.tl.result['sctransform'][0]['scale.data'].to_numpy().T
        main_matrix = data.exp_matrix

        # check `main_matrix` is `scale_matrix`, and default var_features_n is 3000
        self.assertEqual(scale_matrix.shape[1], 3000)
        self.assertEqual(main_matrix.shape, scale_matrix.shape)
        self.assertAlmostEqual(main_matrix[1].sum(), scale_matrix[1].sum(), places=4)

        write_h5ad(self.DATA, output=TEST_DATA_PATH + "135.sct.h5ad")
        _ = read_stereo_h5ad(TEST_DATA_PATH + "135.sct.h5ad")

        ann_data = stereo_to_anndata(self.DATA)
        ann_raw_matrix = ann_data.raw.X
        ann_main_matrix = ann_data.X

        # check `ann_raw_matrix` is `scale_matrix`
        self.assertEqual(ann_raw_matrix.shape, raw_matrix.shape)
        self.assertAlmostEqual(ann_raw_matrix[1].sum(), raw_matrix[1].sum(), places=4)

        # check `ann_main_matrix` is `scale_matrix`
        self.assertEqual(ann_main_matrix.shape, scale_matrix.shape)
        self.assertAlmostEqual(ann_main_matrix[1].sum(), scale_matrix[1].sum(), places=4)

    # def test_bwJS(self):
    #     from stereo.algorithm.sctransform.bw import bwSJ
    #     import numpy as np
    #     print(bwSJ(np.array([-2.3801186, -2.3801186, -2.1576715, -2.3801186, -2.3801186,
    #                          -2.3801186, -2.3801186, -2.4594538, -2.3130193, -2.2780812,
    #                          -2.3801186, -2.4594538, -2.2035720, -2.4594538, -2.4594538,
    #                          -2.4594538, -2.3801186, -2.3130193, -2.4594538, -2.4594538,
    #                          -2.3801186, -2.3801186, -2.1576715, -2.3801186, -2.4594538,
    #                          -2.3396304, -2.2035720, -2.4594538, -2.2035720, -2.3130193,
    #                          -2.4594538, -2.2548757, -2.3801186, -2.1576715, -2.1161213,
    #                          -2.3130193, -2.3801186, -2.1576715, -2.4594538, -2.4594538,
    #                          -2.4594538, -2.0935314, -2.3801186, -2.4594538, -2.3130193,
    #                          -2.4594538, -2.2035720, -2.0432710, -2.1576715, -2.3801186,
    #                          -2.3801186, -2.4594538, -2.3130193, -2.3801186, -2.4594538,
    #                          -2.4594538, -2.3130193, -2.4594538, -2.2035720, -2.3130193,
    #                          -2.4113126, -2.4594538, -2.3130193, -2.2548757, -2.2548757])))
    #     print(bwSJ(np.array([1.0, 2.0, 3.0])))

    # def test_nan(self):
    #     import stereo as st
    #     data = st.io.read_gef(file_path='/mnt/d/projects/stereopy_dev/demo_data/SS200000139BL_D5.gef', bin_size=50)
    #     data.tl.cal_qc()
    #     data.tl.filter_genes(min_cell=3, max_cell=100, gene_list=None, inplace=True)
    #     data.tl.cal_qc()
    #     data.tl.filter_cells(min_gene=1, min_n_genes_by_counts=1, max_n_genes_by_counts=2541, pct_counts_mt=100, inplace=True)
    #     data.tl.cal_qc()
    #     import numpy as np
    #     np.seterr(all="raise")
    #     data.tl.sctransform(n_cells=5000, n_genes=2000, filter_hvgs=False, var_features_n=None, inplace=True,
    #                         res_key='sctransform', exp_matrix_key="counts", seed_use=1448145, do_correct_umi=True)
    #     print(data.tl.result['sctransform'][0]['scale.data'])