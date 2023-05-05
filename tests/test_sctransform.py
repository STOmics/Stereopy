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
