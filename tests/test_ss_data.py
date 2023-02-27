import copy
import unittest

from stereo.core.ss_data import SSData
from stereo.io.reader import read_gef


class SSDataTestCases(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super(SSDataTestCases, self).setUp()
        self.ss_data = SSData()
        # self.obj = read_gef('d:\\projects\\stereopy_dev\\demo_data\\SS200000135TL_D1\\SS200000135TL_D1.tissue.gef')
        self.obj = read_gef('/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef')
        self.obj2 = copy.deepcopy(self.obj)

        self.ss_data.add_data(self.obj)
        self.ss_data.add_data(self.obj2, 'a')

    def test_length_match_init(self):
        ss_data = SSData(_data_list=[copy.deepcopy(self.obj), copy.deepcopy(self.obj2)], _s_names=['a', 'c'])
        self.assertEqual(len(ss_data._s_names), len(ss_data.data_list))
        self.assertIs(ss_data[0], ss_data['a'])
        self.assertIs(ss_data[1], ss_data['c'])

    def test_length_not_match_init(self):
        ss_data = SSData(_data_list=[copy.deepcopy(self.obj), copy.deepcopy(self.obj2)], _s_names=['a'])
        self.assertEqual(len(ss_data._s_names), len(ss_data.data_list))
        self.assertIs(ss_data[0], ss_data['a'])
        self.assertIs(ss_data[1], ss_data['0'])

    def test_add(self):
        self.assertEqual(len(self.ss_data), 2)
        self.assertIs(self.ss_data[0], self.obj)
        self.assertIs(self.ss_data[1], self.obj2)

    def test_add_path(self):
        self.ss_data.add_data('/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef')

    def test_multi_add(self):
        self.ss_data.add_data(copy.deepcopy(self.obj2), 'c')
        self.assertIs(self.ss_data['0'], self.ss_data[0])
        self.assertIs(self.ss_data['c'], self.ss_data[2])

    def test_multi_add_path(self):
        self.ss_data.add_data(
            [
                '/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef',
                '/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1_script_res_gem.h5ad',
                '/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gem'
            ],
            [
                'z',
                'x',
                'y'
            ],
            bin_size=[100, 100, 200],
            bin_type=['bins', 'cell_bins', 'bins'],
        )

    def test_del_data(self):
        self.ss_data.del_data('0')
        self.assertNotIn('0', self.ss_data)
        self.assertEqual(len(self.ss_data), 1)

    def test_names(self):
        for idx, data_obj in enumerate(self.ss_data.data_list):
            self.assertIs(self.ss_data[self.ss_data._s_names[idx]], data_obj)

    def test_rename(self):
        # rename with a not existed key and three existed keys
        obj_0 = self.ss_data['0']
        obj_1 = self.ss_data['a']
        self.ss_data.rename({'0': '100', 'a': '101'})
        self.assertIs(obj_0, self.ss_data['100'])
        self.assertIs(obj_1, self.ss_data['101'])
        self.assertNotIn('cc', self.ss_data._s_names)

    def test_names_order(self):
        for idx, name in enumerate(self.ss_data._s_names):
            self.assertIs(self.ss_data[name], self.ss_data[idx])

    def test_reset_name(self):
        self.ss_data.reset_name()
        self.assertIs(self.ss_data['0'], self.obj)
        self.assertIs(self.ss_data['1'], self.obj2)

    def test_name_contain(self):
        self.assertIn('0', self.ss_data)
        self.assertIn(self.obj, self.ss_data)
        self.assertIn('a', self.ss_data)
        self.assertIn(self.obj2, self.ss_data)

    def test_tl_method(self):
        self.ss_data.tl.log1p()

    def test_tl_method_algorithm_base(self):
        self.ss_data.tl.log1p_fake()

    def test_s_n(self):
        self.assertEqual(len(self.ss_data), self.ss_data.s_n, len(self.ss_data.data_list))

    def test_copy(self):
        self.assertIs(copy.deepcopy(self.ss_data), self.ss_data, copy.copy(self.ss_data))

    def test_clustering(self):
        self.ss_data.tl.cal_qc()
        self.ss_data.tl.filter_cells(min_gene=200, min_n_genes_by_counts=3, max_n_genes_by_counts=7000, pct_counts_mt=8,
                                     inplace=False)
        self.ss_data.tl.log1p()
        self.ss_data.tl.normalize_total(target_sum=1e4)
        self.ss_data.tl.pca(use_highly_genes=False, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca',
                            svd_solver='arpack')
        self.ss_data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=8)
        self.ss_data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap', init_pos='spectral')
        self.ss_data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')


if __name__ == "__main__":
    unittest.main()
