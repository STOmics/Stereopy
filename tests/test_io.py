import unittest

from scipy.sparse import issparse

import stereo as st
from stereo.io import read_gef, read_gem, read_ann_h5ad
from stereo.utils._download import _download
from settings import DEMO_DATA_URL, TEST_DATA_PATH, DEMO_DATA_135_TISSUE_GEM_GZ_URL, DEMO_135_CELL_BIN_GEF_URL, \
    DEMO_135_CELL_BIN_GEM_URL, DEMO_H5AD_URL


class TestIO(unittest.TestCase):
    OBS_KEYS = {'total_counts', 'pct_counts_mt', 'n_genes_by_counts', 'orig.ident', 'x', 'y', 'leiden'}
    VAR_KEYS = {'n_counts', 'n_cells', 'means', 'dispersions', 'dispersions_norm', 'highly_variable'}
    UNS_KEYS = {'gene_exp_cluster@gene_exp_leiden', 'hvg', 'neighbors', 'sn'}
    OBSM_KEYS = {'X_pca', 'X_umap', 'spatial'}
    OBSP_KEYS = {'connectivities', 'distances'}
    SCT_UNS_KEYS = {'sn', 'hvg', 'sct_counts', 'sct_data', 'sct_scale', 'sct_scale_genename',
                    'sct_top_features', 'sct_cellname', 'sct_genename'}

    @classmethod
    def setUpClass(cls) -> None:
        cls.DEMO_135_TISSUE_GEF_PATH = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        cls.DEMO_135_CELL_BIN_GEF_PATH = _download(DEMO_135_CELL_BIN_GEF_URL, dir_str=TEST_DATA_PATH)

        cls.DEMO_135_TISSUE_GEM_GZ_PATH = _download(DEMO_DATA_135_TISSUE_GEM_GZ_URL, dir_str=TEST_DATA_PATH)
        cls.DEMO_135_CELL_BIN_GEM_PATH = _download(DEMO_135_CELL_BIN_GEM_URL, dir_str=TEST_DATA_PATH)

        cls.DEMO_H5AD_PATH = _download(DEMO_H5AD_URL, dir_str=TEST_DATA_PATH)

    def test_read_gef_bin(self):
        data = read_gef(
            self.DEMO_135_TISSUE_GEF_PATH,
            bin_type="bins",
            bin_size=100,
            is_sparse=True
        )
        self.assertEqual(data.shape, (9111, 20816))
        self.assertEqual(data.exp_matrix.shape, (9111, 20816))
        self.assertTrue(issparse(data.exp_matrix))
        # self.assertIs(data.exp_matrix.dtype, numpy.uint32)
        self.assertEqual(data.sn, "SS200000135TL_D1")
        self.assertEqual(data.attr['minX'], 0)
        self.assertEqual(data.attr['minY'], 2)
        self.assertEqual(data.attr['maxX'], 13441)
        self.assertEqual(data.attr['maxY'], 19776)
        self.assertEqual(data.attr['maxExp'], 14)
        self.assertEqual(data.attr['resolution'], 500)
        self.assertEqual(data.bin_size, 100)
        self.assertEqual(data.bin_type, "bins")
        self.assertEqual(data.offset_x, 0)
        self.assertEqual(data.offset_y, 2)
        # self.assertIs(data.position.dtype, numpy.uint32)
        self.assertEqual(data.position.shape, (9111, 2))
        self.assertIs(data.raw, None)
        self.assertEqual(data.merged, False)
        self.assertEqual(len(data.cell_names), len(data.cells.cell_name))
        self.assertEqual(len(data.cells._obs.index), data.exp_matrix.shape[0])
        self.assertEqual(len(data.gene_names), len(data.genes.gene_name))
        self.assertEqual(len(data.genes._var.index), data.exp_matrix.shape[1])

    def test_read_gef_cell_bin(self):
        data = read_gef(
            self.DEMO_135_CELL_BIN_GEF_PATH,
            bin_type="cell_bins",
            bin_size=100,
            is_sparse=True
        )
        self.assertEqual(data.shape, (56204, 24661))
        self.assertEqual(data.exp_matrix.shape, (56204, 24661))
        self.assertTrue(issparse(data.exp_matrix))
        # self.assertIs(data.exp_matrix.dtype, numpy.uint32)
        self.assertEqual(data.sn, "SS200000135TL_D1")
        self.assertEqual(data.attr['resolution'], 500)
        self.assertEqual(data.bin_size, 100)
        self.assertEqual(data.bin_type, "cell_bins")
        self.assertIs(data.offset_x, None)
        self.assertIs(data.offset_y, None)
        # self.assertIs(data.position.dtype, numpy.uint32)
        self.assertEqual(data.position.shape, (56204, 2))
        self.assertEqual(data.merged, False)
        self.assertEqual(len(data.cell_names), len(data.cells.cell_name))
        self.assertEqual(len(data.cells._obs.index), data.exp_matrix.shape[0])
        self.assertEqual(len(data.gene_names), len(data.genes.gene_name))
        self.assertEqual(len(data.genes._var.index), data.exp_matrix.shape[1])
        self.assertEqual(data.cells.cell_border.shape[0], 56204)

    def test_read_gem_bin(self):
        data = read_gem(
            self.DEMO_135_TISSUE_GEM_GZ_PATH,
            bin_type="bins",
            bin_size=100,
            is_sparse=True
        )
        self.assertEqual(data.shape, (9100, 20816))
        self.assertEqual(data.exp_matrix.shape, (9100, 20816))
        self.assertTrue(issparse(data.exp_matrix))
        # self.assertIs(data.exp_matrix.dtype, numpy.uint32)
        self.assertEqual(data.sn, "SS200000135TL_D1")
        self.assertEqual(data.attr['minX'], 3247)
        self.assertEqual(data.attr['minY'], 6209)
        self.assertEqual(data.attr['maxX'], 13441)
        self.assertEqual(data.attr['maxY'], 19776)
        self.assertEqual(data.attr['maxExp'], 303)
        self.assertEqual(data.attr['resolution'], 500)
        self.assertEqual(data.bin_size, 100)
        self.assertEqual(data.bin_type, "bins")
        self.assertEqual(data.offset_x, 3247)
        self.assertEqual(data.offset_y, 6209)
        # self.assertIs(data.position.dtype, numpy.uint32)
        self.assertEqual(data.position.shape, (9100, 2))
        self.assertEqual(data.merged, False)
        self.assertEqual(len(data.cell_names), len(data.cells.cell_name))
        self.assertEqual(len(data.cells._obs.index), data.exp_matrix.shape[0])
        self.assertEqual(len(data.gene_names), len(data.genes.gene_name))
        self.assertEqual(len(data.genes._var.index), data.exp_matrix.shape[1])

    def test_read_gem_cell_bin(self):
        data = read_gem(
            self.DEMO_135_CELL_BIN_GEM_PATH,
            bin_type="cell_bins",
            bin_size=100,
            is_sparse=True
        )
        self.assertEqual(data.shape, (56200, 22413))
        self.assertEqual(data.exp_matrix.shape, (56200, 22413))
        self.assertTrue(issparse(data.exp_matrix))
        # self.assertIs(data.exp_matrix.dtype, numpy.uint32)
        self.assertEqual(data.sn, "SS200000135TL_D1")
        self.assertEqual(data.attr['minX'], 3254)
        self.assertEqual(data.attr['minY'], 6211)
        self.assertEqual(data.attr['maxX'], 13409)
        self.assertEqual(data.attr['maxY'], 19777)
        self.assertEqual(data.attr['maxExp'], 361)
        self.assertEqual(data.attr['resolution'], 500)
        self.assertEqual(data.bin_size, 100)
        self.assertEqual(data.bin_type, "cell_bins")
        self.assertEqual(data.offset_x, 3254)
        self.assertEqual(data.offset_y, 6211)
        # self.assertIs(data.position.dtype, numpy.uint32)
        self.assertEqual(data.position.shape, (56200, 2))
        self.assertEqual(data.merged, False)
        self.assertEqual(len(data.cell_names), len(data.cells.cell_name))
        self.assertEqual(len(data.cells._obs.index), data.exp_matrix.shape[0])
        self.assertEqual(len(data.gene_names), len(data.genes.gene_name))
        self.assertEqual(len(data.genes._var.index), data.exp_matrix.shape[1])

    def test_read_ann_h5ad(self):
        data = read_ann_h5ad(self.DEMO_H5AD_PATH, spatial_key='spatial')
        self.assertEqual(data.shape, (61857, 24562))
        self.assertEqual(data.exp_matrix.shape, (61857, 24562))
        self.assertTrue(issparse(data.exp_matrix))
        # self.assertIs(data.exp_matrix.dtype, numpy.uint32)
        self.assertEqual(data.sn, "SS200000141TL_B5_raw")
        # self.assertIs(data.attr, None)
        self.assertIs(data.bin_size, None)
        self.assertIs(data.bin_type, None)
        self.assertIs(data.offset_x, None)
        self.assertIs(data.offset_y, None)
        # self.assertIs(data.position.dtype, numpy.uint32)
        self.assertEqual(data.position.shape, (61857, 2))
        self.assertEqual(data.merged, False)
        self.assertEqual(len(data.cell_names), len(data.cells.cell_name))
        self.assertEqual(len(data.cells._obs.index), data.exp_matrix.shape[0])
        self.assertEqual(len(data.gene_names), len(data.genes.gene_name))
        self.assertEqual(len(data.genes._var.index), data.exp_matrix.shape[1])

    def test_read_stereo_h5ad(self):
        pass

    def test_read_seurat_h5ad(self):
        pass

    def test_anndata_to_stereo(self):
        pass

    def test_stereo_to_anndata(self):
        data = st.io.read_gef(self.DEMO_135_TISSUE_GEF_PATH, bin_size=100)
        data.tl.cal_qc()
        data.tl.raw_checkpoint()
        data.tl.normalize_total(target_sum=10000)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000, res_key='highly_variable_genes'
        )
        data.tl.scale(zero_center=True, max_value=None, inplace=True, res_key='scale')
        # TODO in order to use less disk
        # st.io.write_h5ad(data, use_raw=True, use_result=True, output=TEST_DATA_PATH + 'test_135_preprocessing.h5ad')
        # data = st.io.read_stereo_h5ad(file_path=TEST_DATA_PATH + 'test_135_preprocessing.h5ad')
        data.tl.pca(use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=50, res_key='pca')
        data.tl.neighbors(pca_res_key='pca', n_pcs=50, n_neighbors=15, res_key='neighbors')
        data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap')
        data.tl.leiden(neighbors_res_key='neighbors', resolution=1.0, res_key='leiden')
        # TODO in order to use less disk
        # st.io.write_h5ad(data, use_raw=True, use_result=True, output=TEST_DATA_PATH + 'test_135_cluster.h5ad')
        # ann_data = st.io.stereo_to_anndata(data, output=TEST_DATA_PATH + 'test_135_cluster_anndata.h5ad')
        ann_data = st.io.stereo_to_anndata(data)
        for obs_key in TestIO.OBS_KEYS:
            self.assertIn(obs_key, ann_data.obs)
        for var_key in TestIO.VAR_KEYS:
            self.assertIn(var_key, ann_data.var)
        for uns_key in TestIO.UNS_KEYS:
            self.assertIn(uns_key, ann_data.uns)
        for obsm_key in TestIO.OBSM_KEYS:
            self.assertIn(obsm_key, ann_data.obsm)
        for obsp_key in TestIO.OBSP_KEYS:
            self.assertIn(obsp_key, ann_data.obsp)
        self.assertIsNotNone(ann_data.raw)
        self.assertAlmostEqual(ann_data.raw.X.sum(), data.tl.raw.exp_matrix.sum())
        # TODO only choose the second line to check
        self.assertAlmostEqual(data.exp_matrix[1].sum(), ann_data.X[1].sum(), places=2)
        self.assertEqual(data.cells_matrix['pca'].shape, ann_data.obsm['X_pca'].shape)
        self.assertEqual(data.cells_matrix['umap'].shape, ann_data.obsm['X_umap'].shape)
        self.assertEqual(
            data.cells_pairwise['neighbors']['connectivities'].shape, ann_data.obsp['connectivities'].shape
        )
        self.assertEqual(
            data.cells_pairwise['neighbors']['connectivities'].shape, ann_data.obsp['connectivities'].shape
        )
        self.assertEqual(
            data.cells_pairwise['neighbors']['nn_dist'].shape, ann_data.obsp['distances'].shape
        )

    def test_stereo_to_anndata_sct(self):
        data = st.io.read_gef(self.DEMO_135_TISSUE_GEF_PATH, bin_size=100)
        data.tl.cal_qc()
        data.tl.raw_checkpoint()
        exp = data.exp_matrix
        genenum = exp.shape[1]
        cellnum = exp.shape[0]
        if genenum >= 2000:
            genenum = 2000
        if cellnum >= 5000:
            cellnum = 5000
        data.tl.sctransform(n_cells=cellnum, n_genes=genenum, filter_hvgs=False, var_features_n=2000, inplace=True,
                            res_key='sctransform', exp_matrix_key="counts", seed_use=1448145, do_correct_umi=True)
        data.tl.raw_checkpoint()
        data.exp_matrix = data.tl.result['sctransform'][0]['scale.data'].T.values
        sct_scale_data = data.tl.result['sctransform'][0]['scale.data'].T
        data.cells.cell_name = sct_scale_data.index.values
        data.genes.gene_name = sct_scale_data.columns.values
        data.tl.highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000, method='cell_ranger',
                                      res_key='highly_variable_genes')
        df = data.tl.result['highly_variable_genes']
        df['highly_variable'] = 0
        for i in data.tl.result['sctransform'][1]['top_features']:
            findcol = df[df.index == i].index.tolist()[0]
            df['highly_variable'][findcol] = 1
        import numpy
        df['highly_variable'] = numpy.array(df['highly_variable'], dtype=bool)
        ann_data = st.io.stereo_to_anndata(data)
        for uns_key in TestIO.SCT_UNS_KEYS:
            self.assertIn(uns_key, ann_data.uns)
        self.assertEqual(len(data.tl.result['sctransform'][1]['top_features']), 2000)
        self.assertEqual(len(ann_data.uns['sct_top_features']), 2000)
        # ann_data.raw.X is sample as data.tl.raw.exp_matrix
        self.assertEqual(ann_data.raw.X.shape, data.tl.raw.exp_matrix.shape)
        self.assertEqual(ann_data.raw.X[1].sum(), data.tl.raw.exp_matrix[1].sum())
        # ann_data.X is sample as data.exp_matrix
        self.assertEqual(ann_data.X.shape, data.exp_matrix.shape)
        self.assertAlmostEqual(ann_data.X[1].sum(), data.exp_matrix[1].sum(), places=4)
        # ann_data.X is from sct `scale.data`
        self.assertEqual(ann_data.X.shape, data.tl.result['sctransform'][0]['scale.data'].T.shape)
        self.assertAlmostEqual(
            ann_data.X[1].sum(), data.tl.result['sctransform'][0]['scale.data'].T.to_numpy()[1].sum(),
            places=4
        )
        print(ann_data)

    def test_write_h5ad(self):
        pass

    def test_update_gef(self):
        pass

    def test_write(self):
        pass

    def test_write_mid_gef(self):
        pass

    def test_save_pkl(self):
        pass
