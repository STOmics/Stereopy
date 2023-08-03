import pytest
import unittest
import numpy as np
import scanpy as sc
from settings import TEST_DATA_PATH
from stereo.utils import _download
from stereo.core.ms_data import MSData
from stereo.utils._download import _download
from settings import DEMO_FORE_BRAIN_DATA_URL
from stereo.core.ms_pipeline import slice_generator


class TestTimeSeriesAnalysis(unittest.TestCase):

    @pytest.mark.heavy
    def test_time_series_analysis(self):
        self._demo_fore_brain_data_file_path = _download(DEMO_FORE_BRAIN_DATA_URL, dir_str=TEST_DATA_PATH)

        from stereo.core.stereo_exp_data import AnnBasedStereoExpData
        data = AnnBasedStereoExpData(self._demo_fore_brain_data_file_path)
        data.tl.normalize_total()
        data.tl.log1p()
        data.tl.pca(svd_solver='arpack', n_pcs=20)
        data.tl.neighbors(n_neighbors=15, n_jobs=2, pca_res_key='pca')
        data.tl.paga(groups='class')
        data.tl.time_series_analysis(run_method="tvg_marker", use_col='timepoint',
                                     branch=['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5'],
                                     p_val_combination='FDR')

        data.plt.boxplot_transit_gene(use_col='timepoint',
                                      branch=['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5'],
                                      genes=['Trim2', 'Camk2b'])
        data.plt.paga_time_series_plot(use_col='class', batch_col='timepoint', height=10)

        data.tl.time_series_analysis(run_method="other")
        data.plt.TVG_volcano_plot(use_col='timepoint',
                                  branch=['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5'])
        data.plt.fuzz_cluster_plot(use_col='timepoint',
                                   branch=['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5'],
                                   threshold='p99.8', n_col=None, width=None, height=None)

    @pytest.mark.heavy
    def test_time_series(self):
        import stereo as st
        self._demo_fore_brain_data_file_path = _download(DEMO_FORE_BRAIN_DATA_URL, dir_str=TEST_DATA_PATH)
        adata = sc.read_h5ad(self._demo_fore_brain_data_file_path)
        data_merge = st.io.anndata_to_stereo(adata, spatial_key='spatial')

        t = adata.obs[['class']]
        t.columns = ['group' if x == 'class' else x for x in t.columns]
        data_merge.tl.result['annotation'] = t

        t = adata.obs[['timepoint']]
        t.columns = ['group' if x == 'timepoint' else x for x in t.columns]
        data_merge.tl.result['timepoint'] = t
        data_merge.cells['timepoint'] = t['group']

        data_merge.tl.normalize_total()
        data_merge.tl.log1p()
        slides = ['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5']

        ms_data = MSData()
        for i in slides:
            celllist = adata.obs.loc[adata.obs['timepoint'] == i].index
            adata_tmp = adata[celllist, :].copy()

            t = adata_tmp.obs[['class']]
            t.columns = ['group' if x == 'class' else x for x in t.columns]

            data = st.io.anndata_to_stereo(adata_tmp, spatial_key='spatial')
            data.tl.result['annotation'] = t

            ms_data[i] = data

        ms_data.integrate()
        ms_data.to_integrate(scope=slice_generator[:], res_key='annotation', _from=slice_generator[:], type='obs',
                             item=['annotation'] * ms_data.num_slice)
        # preprocessing
        ms_data.tl.normalize_total()
        ms_data.tl.log1p()
        # embedding
        ms_data.tl.pca(svd_solver='arpack', n_pcs=20)
        ms_data.tl.neighbors(n_neighbors=15, n_jobs=2, pca_res_key='pca')
        ms_data.tl.paga()
        ms_data.plt.time_series_tree_plot(use_result='annotation', edges='paga', height=3)

        edges = [('Dorsal forebrain', 'Neuronal intermediate progenitor'),
                 ('Neuronal intermediate progenitor', 'Cortical or hippocampal glutamatergic')]
        ms_data.plt.time_series_tree_plot(use_result='annotation', method='dot', ylabel_pos='right', edges=edges,
                                          height=3)

        ms_data.tl.result['scope_[0,1,2,3,4,5,6,7]']['iroot'] = np.flatnonzero(ms_data.obs['batch'] == '0')[10]
        ms_data.tl.dpt()

        ms_data.plt.ms_paga_time_series_plot(use_col='annotation', height=10)
