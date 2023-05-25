import pytest
import unittest


class TestTimeSeriesAnalysis(unittest.TestCase):

    @pytest.mark.heavy
    def test_time_series_analysis(self):
        from stereo.core.stereo_exp_data import AnnBasedStereoExpData
        data = AnnBasedStereoExpData('/mnt/d/projects/stereopy_dev/demo_data/forebrain.anndata_075.repair2.h5ad')
        data.tl.normalize_total()
        data.tl.log1p()
        data.tl.pca(svd_solver='arpack', n_pcs=20)
        data.tl.neighbors(n_neighbors=15, n_jobs=2, pca_res_key='pca')
        data.tl.paga(groups='class')
        data.tl.time_series_analysis(run_method="tvg_marker", use_col='timepoint',
                                     branch=['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5'],
                                     p_val_combination='FDR')

        data.plt.boxplot_transit_gene(data, use_col='timepoint',
                                      branch=['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5'],
                                      genes=['Trim2', 'Camk2b'])
        data.plt.paga_time_series_plot(data, use_col='class', batch_col='timepoint', fig_height=10)

        data.tl.time_series_analysis(run_method="other")
        data.plt.TVG_volcano_plot(data, use_col='timepoint',
                                  branch=['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5'])
