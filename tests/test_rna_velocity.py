import unittest
import stereo as st

from stereo.tools import generate_loom

from stereo.utils._download import _download

from settings import TEST_DATA_PATH, DEMO_DATA_URL, DEMO_GTF_URL, TEST_IMAGE_PATH

class TestSpatialHotspot(unittest.TestCase):
    def setUp(self) -> None:
        self.test_file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        self.gtf_file = _download(DEMO_GTF_URL, dir_str=TEST_DATA_PATH)

    def test_spatial_hotspot(self):

        loom_data = generate_loom(
            gef_path=self.test_file_path, 
            gtf_path=self.gtf_file,
            bin_type='bins',
            bin_size=100,
            out_dir=TEST_DATA_PATH
            )
        
        try:
            import dynamo as dyn
        except ImportError as e:
            st.logger.info(f'Please pip install dynamo-release, exception: {str(e)}')
            return
        
        dyn.configuration.set_figure_params('dynamo', background='white')
        # read data
        adata = dyn.read_loom(loom_data)
        dyn.pp.recipe_monocle(
            adata,
            num_dim=30,
            keep_filtered_genes = True
            )
        
        dyn.tl.dynamics(adata, model='stochastic', cores=60)
        dyn.tl.moments(adata)
        dyn.tl.reduceDimension(adata)

        adata.obs['x'] = list(map(lambda x: float(x.split("_")[0]) ,list(adata.obs.index)))
        adata.obs['y'] = list(map(lambda x: float(x.split("_")[1]) ,list(adata.obs.index)))

        adata.obsm['spatial'] = adata.obs[['x', 'y']].values.astype(float)
        adata.obsm['X_spatial'] = adata.obs[['x', 'y']].values.astype(float)

        dyn.tl.cell_velocities(
            adata, 
            method='fp', 
            basis='X_spatial', 
            enforce=True,  
            transition_genes = list(adata.var_names[adata.var.use_for_pca])
            )

        dyn.tl.cell_wise_confidence(adata)
        dyn.tl.leiden(adata, result_key='spatial_leiden_res')

        dyn.pl.streamline_plot(
            adata, 
            color = 'spatial_leiden_res', 
            basis='X_spatial',
            quiver_length=6, 
            quiver_size=6,  
            show_arrowed_spines=True,
            figsize=(6, 6),
            save_show_or_return='save',
            save_kwargs={"path": TEST_IMAGE_PATH, "prefix": 'streamline_plot', "dpi": None, "ext": 'png', "transparent": True, "close": True, "verbose": True}
            )