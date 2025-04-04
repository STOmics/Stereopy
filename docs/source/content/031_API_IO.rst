.. module:: stereo
.. automodule:: stereo
   :noindex:

IO:  `io`
====================

Import Stereopy :

   import stereo as st


The `io` module includes input and output parts. 

The input part supports reading files in multiple formats to generate StereoExpData object, \
such as GEM, GEF, H5ad (from Scanpy and Seurat) and so on. In addition, we provide the conversion \
function between StereoExpData and AnnData so that swithing between tools would be smoother. \
In output part, we support writing StereoExpData as H5ad file for storage.


   The StereoExpData object is designed for efficient parallel computation, \
   which can satisfy the computing needs of multidimensional and massive data \
   especially generated by Stereo-seq with high resolution and large FOV.

    .. autosummary::
        :toctree: .

        core.StereoExpData
        core.ms_data.MSData


.. autosummary::
    :toctree: .

    io.read_gef_info
    io.read_gem
    io.read_gef
    io.read_h5ad
    io.read_seurat_h5ad
    io.read_h5ms
    io.mudata_to_msdata
    io.stereo_to_anndata
    io.write_h5ad
    io.write_mid_gef
    io.update_gef
    io.write_h5ms
    io.write_h5mu
    core.ms_data.MSData.integrate
    core.ms_data.MSData.to_integrate
    core.ms_data.MSData.to_isolated
