Format Conversion
==================
This section contains the introduction of format conversion, so you can work with other spatial tools.

Working with scanpy package
--------------------------------------------------
The io module provides the function :mod:`stereo.io.stereo_to_anndata` to convert the StereoExpData into Anndata and output the
corresponding h5ad file(.h5ad).

StereoExpData to Anndata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:param data: StereoExpData object
:param flavor: 'scanpy' or 'seurat'. If you want to convert the output_h5ad into the rds file, set flavor='seurat'.
:param sample_id: name of sample. This will be set as 'orig.ident' in adata.obs.
:param reindex: whether to reindex the cell. The new index looks like "{sample_id}:{position_x}_{position_y}" format.
:param output: Default is None. If None, it will not generate a h5ad file.
:return: Anndata object

If you want to get the normalizetion result and convert the output_h5ad into the rds file,
you need to save raw data before you use normalization. Otherwise, it will raise errors during conversion.
Example like this:

.. code:: python

    import warnings
    warnings.filterwarnings('ignore')
    import stereo as st

    # read the GEF file
    mouse_data_path = './DP8400013846TR_F5.SN.tissue.gef'
    data = st.io.read_gef(file_path=mouse_data_path, bin_size=50)

    # Conversion.
    # If you want to convert the output_h5ad into the rds file, set flavor='seurat'.
    # Otherwise, it will raise errors during conversion.
    adata = st.io.stereo_to_anndata(data,flavor='seurat',output='out.h5ad')

Working with seurat package
-------------------------------------------------

h5ad to rds file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output h5ad could be converted into rds file by `annh5ad2rds.R <https://github.com/BGIResearch/stereopy/blob/dev/docs/source/_static/annh5ad2rds.R>`_.

It will generate a h5seurat file at first and then generate a rds file, so you can work with seurat package,

You can run this script in your own R environment:

.. code:: bash

    Rscript annh5ad2rds.R --infile <h5ad file> --outfile <rds file>
