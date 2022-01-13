Format Conversion
============
This section contains various short examples.

StereoExpData to Anndata
-------------------
The io module provides the function to convert the StereoExpData into anndata format and output the
corresponding h5ad file(.h5ad).

.. autosummary::
   :toctree: .

    io.stereo_to_anndata

parameters
~~~~~~~~

:param data: StereoExpData object
:param flavor: 'scanpy' or 'seurat'.
If you want to converted the output_h5ad into the rds file, please set flavor='seurat'.

:param sample_id: name of sample.
this will be set as 'orig.ident' in adata.obs.

:param reindex: whether to reindex the cell.
The new index looks like "{sample_id}:{position_x}_{position_y}" format.

:param output: Default is None
If None, it will return an Anndata object, but not generate a h5ad file.

:return: Anndata object

h5ad to rds file
----------------------------------
The output h5ad could be converted into rds file by annh5ad2rds.R.

It will generate a h5seurat file at first and then generate a rds file.

You can run this script in your own R environment.

.. autosummary::
   :toctree: .

    utils.annh5ad2rds.R

    Rscript annh5ad2rds.R --infile <h5ad file> --outfile <rds file>
