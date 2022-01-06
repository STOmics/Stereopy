Format Conversion
============
This section contains various short examples of format conversion.

StereoExpData to Anndata
-------------------
The io module provides the function :mod:`stereo.io.stereo_to_anndata` to convert the StereoExpData into anndata format and output the
corresponding h5ad file(.h5ad).

parameters
~~~~~~~~

:param data: StereoExpData object
:param flavor: 'scanpy' or 'seurat'.
If you want to converted the output_h5ad into the rds file, please set flavor='seurat'.

:param sample_id: name of sample.
Only when flavor == 'seurat', this will be set as 'orig.ident' in adata.obs.

When flavor='scanpy', this parameter is useless.

:param reindex: whether to reindex the cell.
Only when flavor='seurat', if you set reindex=True, it will combine the sample_id and position information to build
the new cell index automatically. Otherwise, index won't be changed in this step.

The new index looks like "{sample_id}:{position_x}_{position_y}" format.

When flavor='scanpy', this parameter is useless.

:param output: Default is None
If None, it will return a Anndata object, but not generate a h5ad file.

:return: Anndata object

h5ad to rds file
----------------------------------
The output h5ad could be converted into rds file by `annh5ad2rds.R <https://github.com/BGIResearch/stereopy/blob/dev/docs/source/_static/annh5adrds.R>`_.

It will generate a h5seurat file at first and then generate a rds file.

You can run this script in your own R environment:

.. code:: bash

    Rscript annh5ad2rds.R --infile <h5ad file> --outfile <rds file>
