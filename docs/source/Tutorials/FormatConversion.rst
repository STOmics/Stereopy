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
If you want to converted the output_h5ad into h5seurat for seurat, please set flavor='seurat'.

:param sample_id: name of sample.
Only when flavor == 'seurat', this will be set as 'orig.ident' in adata.obs.

:param reindex: the present index of cell is as same as your input file
When flavor='seurat', if you set reindex=True, the program will combine the sample_id and position information to build
the new cell index, and it will be reindexd as "{sample_id}:{position_x}_{position_y}" format.

When flavor='scanpy', this parameter is useless.

:param output: path of output_file

:return: Anndata object

h5ad to rds file
----------------------------------
The output h5ad could be converted into rds file by annh5ad2rds.R.

It will generate a h5seurat file at first and then generate a rds file.

.. autosummary::
   :toctree: .

    utils.annh5ad2rds.R

    Rscript annh5ad2rds.R --infile <h5ad file> --outfile <rds file>
