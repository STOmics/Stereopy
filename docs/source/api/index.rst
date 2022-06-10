.. module:: stereo
.. automodule:: stereo
   :noindex:

API
===

Import stereopy as::

   import stereo as st


IO
-------------------

The io module supports reading files in multiple formats to generate StereoExpData object, such as gem, gef, h5ad and
so on. In addition, we provide the conversion function between AnnData and StereoExpData. Finally, we support writting
StereoExpData as h5ad file for storage.

.. autosummary::
   :toctree: .

    io.read_gef_info
    io.read_gem
    io.read_gef
    io.read_ann_h5ad
    io.read_stereo_h5ad
    io.anndata_to_stereo
    io.stereo_to_anndata
    io.write_h5ad
    io.write_mid_gef
    io.update_gef


StereoExpData
-------------------

A data designed for express matrix of spatial omics.

.. autoclass:: stereo.core.stereo_exp_data.StereoExpData
   :members:
   :inherited-members:

StPipeline
-------------------

A class for basic analysis of StereoExpData. It is the `StereoExpData.tl`.

.. autoclass:: stereo.core.st_pipeline.StPipeline
   :members:
   :inherited-members:

plots: `plt`
-------------------

.. module:: stereo.plt
.. currentmodule:: stereo

Visualization module

.. autosummary::
   :toctree: .


plot collection
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   plots.PlotCollection


scatter
~~~~~~~~
.. autosummary::
   :toctree: .

   plots.scatter.base_scatter
   plots.scatter.volcano
   plots.scatter.highly_variable_genes


interactive plot
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: .

   plots.interact_spatial_cluster
   plots.InteractiveScatter


others
~~~~~~~~
.. autosummary::
   :toctree: .

   plots.violin_distribution
   plots.marker_genes_text
   plots.marker_genes_heatmap




image: `im`
-------------------

Image parse module.

.. autosummary::
   :toctree: .

    image.merge_pyramid
    image.create_pyramid
    image.cell_seg
    image.cell_seg_deepcell
    image.tissue_seg


tools: `tools`
-------------------

Tools module.

.. autosummary::
   :toctree: .

    tools.cell_correct