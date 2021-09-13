.. module:: stereo
.. automodule:: stereo
   :noindex:

API
===

Stereo-seq Analysis By Python

Import stereopy as::

   import stereo as st


IO
-------------------

.. module:: stereo.io
.. currentmodule:: stereo

io module, reading gem format file into StereoExoData object.

.. autosummary::
   :toctree: .

   io.read_gem
   io.read_stereo_h5ad
   io.write_h5ad


data: `StereoExoData`
-------------------

.. module:: stereo
.. currentmodule:: stereo

.. autoclass:: stereo.core.stereo_exp_data.StereoExpData
   :members:
   :inherited-members:


pipeline: `tl`
-------------------

analysis tool module

.. autoclass:: stereo.core.st_pipeline.StPipeline
   :members:
   :inherited-members:

plots: `plt`
-------------------

.. module:: stereo.plt
.. currentmodule:: stereo

visualization module

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
   plots.genes_count
   plots.spatial_distribution


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
