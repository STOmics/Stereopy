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

io module, reading gem format file into StereoExoData object.

.. autosummary::
   :toctree: .

   io.read_gem
   io.read_stereo_h5ad
   io.write_h5ad


StereoExoData
-------------------

.. autosummary::
   :toctree: generated/classes

   core.stereo_exp_data.StereoExpData



StPipeline: `StereoExpData.tl`
-------------------

analysis tool module

.. autoclass:: stereo.core.st_pipeline.StPipeline
   :members:
   :inherited-members:

PlotCollection: `StereoExpData.plt`
-------------

.. autoclass:: stereo.plt.PlotCollection
   :members:
   :inherited-members:

plots: `plt`
-------------------

.. module:: stereo.plt
.. currentmodule:: stereo


