.. module:: stereo
.. automodule:: stereo
   :noindex:

API
===

Stereo-seq Analysis By Python

Import stereopy as::

   import stereo as st


io: `reader`
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

.. autoclass:: stereo.plt.PlotCollection
   :members:
   :inherited-members:
