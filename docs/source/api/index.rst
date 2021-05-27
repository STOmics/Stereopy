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

io module, reading matrix and transform to annotation data.

reader
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   io.read_stereo_data
   io.read_10x_data


preprocess: `pp`
-------------------

.. module:: stereo.pp
.. currentmodule:: stereo

preprocess module

filter
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   preprocess.filter_cells
   preprocess.filter_genes
   preprocess.filter_coordinates


normalize
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   preprocess.Normalizer
   preprocess.normalize_total
   preprocess.normalize_zscore_disksmooth
   preprocess.quantile_norm


qc
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   preprocess.cal_qc


tools: `tl`
-------------------

.. module:: stereo.tl
.. currentmodule:: stereo

analysis tool module

Cell type annotation
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tools.CellTypeAnno


clustering
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tools.Clustering

dimensionality reduce
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tools.DimReduce
   tools.pca
   tools.u_map
   tools.factor_analysis
   tools.low_variance
   tools.t_sne

find marker
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tools.FindMarker


Spatial pattern score
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tools.SpatialPatternScore


Spatial lag
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tools.SpatialLag


plots: `pt`
-------------------

.. module:: stereo.pt
.. currentmodule:: stereo

visualization module

quality control plots
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   plots.plot_spatial_distribution
   plots.plot_genes_count
   plots.plot_violin_distribution


analysis plots
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   plots.plot_dim_reduce
   plots.plot_spatial_cluster
   plots.plot_scatter
   plots.plot_marker_genes
   plots.plot_heatmap_marker_genes
