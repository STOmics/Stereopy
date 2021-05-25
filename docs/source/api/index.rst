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

文件读取模块

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

预处理模块

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

工具分析模块

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

画图模块

Basic plots
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   plots.plot_spatial_distribution
   plots.plot_genes_count
   plots.plot_violin_distribution
   plots.plot_dim_reduce
   plots.plot_spatial_cluster
   plots.plot_scatter
   plots.plot_marker_genes
   plots.plot_heatmap_marker_genes
