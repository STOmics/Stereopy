.. module:: stereo
.. automodule:: stereo
   :noindex:

API
===

Import Stereopy :

   import stereo as st

.. toctree::
   :maxdepth: 1

   031_API_IO

The `io` module includes input and output parts. 

The input part supports reading files in multiple formats to generate StereoExpData object, \
such as GEM, GEF, H5ad (from Scanpy and Seurat) and so on. In addition, we provide the conversion \
function between StereoExpData and AnnData so that swithing between tools would be smoother. \
In output part, we support writing StereoExpData as H5ad file for storage.


.. toctree::
   :maxdepth: 1

   032_API_StPipeline

A tool-integrated class for basic analysis of StereoExpData, \
which is compromised of basic preprocessing, embedding, clustering, and so on. 


.. toctree::
   :maxdepth: 1

   033_API_Plots

Here supports both static and interactive plotting modes to visualize our analysis results vividly. 


.. toctree::
   :maxdepth: 1

   034_API_Image

The module works on image files, generating results of tissue or cell segmentation that you are interested in.


.. toctree::
   :maxdepth: 1

   035_API_Tools

The module helps correct cell segmentation and generates corresponding CGEF file after cell cutting.


.. toctree::
   :maxdepth: 1

   036_API_Utils

The module handles with multiple batches of data.