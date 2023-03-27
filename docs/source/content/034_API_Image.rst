.. module:: stereo
.. automodule:: stereo
   :noindex:

Image: `im`
====================

Import Stereopy :

   import stereo as st

The module works on image files, generating results of tissue or cell segmentation that you are interested in.

.. autosummary::
   :toctree: .

    image.pyramid.merge_pyramid
    image.pyramid.create_pyramid
    image.segmentation.segment.cell_seg
    image.segmentation_deepcell.segment.cell_seg_deepcell
    image.tissue_cut.SingleStrandDNATissueCut.__init__
    image.tissue_cut.RNATissueCut.__init__