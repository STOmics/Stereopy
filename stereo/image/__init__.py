#!/usr/bin/env python3
# coding: utf-8
# flake8: noqa
try:
    from .pyramid import merge_pyramid, create_pyramid
    from .segmentation.segment import cell_seg
    from .segmentation.seg_utils.v3 import CellSegPipeV3 as cell_seg_v3

    from . import tissue_cut
    from .segmentation_deepcell.segment import cell_seg_deepcell
except ImportError as e:
#     errmsg = """function `merge_pyramid`, `create_pyramid`, `cell_seg`, `cell_seg_v3`, `tissue_cut`, 
# `cell_seg_deepcell` is not import at `stereo.image` module.
# ************************************************
# * Some necessary modules may not be installed. *
# * Please install them by:                      *
# *   pip install tensorflow==2.7.0              *
# *   pip install torch==1.10.0                  *
# *   pip install torchvision==0.11.1            *
# *   pip install albumentations==0.4.6          *
# ************************************************
#     """
    errmsg = "Some necessary modules may not be installed. More information please refer to the documentation."
    raise ImportError(errmsg)
