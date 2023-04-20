#!/usr/bin/env python3
# coding: utf-8
from .cellbin.modules.cell_segmentation import cell_seg_v3

try:
    from .pyramid import merge_pyramid, create_pyramid
    from .segmentation.segment import cell_seg

    from . import tissue_cut
    from .segmentation_deepcell.segment import cell_seg_deepcell
except ImportError as e:
    errmsg = """
Warning: function `merge_pyramid`, `create_pyramid`, `cell_seg`, `tissue_cut`, `cell_seg_deepcell` is not import at 
`stereo.image` module.
************************************************
* Some necessary modules may not be installed. *
* Include:                                     *
*         tensorflow==2.7.0                    *
*         torch==1.10.0                        *
*         torchvision==0.11.1                  *
*         albumentations==0.4.6                *
* Please install them by:                      *
*   pip install tensorflow==2.7.0              *
*   pip install torch==1.10.0                  *
*   pip install torchvision==0.11.1            *
*   pip install albumentations==0.4.6          *
************************************************
    """
    print(errmsg)
