#!/usr/bin/env python3
# coding: utf-8

try:
    from stereo.algorithm.cell_pose_seg.cell_pose_seg import CellSegmentation as cell_pose_seg
except ImportError:
    errmsg = """class `CellSegmentation` is not import at `stereo.cell_pose_seg` module.
************************************************
* Some necessary modules may not be installed. *
* Please install them by:                      *
*   pip install patchify                       *
*   pip install torch                          *
*   pip install fastremap                      *
*   pip install roifile                        *
************************************************
    """
    raise ImportError(errmsg)
