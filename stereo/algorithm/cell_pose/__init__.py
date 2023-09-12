#!/usr/bin/env python3
# coding: utf-8

try:
    from stereo.algorithm.cell_pose.cell_pose import Cellpose
except ImportError:
    errmsg = """class `Cellpose` is not import.
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
