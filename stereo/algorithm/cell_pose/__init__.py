#!/usr/bin/env python3
# coding: utf-8

try:
    from stereo.algorithm.cell_pose.cell_pose import CellPost as cell_pose
except ImportError:
    errmsg = """class `CellPost` is not import at `stereo.cell_pose` module.
************************************************
* Some necessary modules may not be installed. *
* Please install them by:                      *
*   pip install pytorch                        *
*   pip install pyqtgraph                      *
*   pip install PyQt5                          *
*   pip install numpy(>=1.16.0)                *
*   pip install numba                          *
*   pip install scipy                          *
*   pip install natsort                        *
************************************************
    """
    raise ImportError(errmsg)
