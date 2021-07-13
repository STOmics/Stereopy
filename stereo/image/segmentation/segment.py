#!/usr/bin/env python
# coding: utf-8

import os
from .seg_utils import cell_seg_pipeline as pipeline


def cell_seg(img_path, out_path, flag, depp_cro_size=20000, overlap=100, gpu=None):
    """
    cell segmentation.

    :param img_path: image path
    :param out_path: the ouput path of mask result
    :param flag: watershed
    :param depp_cro_size: deep crop size
    :param overlap: the size of overlap
    :param gpu: the id of gpu, if None,use the cpu to predict.
    :return:
    """
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cell_seg_pipeline = pipeline.CellSegPipe(img_path, out_path, flag, depp_cro_size, overlap)
    cell_seg_pipeline.run()
