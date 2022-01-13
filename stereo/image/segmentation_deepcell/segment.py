#!/usr/bin/env python
# coding: utf-8

import os
from .seg_utils import cell_seg_pipeline as pipeline


def cell_seg_deepcell(model_path, img_path, out_path, depp_cro_size=20000, overlap=100, gpu=-1):
    """
    cell segmentation.

    :param model_path: the dir path of model.
    :param img_path: image path
    :param out_path: the ouput path of mask result
    :param depp_cro_size: deep crop size
    :param overlap: the size of overlap
    :param gpu: the id of gpu, if -1,use the cpu to predict.
    :return:
    """
    try:
        import tensorflow as tf
    except Exception:
        raise Exception('please install tensorflow via `pip install tensorflow==2.4.1`.')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    flag = 0
    cell_seg_pipeline = pipeline.CellSegPipe(img_path, out_path, flag, depp_cro_size, overlap, model_path)
    cell_seg_pipeline.run()
