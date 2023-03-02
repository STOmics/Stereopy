#!/usr/bin/env python
# coding: utf-8

import os
from .seg_utils import cell_seg_pipeline as pipeline


def cell_seg_deepcell(
        model_path: str, 
        img_path: str, 
        out_path: str, 
        depp_cro_size: int=20000, 
        overlap: int=100, 
        gpu: str='-1'):
    """
    Implement cell segmentation by deep cell model.

    Parameters
    ------------------------
    model_path
        - the path to deep cell model.
    img_path
        - the path to image file.
    out_path
        - the path to output mask result.
    depp_cro_size
        - deep crop size.
    overlap
        - overlap size.
    gpu
        - set gpu id, if `'-1'`, use cpu for prediction.

    Returns
    ------------------
    None
    """
    try:
        import tensorflow as tf
    except Exception:
        raise Exception('please install tensorflow via `pip install tensorflow==2.4.1`.')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    flag = 0
    cell_seg_pipeline = pipeline.CellSegPipe(img_path, out_path, flag, depp_cro_size, overlap, model_path)
    cell_seg_pipeline.run()
