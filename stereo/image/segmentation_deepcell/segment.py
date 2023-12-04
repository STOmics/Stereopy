#!/usr/bin/env python
# coding: utf-8

import os

from .seg_utils import cell_seg_pipeline as pipeline


def cell_seg_deepcell(
        img_path: str,
        out_path: str,
        model_path: str = None,
        depp_cro_size: int = 20000,
        overlap: int = 100,
        gpu: str = '-1',
        tissue_seg_model_path: str = None,
        tissue_seg_method: str = None,
        post_processing_workers: int = 10,
        version: str = 'v1'
):
    """
    Implement cell segmentation by deep cell model.

    Parameters
    ------------------------
    model_path
        the path to deep cell model.
    img_path
        the path to image file.
    out_path
        the path to output mask result.
    depp_cro_size
        deep crop size.
    overlap
        overlap size.
    gpu
        set gpu id, if `'-1'`, use cpu for prediction.
    tissue_seg_model_path
        the path of deep learning model of tissue segmentation, if set it to None, it would use OpenCV to process.
    tissue_seg_method
        the method of tissue segmentation, 1 is based on deep learning and 0 is based on OpenCV.
    post_processing_workers
        the number of processes for post-processing.
    version
        the version

    Returns
    ------------------
    None
    """
    try:
        import tensorflow as tf  # noqa
    except Exception:
        raise Exception('please install tensorflow via `pip install tensorflow==2.4.1`.')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    flag = 0
    cell_seg_pipeline = pipeline.CellSegPipe(
        img_path,
        out_path,
        flag,
        depp_cro_size,
        overlap,
        model_path,
        tissue_seg_model_path=tissue_seg_model_path,
        tissue_seg_method=tissue_seg_method,
        post_processing_workers=post_processing_workers
    )
    cell_seg_pipeline.run()
