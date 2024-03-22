#!/usr/bin/env python
# coding: utf-8

import os

from stereo.image.tissue_cut.pipeline import SingleStrandDNATissueCut
from .seg_utils import cell_seg_pipeline as pipeline


def cell_seg_deepcell(
        img_path: str,
        out_path: str,
        model_path: str = None,
        deep_crop_size: int = 20000,
        overlap: int = 100,
        # tissue_seg_method: str = None,
        post_processing_workers: int = 10,
        tissue_seg_model_path: str = None,
        tissue_seg_staining_type: str = None,
        tissue_seg_num_threads: int = -1,
        gpu: str = '-1'
):
    """
    Cell segmentation on regist.tif by deep cell model.

    :param img_path: the path of regist.tif without tissue segmentation.
    :param out_path: the path of directory to save the result cell mask.tif.
    :param model_path: the path of deep cell model.
    :param deep_crop_size: deep crop size, defaults to 20000.
    :param overlap: over lap size, defaults to 100.
    :param post_processing_workers: the number of processes on post processing, defaults to 10.
    :param tissue_seg_model_path: the path of model used to tissue segmentation, defaults to None.
    :param tissue_seg_staining_type: the staining type of regist.mask, defaults to None.
    :param tissue_seg_num_threads: the number of threads when model work on cpu, -1 means using all the cores.
    :param gpu: the gpu on which the model works, available for both cell segmtation and tissue segmtation, '-1' means working on cpu.

    """

    try:
        import tensorflow as tf  # noqa
    except Exception:
        raise Exception('please install tensorflow via `pip install tensorflow==2.7.0`.')
    
    tissue_seg = SingleStrandDNATissueCut(
        src_img_path=img_path,
        dst_img_path=out_path,
        model_path=tissue_seg_model_path,
        staining_type=tissue_seg_staining_type,
        gpu=gpu,
        num_threads=tissue_seg_num_threads
    )
    tissue_seg.tissue_seg()
    tissue_mask = tissue_seg.mask

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    flag = 0
    cell_seg_pipeline = pipeline.CellSegPipe(
        img_path,
        out_path,
        flag,
        deep_crop_size,
        overlap,
        model_path,
        # tissue_seg_model_path=tissue_seg_model_path,
        # tissue_seg_method=tissue_seg_method,
        tissue_mask=tissue_mask,
        post_processing_workers=post_processing_workers
    )
    cell_seg_pipeline.run()
