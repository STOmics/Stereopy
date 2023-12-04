#!/usr/bin/env python
# coding: utf-8

import os

from stereo.constant import VersionType
from stereo.image.segmentation.seg_utils.v1 import CellSegPipeV1
from stereo.image.segmentation.seg_utils.v1_pro import CellSegPipeV1Pro
from stereo.image.segmentation.seg_utils.v3 import CellSegPipeV3


def cell_seg(
        model_path: str,
        img_path: str,
        out_path: str,
        deep_crop_size: int = 20000,
        overlap: int = 100,
        gpu: str = '-1',
        tissue_seg_model_path: str = None,
        tissue_seg_method: str = None,
        post_processing_workers: int = 10,
        is_water: bool = False,
        num_threads: int = 0,
        need_tissue_cut=True,
        method: str = 'v1',
):
    """
    Implement cell segmentation by deep learning model.

    Parameters
    -----------------
    model_path
        the path to deep learning model.
    img_path
        the path to image file.
    out_path
        the path to output mask result.
    deep_crop_size
        deep crop size.
    overlap
        overlap size.
    gpu
        set gpu id, if int(gpu)<0, use cpu for prediction, otherwise use gpu for prediction
    tissue_seg_model_path
        the path of deep learning model of tissue segmentation, if set it to None, it would use OpenCV to process.
    tissue_seg_method
        the method of tissue segmentation, 1 is based on deep learning and 0 is based on OpenCV.
    post_processing_workers
        the number of processes for post-processing.
    is_water:
        The file name used to generate the mask. If true, the name ends with _watershed.
    num_threads
        multi threads num of the model reading process
    need_tissue_cut
        whether cut image as tissue before cell segmentation
    method
        the method, method must be `v1` , `v1_pro`, `v3`

    Returns
    ------------
    None

    """
    if method not in VersionType.get_version_list():
        raise Exception("version must be %s" % ('、'.join(VersionType.get_version_list())))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if method == VersionType.v1.value:
        cell_seg_pipeline = CellSegPipeV1(
            model_path,
            img_path,
            out_path,
            is_water,
            deep_crop_size,
            overlap,
            gpu=gpu,
            need_tissue_cut=need_tissue_cut,
            tissue_seg_model_path=tissue_seg_model_path,
            tissue_seg_method=tissue_seg_method,
            post_processing_workers=post_processing_workers,
        )
        cell_seg_pipeline.run()
    elif method == VersionType.v3.value:
        cell_seg_pipeline = CellSegPipeV3(
            model_path,
            img_path,
            out_path,
            is_water,
            deep_crop_size,
            overlap,
            gpu=gpu,
            need_tissue_cut=need_tissue_cut,
            tissue_seg_model_path=tissue_seg_model_path,
            tissue_seg_method=tissue_seg_method,
            post_processing_workers=post_processing_workers,
        )
        cell_seg_pipeline.run()
    elif method == VersionType.v1_pro.value:
        cell_seg_pipeline = CellSegPipeV1Pro(
            model_path,
            img_path,
            out_path,
            is_water,
            deep_crop_size,
            overlap,
            gpu=gpu,
            need_tissue_cut=need_tissue_cut,
            tissue_seg_model_path=tissue_seg_model_path,
            tissue_seg_method=tissue_seg_method,
            post_processing_workers=post_processing_workers,
        )
        cell_seg_pipeline.run()
