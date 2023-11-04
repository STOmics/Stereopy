#!/usr/bin/env python
# coding: utf-8

import os

from stereo.constant import VersionType
from stereo.image.segmentation.seg_utils.v1 import CellSegPipeV1
from stereo.image.segmentation.seg_utils.v1_pro import CellSegPipeV1Pro
from ..cellbin.modules.cell_segmentation import cell_seg_v3


def cell_seg(
        img_path: str,
        out_path: str,
        model_path: str = None,
        deep_crop_size: int = 20000,
        overlap: int = 100,
        gpu: str = '-1',
        tissue_seg_model_path: str = None,
        tissue_seg_method: str = None,
        post_processing_workers: int = 10,
        is_water: bool = False,
        tissue_seg_dst_img_path=None,
        num_threads: int = 0,
        need_tissue_cut=True,
        version: str = 'v1',
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
        set gpu id, if `'-1'`, use cpu for prediction.
    tissue_seg_model_path
        the path of deep learning model of tissue segmentation, if set it to None, it would use OpenCV to process.
    tissue_seg_method
        the method of tissue segmentation, 1 is based on deep learning and 0 is based on OpenCV.
    post_processing_workers
        the number of processes for post-processing.
    is_water:
        The file name used to generate the mask. If true, the name ends with _watershed.
    tissue_seg_dst_img_path
        default to the img_path's directory.
    num_threads
        multi threads num of the model reading process
    need_tissue_cut
        whether cut image as tissue before cell segmentation
    version
        the version, version must be `v1` , `v1_pro`, `v3`

    Returns
    ------------
    None

    """
    if version not in VersionType.get_version_list():
        raise Exception("version must be %s" % ('、'.join(VersionType.get_version_list())))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if version in [VersionType.v1.value, VersionType.v3.value] and not model_path:
        raise Exception("cell_seg() missing 1 required keyword argument: 'model_path'")
    if version == VersionType.v1.value:
        cell_seg_pipeline = CellSegPipeV1(
            img_path,
            out_path,
            is_water,
            deep_crop_size,
            overlap,
            tissue_seg_model_path=tissue_seg_model_path,
            tissue_seg_method=tissue_seg_method,
            post_processing_workers=post_processing_workers,
            model_path=model_path
        )
        cell_seg_pipeline.run()
    elif version == VersionType.v1_pro.value:
        cell_seg_pipeline = CellSegPipeV1Pro(
            img_path,
            out_path,
            is_water,
            deep_crop_size,
            overlap,
            tissue_seg_model_path=tissue_seg_model_path,
            tissue_seg_method=tissue_seg_method,
            post_processing_workers=post_processing_workers,
            model_path=model_path
        )
        cell_seg_pipeline.run()
    else:
        cell_seg_v3(
            model_path,
            img_path,
            out_path,
            gpu=gpu,
            num_threads=num_threads,
            need_tissue_cut=need_tissue_cut,
            tissue_seg_model_path=tissue_seg_model_path,
            tissue_seg_method=tissue_seg_method,
            tissue_seg_dst_img_path=tissue_seg_dst_img_path,
        )
