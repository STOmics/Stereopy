#!/usr/bin/env python
# coding: utf-8

import os

from stereo.constant import VersionType
from stereo.image.segmentation.seg_utils.v1 import CellSegPipeV1
from stereo.image.segmentation.seg_utils.v1_pro import CellSegPipeV1Pro
from stereo.image.segmentation.seg_utils.v3 import CellSegPipeV3
from stereo.image.tissue_cut.pipeline import SingleStrandDNATissueCut


def cell_seg(
        model_path: str,
        img_path: str,
        out_path: str,
        deep_crop_size: int = 20000,
        overlap: int = 100,
        post_processing_workers: int = 10,
        is_water: bool = False,
        method: str = 'v3',
        need_tissue_cut=True,
        tissue_seg_model_path: str = None,
        tissue_seg_staining_type: str = None,
        gpu: str = '-1',
        num_threads: int = -1
):
    """
    Cell segmentation on regist.tif/mask.tif by deeplearning model.

    :param model_path: the path of model used to cell segmentation.
    :param img_path: the path of regist.tif/mask.tif.
    :param out_path: the path of directory to save the result cell mask.tif.
    :param deep_crop_size: deep crop size, defaults to 20000
    :param overlap: over lap size, defaults to 100
    :param post_processing_workers: the number of processes on post processing, defaults to 10.
    :param is_water: defaults to False.
    :param method: v1, v1_pro or v3, recommend to use v3.
    :param need_tissue_cut: whether to run tissue segmentation, defaults to True.
                            the method v1 and v1_pro have to run tissue segmentation, so these two methods must be based on regist.tif,
                            method v3 can use mask.tif from tissue segmentation or regist.tif without tissue segmentation.
    :param tissue_seg_model_path: the path of model used to tissue segmentation, defaults to None
    :param tissue_seg_staining_type: the staining type of regist.mask, defaults to None
    :param gpu: the gpu on which the model works, available for both cell segmtation and tissue segmtation, '-1' means working on cpu.
    :param num_threads: the number of threads when model work on cpu, 
                        available for both v3 cell segmtation and tissue segmtation, -1 means using all the cores.

    """
    if method not in VersionType.get_version_list():
        raise Exception("version must be %s" % ('、'.join(VersionType.get_version_list())))
    
    tissue_mask = None
    if method in ('v1', 'v1_pro'):
        need_tissue_cut = True
    if need_tissue_cut:
        tissue_seg = SingleStrandDNATissueCut(
            src_img_path=img_path,
            dst_img_path=out_path,
            model_path=tissue_seg_model_path,
            staining_type=tissue_seg_staining_type,
            gpu=gpu,
            num_threads=num_threads
        )
        tissue_seg.tissue_seg()
        tissue_mask = tissue_seg.mask

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if method == VersionType.v1.value:
        cell_seg_pipeline = CellSegPipeV1(
            model_path,
            img_path,
            out_path,
            is_water,
            deep_crop_size,
            overlap,
            gpu=gpu,
            post_processing_workers=post_processing_workers,
            tissue_mask=tissue_mask
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
            post_processing_workers=post_processing_workers,
            tissue_mask=tissue_mask,
            num_threads=num_threads
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
            post_processing_workers=post_processing_workers,
            tissue_mask=tissue_mask
        )
        cell_seg_pipeline.run()
