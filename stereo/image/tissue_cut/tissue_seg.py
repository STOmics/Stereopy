#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tissueCut_utils.tissue_seg_pipeline as pipeline


def tissue_seg(img_path, out_path, type, deep, model_path, backbone_path):
    """
    tissueCut function entry.

    :param img_path: image input path
    :param out_path: output path
    :param type: img type, ssdna:1; rna:0
    :param deep: function method, deep:1; intensity:0
    :param model_path: model path
    :param backbone_path: model backbone weight path
    :return:
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    cell_seg_pipeline = pipeline.tissueCut(img_path, out_path, type, deep, model_path, backbone_path)
    ref = cell_seg_pipeline.tissue_seg()
    return ref

