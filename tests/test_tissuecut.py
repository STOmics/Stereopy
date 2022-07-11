#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_tissuecut.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/12/21  create file.
"""
from stereo import image as im
import time


# FIXME: this function can not use after merge the new version of TissueCut
def test_rna(deep=0):
    print('start')
    t = time.time()
    img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/demoData/rna/SS200000139BL_F2.tif'
    out_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/demoData/res_rna'
    model_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/tissue_cut_1229/tissueCut_model/ssdna_seg.pth'
    backbone_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/tissue_cut_1229/tissueCut_model/backbone.pth'
    im.tissue_seg(img_path=img_path, out_path=out_path, type=0, deep=deep, model_path=model_path, backbone_path=backbone_path)
    print(time.time()-t)
    print('end')


# FIXME: this function can not use after merge the new version of TissueCut
def test_ssdna(deep=0):
    print('start')
    t = time.time()
    img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/demoData/ssdna/SS200000130BR_A2.tiff'
    out_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/demoData/res_ssdna'
    model_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/tissue_cut_1229/tissueCut_model/ssdna_seg.pth'
    backbone_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/tissue_cut_1229/tissueCut_model/backbone.pth'
    im.tissue_seg(img_path=img_path, out_path=out_path, type=1, deep=deep, model_path=model_path, backbone_path=backbone_path)
    print(time.time()-t)
    print('end')


if __name__ == '__main__':
    test_rna(1)
    test_ssdna(1)
