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


def test_tissuecut():
    print('start')
    t = time.time()
    img_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/demoData/SS200000130BR_A2.tiff'  # ssdna
    out_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/demoData/result1'
    model_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/tissueCut_model/ssdna_seg.pth'
    backbone_path = '/ldfssz1/ST_BI/USER/stereopy/workspace/st/models/tissuecut/tissueCut_model/backbone.pth'
    im.tissue_seg(img_path=img_path, out_path=out_path, type=1, deep=1, model_path=model_path, backbone_path=backbone_path)
    print(time.time()-t)
    print('end')


if __name__ == '__main__':
    test_tissuecut()
