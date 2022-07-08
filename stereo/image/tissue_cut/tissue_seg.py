#!/usr/bin/env python
#coding: utf-8
###### Import Modules ##########
import argparse
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waterSeg'))
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postProcess'))
# print(sys.path)
import time
import numpy as np
import tissueCut_utils.tissue_seg_pipeline as pipeline


#########################################################
#########################################################
# tissue segmentation
#########################################################
#########################################################


usage = '''
     limin  %s
     Usage: %s imagePath outPath imageType(1:ssdna; 0:RNA)  method(1:deep; 0:other)
''' % ('2021-07-15', os.path.basename(sys.argv[0]))


def args_parse():
    ap = argparse.ArgumentParser(usage=usage)
    ap.add_argument('-i', '--img_path', action='store', help='image path')
    ap.add_argument('-o', '--out_path', action='store',  help='mask path')
    ap.add_argument('-t', '--img_type', dest='img_type', type=int, help='ssdna:1; rna:0', default=1)
    ap.add_argument('-m', '--seg_method', dest='seg_method', type=int, help='deep:1; intensity:0', default=1)
    return ap.parse_args()


def tissueSeg(img_path, out_path, type, deep, conf=''):
    cell_seg_pipeline = pipeline.tissueCut(img_path, out_path, type, deep, conf)
    cell_seg_pipeline.tissue_seg()
    # ref = cell_seg_pipeline.tissue_seg()
    # return ref


def tissue_segment_entry(args):
    args = vars(args)
    img_path = args['img_path']
    out_path = args['out_path']
    type = args['img_type']
    deep = args['seg_method']

    t0 = time.time()
    ref = tissueSeg(img_path, out_path, type, deep)
    t1 = time.time()
    print('running time:', t1 - t0)



def main():
    ######################### Phrase parameters #########################
    args = args_parse()
    print(args)

    # call segmentation
    tissue_segment_entry(args)

if __name__ == '__main__':
    main()
