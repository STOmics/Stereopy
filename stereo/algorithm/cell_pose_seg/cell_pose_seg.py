#!/usr/bin/env python3
# coding: utf-8
"""
@author: zhen bin  wenzhenbin@genomics.cn
@last modified by: zhen bin
@file:common.py
@time:2023/08/24
"""
import cv2
import patchify
import numpy as np
from math import ceil
from typing import Optional

from scipy.ndimage import distance_transform_edt
from stereo.algorithm.cell_pose_seg import utils
from stereo.algorithm.cell_pose_seg import models
from stereo.utils.time_consume import log_consumed_time


class CellSegmentation:

    def __init__(self,
                 open_path: str,
                 save_path: str,
                 photo_size: Optional[int] = 2048,
                 photo_step: Optional[int] = 2000,
                 dmin: Optional[int] = 10,
                 dmax: Optional[int] = 40,
                 step: Optional[int] = 10):
        self.open_path = open_path
        self.save_path = save_path
        self.photo_size = photo_size
        self.photo_step = photo_step
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.segment_cells()

    @log_consumed_time
    def _process_image(self, img_data):
        overlap = self.photo_size - self.photo_step
        if (overlap % 2) == 1:
            overlap = overlap + 1
        act_step = ceil(overlap / 2)
        im = cv2.imread(self.open_path)
        image = np.array(im)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res_image = np.pad(gray_image, ((act_step, act_step), (act_step, act_step)), 'constant')
        a = res_image.shape[0]
        b = res_image.shape[1]
        res_a = ceil((a - self.photo_size) / self.photo_step) * self.photo_step + self.photo_size
        res_b = ceil((b - self.photo_size) / self.photo_step) * self.photo_step + self.photo_size
        padding_rows = res_a - a
        padding_cols = res_b - b
        regray_image = np.pad(res_image, ((0, padding_rows), (0, padding_cols)), mode='constant')
        patches = patchify.patchify(regray_image, (self.photo_size, self.photo_size), step=self.photo_step)
        wid = patches.shape[0]
        high = patches.shape[1]
        model = models.Cellpose(gpu=True, model_type='cyto2')
        a_patches = np.full((wid, high, (self.photo_step), (self.photo_step)), 255)
        for i in range(wid):
            for j in range(high):
                img_data = patches[i, j, :, :]
                num0min = wid * high * 800000000000000
                for k in range(self.dmin, self.dmax, self.step):

                    masks, flows, styles, diams = model.eval(img_data, diameter=k, channels=[0, 0],
                                                             flow_threshold=0.9)
                    num0 = np.sum(masks == 0)

                    if num0 < num0min:
                        num0min = num0
                        outlines = utils.masks_to_outlines(masks)
                        outlines = (outlines == True).astype(int) * 255

                        try:
                            a_patches[i, j, :, :] = outlines[act_step:(self.photo_step + act_step),
                                                    act_step:(self.photo_step + act_step)]
                            output = masks.copy()
                        except:
                            a_patches[i, j, :, :] = output[act_step:(self.photo_step + act_step),
                                                    act_step:(self.photo_step + act_step)]
        patch_nor = patchify.unpatchify(a_patches, ((wid) * (self.photo_step), (high) * (self.photo_step)))
        nor_imgdata = np.array(patch_nor)
        cropped_1 = nor_imgdata[0:gray_image.shape[0], 0:gray_image.shape[1]]
        cropped_1 = np.uint8(cropped_1)
        return cropped_1

    @log_consumed_time
    def _post_image(self, process_image):
        contour_thickness = 0
        contour_coords = np.argwhere(process_image == 255)
        distance_transform = distance_transform_edt(process_image == 0)
        expanded_image = np.zeros_like(process_image)
        for y, x in contour_coords:
            mask = distance_transform[y, x] <= contour_thickness
            expanded_image[y - contour_thickness:y + contour_thickness + 1,
            x - contour_thickness:x + contour_thickness + 1] = mask * 255
        contours, _ = cv2.findContours(expanded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        height, width = process_image.shape
        black_background = np.zeros((height, width), dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 10000:
                cv2.drawContours(black_background, [contour], -1, 255, thickness=cv2.FILLED)
        black_background = np.uint8(black_background)
        return black_background, expanded_image

    def _merger_image(self, merger_image1, merger_image2):
        merger_image1[merger_image2 == 255] = 0
        return merger_image1

    def segment_cells(self):
        inverted_image = self._process_image(self.open_path)
        post_image, expanded_image = self._post_image(inverted_image)
        result_image = self._merger_image(post_image, expanded_image)
        cv2.imwrite(self.save_path, result_image)
