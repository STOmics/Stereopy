import os
import time
from os.path import join

import numpy as np
import tifffile
from skimage import measure

from stereo.image.segmentation.seg_utils.base_cell_seg_pipe import grade
from stereo.image.segmentation.seg_utils.base_cell_seg_pipe.cell_seg_pipeline import CellSegPipe
from stereo.image.segmentation.seg_utils.v1 import (
    utils,
    cell_infer
)


class CellSegPipeV1(CellSegPipe):

    def save_each_file_result(self, file_name, idx):
        mask_outline_name = r'_watershed_outline.tif' if self.is_water else r'_outline.tif'
        mask_name = r'_watershed_mask.tif' if self.is_water else r'_mask.tif'

        tifffile.imsave(join(self.out_path, file_name + r'_score.tif'), self.score_mask_list[idx])
        tifffile.imsave(join(self.out_path, file_name + mask_outline_name), utils.outline(self.post_mask_list[idx]))
        tifffile.imsave(join(self.out_path, file_name + mask_name), self.post_mask_list[idx])

    def save_cell_mask(self):
        """save cell mask from network or watershed"""
        for idx, file in enumerate(self.file):
            file_name, _ = os.path.splitext(file)
            self.save_each_file_result(file_name, idx)

        if not self.is_list:
            self.mkdir_subpkg()
            mask_list, x_list, y_list, _, _ = utils.split(self.post_mask_list[0], self.deep_crop_size)
            mask_list_outline = map(utils.outline, mask_list)
            mask_list_outline = [mask for mask in mask_list_outline]
            score_list, _, _, _, _ = utils.split(self.score_mask_list[0], self.deep_crop_size)
            for idx, img in enumerate(mask_list):
                shapes = self.img_list[0].shape
                tifffile.imsave(
                    os.path.join(self.subpkg_mask,
                                 self.file_name[0] + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' +
                                 str(x_list[idx]) + '_' + str(y_list[idx]) + '.tif'), img)
                tifffile.imsave(
                    os.path.join(self.subpkg_mask_outline,
                                 self.file_name[0] + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' +
                                 str(x_list[idx]) + '_' + str(y_list[idx]) + '.tif'), mask_list_outline[idx])
                tifffile.imsave(
                    os.path.join(self.subpkg_score,
                                 self.file_name[0] + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' +
                                 str(x_list[idx]) + '_' + str(y_list[idx]) + '.tif'), score_list[idx])

    def get_roi(self):
        for idx, tissue_mask in enumerate(self.tissue_mask):
            label_image = measure.label(tissue_mask, connectivity=2)
            props = measure.regionprops(label_image, intensity_image=self.img_list[idx])

            # remove noise tissue mask
            filtered_props = self.filter_roi(props)
            if len(props) != len(filtered_props):
                tissue_mask_filter = np.zeros((tissue_mask.shape), dtype=np.uint8)
                for tissue_tile in filtered_props:
                    bbox = tissue_tile['bbox']
                    tissue_mask_filter[bbox[0]: bbox[2], bbox[1]: bbox[3]] += tissue_tile['image']
                self.tissue_mask[idx] = np.uint8(tissue_mask_filter > 0)
            self.tissue_num.append(len(filtered_props))
            self.tissue_bbox.append([p['bbox'] for p in filtered_props])

    def tissue_cell_infer(self, q=None):
        """cell segmentation in tissue area by neural network"""
        tissue_cell_label = []
        for img, tissue_bbox in zip(self.img_filter, self.tissue_bbox):
            tissue_img = [img[p[0]: p[2], p[1]: p[3]] for p in tissue_bbox]
            label_list = cell_infer.cellInfer(self.model_path, tissue_img, self.deep_crop_size, self.overlap, self.gpu)
            tissue_cell_label.append(label_list)
        if q is not None:
            q.put(tissue_cell_label)
        return tissue_cell_label

    def watershed_score(self, cell_mask):
        """watershed and score on cell mask by neural network"""
        for idx, cell_mask in enumerate(cell_mask):
            cell_mask = np.squeeze(cell_mask)
            cell_mask_tile, x_list, y_list, mask_width_add, mask_height_add = utils.split(
                cell_mask, self.deep_crop_size)
            img_tile, _, _, _, _ = utils.split(self.img_list[idx], self.deep_crop_size)
            input_list = [[cell_mask_tile[id], img] for id, img in enumerate(img_tile)]
            if self.is_water:
                post_list_tile = grade.watershed_multi(input_list, self.post_processing_workers)
            else:
                post_list_tile = grade.score_multi(input_list, self.post_processing_workers)

            post_mask_tile = [label[0] for label in post_list_tile]
            score_mask_tile = [label[1] for label in post_list_tile]  # grade saved
            post_mask = utils.merge(post_mask_tile, x_list, y_list, cell_mask.shape, width_add=mask_width_add,
                                    height_add=mask_height_add)
            score_mask = utils.merge(score_mask_tile, x_list, y_list, cell_mask.shape, width_add=mask_width_add,
                                     height_add=mask_height_add)
            self.post_mask_list.append(post_mask)
            self.score_mask_list.append(score_mask)

    def __get_img_filter(self):
        """get tissue image by tissue mask"""
        for img, tissue_mask in zip(self.img_list, self.tissue_mask):
            img_filter = np.multiply(img, tissue_mask).astype(np.uint8)
            self.img_filter.append(img_filter)

    def run(self):
        from stereo.log_manager import logger
        self.__get_img_filter()

        t0 = time.time()
        # cell segmentation in roi
        tissue_cell_label = self.tissue_cell_infer()
        t1 = time.time()
        logger.info('Cell inference : %.2f' % (t1 - t0))

        # filter by tissue mask
        tissue_cell_label_filter = self.tissue_label_filter(tissue_cell_label)
        t2 = time.time()
        logger.info('Filter by tissue mask : %.2f' % (t2 - t1))

        # mosaic tissue roi
        cell_mask = self.mosaic(tissue_cell_label_filter)
        t3 = time.time()
        logger.info('Mosaic tissue roi : %.2f' % (t3 - t2))

        # post process
        self.watershed_score(cell_mask)
        t4 = time.time()
        logger.info('Post-processing : %.2f' % (t4 - t3))

        self.save_cell_mask()
        logger.info('Result saved : %s ' % (self.out_path))
