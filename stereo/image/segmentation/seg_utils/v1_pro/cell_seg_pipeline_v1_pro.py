# import image
import os
import time
from os.path import join

import numpy as np
import tifffile
from skimage import measure

from stereo.image.segmentation.seg_utils.base_cell_seg_pipe.cell_seg_pipeline import CellSegPipe
from stereo.image.segmentation.seg_utils.v1_pro import grade
from stereo.log_manager import logger
from .cell_infer import cellInfer
from .utils import transfer_16bit_to_8bit


class CellSegPipeV1Pro(CellSegPipe):

    def tissue_cell_infer(self):
        """cell segmentation in tissue area by neural network"""
        self.tissue_cell_label = []
        for idx, img in enumerate(self.img_list):
            tissue_bbox = self.tissue_bbox[idx]
            tissue_img = [img[p[0]: p[2], p[1]: p[3]] for p in tissue_bbox]
            label_list = cellInfer(tissue_img, self.deep_crop_size, self.overlap)
            self.tissue_cell_label.append(label_list)
        return 0

    def tissue_label_filter(self, tissue_cell_label):
        """filter cell mask in tissue area"""
        tissue_cell_label_filter = []
        for idx, label in enumerate(tissue_cell_label):
            tissue_bbox = self.tissue_bbox[idx]
            label_filter_list = []
            for i in range(self.tissue_num[idx]):
                if len(self.tissue_mask) != 0:
                    tiss_bbox_tep = tissue_bbox[i]
                    label_filter = np.multiply(
                        label[i],
                        self.tissue_mask[idx][tiss_bbox_tep[0]: tiss_bbox_tep[2], tiss_bbox_tep[1]: tiss_bbox_tep[3]]
                    ).astype(np.uint8)
                    label_filter_list.append(label_filter)
                else:
                    label_filter_list.append(label[i])
            tissue_cell_label_filter.append(label_filter_list)
        return tissue_cell_label_filter

    def run(self):
        logger.info('Start do cell mask, this will take some minutes.')
        t1 = time.time()

        self.tissue_cell_infer()
        t2 = time.time()
        logger.info('Cell inference : %.2f' % (t2 - t1))

        # filter by tissue mask
        tissue_cell_label_filter = self.tissue_label_filter(self.tissue_cell_label)
        t3 = time.time()
        logger.info('Filter by tissue mask : %.2f' % (t3 - t2))

        # mosaic tissue roi
        cell_mask = self.mosaic(tissue_cell_label_filter)
        del tissue_cell_label_filter
        t4 = time.time()
        logger.info('Mosaic tissue roi : %.2f' % (t4 - t3))

        # post process
        self.watershed_score(cell_mask)
        t5 = time.time()
        logger.info('Post-processing : %.2f' % (t5 - t4))

        self.save_cell_mask()
        logger.info('Result saved : %s ' % (self.out_path))

    def save_each_file_result(self, file_name, idx):
        mask_name = r'_watershed_mask.tif' if self.is_water else r'_mask.tif'
        tifffile.imsave(join(self.out_path, file_name + mask_name), self.post_mask_list[idx])

    def save_cell_mask(self):
        """save cell mask from network or watershed"""
        for idx, file in enumerate(self.file):
            file_name, _ = os.path.splitext(file)
            self.save_each_file_result(file_name, idx)

    def watershed_score(self, cell_mask):
        """watershed and score on cell mask by neural network"""
        for idx, cell_mask in enumerate(cell_mask):
            post_mask = grade.edgeSmooth(cell_mask)
        self.post_mask_list.append(post_mask)

    def get_roi(self):
        if len(self.tissue_mask) == 0:
            self.tissue_num.append(1)
            self.tissue_bbox.append([(0, 0, self.img_list[0].shape[0], self.img_list[0].shape[1])])
        else:
            for idx, tissue_mask in enumerate(self.tissue_mask):
                label_image = measure.label(tissue_mask, connectivity=2)
                props = measure.regionprops(label_image, intensity_image=self.img_list[idx])

                # remove noise tissue mask
                filtered_props = props
                if len(props) != len(filtered_props):
                    tissue_mask_filter = np.zeros((tissue_mask.shape), dtype=np.uint8)
                    for tissue_tile in filtered_props:
                        bbox = tissue_tile['bbox']
                        tissue_mask_filter[bbox[0]: bbox[2], bbox[1]: bbox[3]] += tissue_tile['image']
                    self.tissue_mask[idx] = np.uint8(tissue_mask_filter > 0)
                self.tissue_num.append(len(filtered_props))
                self.tissue_bbox.append([p['bbox'] for p in filtered_props])

    def trans16to8(self):
        for idx, img in enumerate(self.img_list):
            assert img.dtype in ['uint16', 'uint8']
            if img.dtype != 'uint8':
                logger.info('%s transfer to 8bit' % self.file[idx])
                self.img_list[idx] = transfer_16bit_to_8bit(img)

    def get_tissue_mask(self, tissue_seg_model_path, tissue_seg_method):
        try:
            self.tissue_mask = [tifffile.imread(os.path.join(self.out_path, self.file_name[0] + '_tissue_cut.tif'))]
        except Exception:
            self.tissue_mask = []
