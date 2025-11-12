# import image
import os
import time
from os.path import (
    join,
    splitext,
    exists,
    split
)

import cv2
import numpy as np
import tifffile
from skimage import measure

# from stereo.image.tissue_cut import (
#     SingleStrandDNATissueCut,
#     DEEP,
#     INTENSITY
# )
from stereo.log_manager import logger
from . import cell_infer as cell_infer
from . import grade as grade
from . import utils as utils


class CellSegPipe(object):

    def __init__(
            self,
            img_path,
            out_path,
            is_water,
            DEEP_CROP_SIZE=20000,
            OVERLAP=100,
            model_path=None,
            # tissue_seg_model_path='',
            # tissue_seg_method=DEEP,
            post_processing_workers=10,
            tissue_mask=None
    ):
        self.deep_crop_size = DEEP_CROP_SIZE
        self.overlap = OVERLAP
        self.__img_path = img_path
        if os.path.isdir(img_path):
            self.__file = os.listdir(img_path)
            self.__is_list = True
        else:
            self.__file = [split(img_path)[-1]]
            self.__is_list = False
        self.__file_name = [splitext(file)[0] for file in self.__file]
        self.__img_suffix = [splitext(file)[-1] for file in self.__file]
        self.img_list = self.__imload_list(img_path)
        self.__convert_gray()
        self.__out_path = out_path
        if not exists(out_path):
            os.makedirs(out_path)
            logger.info('Create new dir : %s' % out_path)
        self.__is_water = is_water
        t0 = time.time()
        self.__trans16to8()
        t1 = time.time()
        logger.info('Transform 16bit to 8bit : %.2f' % (t1 - t0))
        self.tissue_mask = tissue_mask
        self.tissue_mask_thumb = []
        self.tissue_num = []  # tissue num in each image
        self.tissue_bbox = []  # tissue roi bbox in each image
        self.img_filter = []  # image filtered by tissue mask
        # self.__get_tissue_mask(tissue_seg_model_path, tissue_seg_method)
        t2 = time.time()
        logger.info('Get tissue mask : %.2f' % (t2 - t1))
        self.__get_img_filter()
        self.__get_roi()
        # self.save_tissue_mask()
        self.cell_mask = []
        self.post_mask_list = []
        self.score_mask_list = []
        self.model_path = model_path
        self.post_processing_workers = post_processing_workers

    def __imload_list(self, img_path):

        if self.__is_list:
            img_list = []
            for idx, file in enumerate(self.__file):
                img_temp = self.__imload(join(img_path, file), idx)
                img_list.append(img_temp)
            return img_list
        else:
            img_temp = self.__imload(img_path, 0)
            return [img_temp]

    def __imload(self, img_path, id):

        assert self.__img_suffix[id] in ['.tif', '.png', '.jpg']
        if self.__img_suffix[id] == '.tif':
            img = tifffile.imread(img_path)
        else:
            img = cv2.imread(img_path, -1)
        return img

    def __convert_gray(self):

        for idx, img in enumerate(self.img_list):
            if len(img.shape) == 3:
                logger.info('Image %s convert to gray!' % self.__file[idx])
                self.img_list[idx] = img[:, :, 0]

    def __trans16to8(self):
        for idx, img in enumerate(self.img_list):
            assert img.dtype in ['uint16', 'uint8']
            if img.dtype != 'uint8':
                logger.info('%s transfer to 8bit' % self.__file[idx])
                self.img_list[idx] = utils.transfer_16bit_to_8bit(img)

    # def __get_tissue_mask(self):

    #     process = 15 if self.__is_list else 1

    #     pre_tissue = tissue_seg.tissue_seg_multi(self.img_list, process)
    #     self.tissue_mask = [label[0] for label in pre_tissue]
    #     self.tissue_mask_thumb = [label[1] for label in pre_tissue]

    # def __get_tissue_mask(self, tissue_seg_model_path, tissue_seg_method):
    #     if tissue_seg_method is None:
    #         tissue_seg_method = DEEP
    #     if tissue_seg_model_path is None or len(tissue_seg_model_path) == 0:
    #         tissue_seg_method = INTENSITY
    #     ssDNA_tissue_cut = SingleStrandDNATissueCut(
    #         src_img_path=self.__img_path,
    #         model_path=tissue_seg_model_path,
    #         dst_img_path=self.__out_path,
    #         seg_method=tissue_seg_method
    #     )
    #     ssDNA_tissue_cut.tissue_seg()
    #     self.tissue_mask = ssDNA_tissue_cut.mask
        # self.tissue_mask_thumb = ssDNA_tissue_cut.mask_thumb

    def __get_img_filter(self):
        """get tissue image by tissue mask"""
        # for idx, img in enumerate(self.img_list):
        for img, tissue_mask in zip(self.img_list, self.tissue_mask):
            img_filter = np.multiply(img, tissue_mask).astype(np.uint8)
            self.img_filter.append(img_filter)

    def __filter_roi(self, props):
        filtered_props = []
        for id, p in enumerate(props):
            black = np.sum(p['intensity_image'] == 0)
            sum = p['bbox_area']
            ratio_black = black / sum
            pixel_light_sum = np.sum(np.unique(p['intensity_image']) > 128)
            if ratio_black < 0.75 and pixel_light_sum > 10:
                filtered_props.append(p)
        return filtered_props

    def __get_roi(self):

        """get tissue area from ssdna"""
        for idx, tissue_mask in enumerate(self.tissue_mask):

            label_image = measure.label(tissue_mask, connectivity=2)
            props = measure.regionprops(label_image, intensity_image=self.img_list[idx])

            # remove noise tissue mask
            filtered_props = self.__filter_roi(props)
            if len(props) != len(filtered_props):
                tissue_mask_filter = np.zeros((tissue_mask.shape), dtype=np.uint8)
                for tissue_tile in filtered_props:
                    bbox = tissue_tile['bbox']
                    tissue_mask_filter[bbox[0]: bbox[2], bbox[1]: bbox[3]] += tissue_tile['image']
                self.tissue_mask[idx] = np.uint8(tissue_mask_filter > 0)
            self.tissue_num.append(len(filtered_props))
            self.tissue_bbox.append([p['bbox'] for p in filtered_props])

    def tissue_cell_infer(self):

        """cell segmentation in tissue area by neural network"""
        tissue_cell_label = []
        for img, tissue_bbox in zip(self.img_filter, self.tissue_bbox):
            tissue_img = [img[p[0]: p[2], p[1]: p[3]] for p in tissue_bbox]

            label_list = cell_infer.cellInfer(tissue_img, self.deep_crop_size, self.overlap, self.model_path)
            tissue_cell_label.append(label_list)
        return tissue_cell_label

    def tissue_label_filter(self, tissue_cell_label):

        """filter cell mask in tissue area"""
        tissue_cell_label_filter = []
        for idx, label in enumerate(tissue_cell_label):
            tissue_bbox = self.tissue_bbox[idx]
            label_filter_list = []
            for i in range(self.tissue_num[idx]):
                tissue_bbox_temp = tissue_bbox[i]
                label_filter = np.multiply(
                    label[i],
                    self.tissue_mask[idx][tissue_bbox_temp[0]: tissue_bbox_temp[2],
                    tissue_bbox_temp[1]: tissue_bbox_temp[3]]).astype(np.uint8)  # noqa
                label_filter_list.append(label_filter)
            tissue_cell_label_filter.append(label_filter_list)
        return tissue_cell_label_filter

    def __mosaic(self, tissue_cell_label_filter):

        """mosaic tissue into original mask"""
        for idx, label_list in enumerate(tissue_cell_label_filter):
            tissue_bbox = self.tissue_bbox[idx]
            cell_mask = np.zeros((self.img_list[idx].shape), dtype=np.uint8)
            for i in range(self.tissue_num[idx]):
                tissue_bbox_temp = tissue_bbox[i]
                cell_mask[tissue_bbox_temp[0]: tissue_bbox_temp[2],
                tissue_bbox_temp[1]: tissue_bbox_temp[3]] = label_list[i]  # noqa
            self.cell_mask.append(cell_mask)
        return self.cell_mask

    def watershed_score(self, cell_mask):
        """watershed and score on cell mask by neural network"""
        for idx, cell_mask in enumerate(cell_mask):
            cell_mask_tile, x_list, y_list = utils.split(cell_mask, self.deep_crop_size)
            img_tile, _, _ = utils.split(self.img_list[idx], self.deep_crop_size)

            input_list = [[cell_mask_tile[id], img] for id, img in enumerate(img_tile)]
            if self.__is_water:
                post_list_tile = grade.watershed_multi(input_list, self.post_processing_workers)
            else:
                post_list_tile = grade.score_multi(input_list, self.post_processing_workers)

            post_mask_tile = [label[0] for label in post_list_tile]
            score_mask_tile = [label[1] for label in post_list_tile]  # grade saved

            post_mask = utils.merge(post_mask_tile, x_list, y_list, cell_mask.shape)
            score_mask = utils.merge(score_mask_tile, x_list, y_list, cell_mask.shape)
            self.post_mask_list.append(post_mask)
            self.score_mask_list.append(score_mask)

    def save_tissue_mask(self):
        for idx, tissue_thumb in enumerate(self.tissue_mask_thumb):
            tifffile.imsave(join(self.__out_path, self.__file_name[idx] + r'_tissue_cut.tif'), tissue_thumb)

    def __mkdir_subpkg(self):
        """make new dir while image is large"""
        assert self.__is_list == False  # noqa
        file = self.__file[0]
        file_name = splitext(file)[0]

        mask_outline_name = r'_watershed_outline' if self.__is_water else r'_outline'
        mask_name = r'_watershed' if self.__is_water else r''

        self.__subpkg_mask = join(self.__out_path, file_name + mask_name)
        self.__subpkg_mask_outline = join(self.__out_path, file_name + mask_outline_name)
        self.__subpkg_score = join(self.__out_path, file_name + r'_score')

        if not os.path.exists(self.__subpkg_mask):
            os.mkdir(self.__subpkg_mask)

        if not os.path.exists(self.__subpkg_mask_outline):
            os.mkdir(self.__subpkg_mask_outline)

        if not os.path.exists(self.__subpkg_score):
            os.mkdir(self.__subpkg_score)

    def __save_each_file_result(self, file_name, idx):
        mask_outline_name = r'_watershed_outline.tif' if self.__is_water else r'_outline.tif'
        mask_name = r'_watershed_mask.tif' if self.__is_water else r'_mask.tif'

        tifffile.imsave(join(self.__out_path, file_name + r'_score.tif'),
                        self.score_mask_list[idx])
        tifffile.imsave(join(self.__out_path, file_name + mask_outline_name),
                        utils.outline(self.post_mask_list[idx]))
        tifffile.imsave(join(self.__out_path, file_name + mask_name),
                        self.post_mask_list[idx])

    def save_cell_mask(self):
        """save cell mask from network or watershed"""
        for idx, file in enumerate(self.__file):
            file_name, _ = os.path.splitext(file)
            self.__save_each_file_result(file_name, idx)

        if not self.__is_list:
            self.__mkdir_subpkg()
            mask_list, x_list, y_list = utils.split(self.post_mask_list[0], self.deep_crop_size)
            mask_list_outline = map(utils.outline, mask_list)
            mask_list_outline = [mask for mask in mask_list_outline]
            score_list, _, _ = utils.split(self.score_mask_list[0], self.deep_crop_size)
            for idx, img in enumerate(mask_list):
                shapes = self.img_list[0].shape
                tifffile.imsave(
                    os.path.join(self.__subpkg_mask,
                                 self.__file_name[0] + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' +
                                 str(x_list[idx]) + '_' + str(y_list[idx]) + '.tif'), img)
                tifffile.imsave(
                    os.path.join(self.__subpkg_mask_outline,
                                 self.__file_name[0] + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' +
                                 str(x_list[idx]) + '_' + str(y_list[idx]) + '.tif'), mask_list_outline[idx])
                tifffile.imsave(
                    os.path.join(self.__subpkg_score,
                                 self.__file_name[0] + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' +
                                 str(x_list[idx]) + '_' + str(y_list[idx]) + '.tif'), score_list[idx])

    def save_result(self):
        """save tissue mask"""
        self.save_cell_mask()

    def run(self):
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
        cell_mask = self.__mosaic(tissue_cell_label_filter)
        t3 = time.time()
        logger.info('Mosaic tissue roi : %.2f' % (t3 - t2))

        # post process
        self.watershed_score(cell_mask)
        t4 = time.time()
        logger.info('Post-processing : %.2f' % (t4 - t3))

        self.save_result()
        logger.info('Result saved : %s ' % (self.__out_path))
