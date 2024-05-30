# import image
import os
import time
from abc import abstractmethod
from os.path import (
    join,
    splitext,
    exists,
    split
)

import cv2
import numpy as np
import tifffile

# from stereo.image.tissue_cut import (
#     DEEP,
#     INTENSITY,
#     SingleStrandDNATissueCut
# )
from stereo.log_manager import logger


class CellSegPipe(object):

    def __init__(
            self,
            model_path,
            img_path,
            out_path,
            is_water,
            DEEP_CROP_SIZE=20000,
            OVERLAP=100,
            # tissue_seg_model_path='',
            # tissue_seg_method=DEEP,
            post_processing_workers=10,
            tissue_mask=None,
            gpu='-1',
            *args,
            **kwargs
    ):
        self.deep_crop_size = DEEP_CROP_SIZE
        self.overlap = OVERLAP
        self.model_path = model_path
        self.img_path = img_path
        if os.path.isdir(img_path):
            self.file = os.listdir(img_path)
            self.is_list = True
        else:
            self.file = [split(img_path)[-1]]
            self.is_list = False
        self.file_name = [splitext(file)[0] for file in self.file]
        self.img_suffix = [splitext(file)[-1] for file in self.file]
        self.img_list = self.__imload_list(img_path)
        self.convert_gray()
        self.out_path = out_path
        if not exists(out_path):
            os.makedirs(out_path)
            logger.info('Create new dir : %s' % out_path)
        self.is_water = is_water
        t0 = time.time()
        self.trans16to8()
        t1 = time.time()
        logger.info('Transform 16bit to 8bit : %.2f' % (t1 - t0))
        self.tissue_mask = tissue_mask
        self.tissue_mask_thumb = []
        # self.tissue_seg_model_path = tissue_seg_model_path
        # self.tissue_seg_method = tissue_seg_method
        self.tissue_num = []  # tissue num in each image
        self.tissue_bbox = []  # tissue roi bbox in each image
        self.img_filter = []  # image filtered by tissue mask
        # self.get_tissue_mask()
        self.get_roi()
        self.cell_mask = []
        self.post_mask_list = []
        self.score_mask_list = []
        self.post_processing_workers = post_processing_workers
        self.tissue_cell_label = None
        self.gpu = gpu
        self.args = args
        self.kwargs = kwargs

    def __imload_list(self, img_path):
        if self.is_list:
            img_list = []
            for idx, file in enumerate(self.file):
                img_temp = self.__imload(join(img_path, file), idx)
                img_list.append(img_temp)
            return img_list
        else:
            img_temp = self.__imload(img_path, 0)
            return [img_temp]

    def __imload(self, img_path, id):
        assert self.img_suffix[id] in ['.tif', '.png', '.jpg']
        if self.img_suffix[id] == '.tif':
            img = tifffile.imread(img_path)
        else:
            img = cv2.imread(img_path, -1)
        return img

    def convert_gray(self):
        for idx, img in enumerate(self.img_list):
            if len(img.shape) == 3:
                logger.info('Image %s convert to gray!' % self.file[idx])
                self.img_list[idx] = img[:, :, 0]

    def get_img_filter(self):
        """get tissue image by tissue mask"""
        for img, tissue_mask in zip(self.img_list, self.tissue_mask):
            img_filter = np.multiply(img, tissue_mask).astype(np.uint8)
            self.img_filter.append(img_filter)

    @staticmethod
    def transfer_32bit_to_8bit(image_32bit):
        min_32bit = np.min(image_32bit)
        max_32bit = np.max(image_32bit)
        return np.array(np.rint(255 * ((image_32bit - min_32bit) / (max_32bit - min_32bit))), dtype=np.uint8)

    @staticmethod
    def transfer_16bit_to_8bit(image_16bit):
        min_16bit = np.min(image_16bit)
        max_16bit = np.max(image_16bit)
        return np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

    def trans16to8(self):
        from stereo.log_manager import logger
        for idx, img in enumerate(self.img_list):
            assert img.dtype in ['uint16', 'uint8']
            if img.dtype != 'uint8':
                logger.info('%s transfer to 8bit' % self.file[idx])
                self.img_list[idx] = self.transfer_16bit_to_8bit(img)

    def save_each_file_result(self, file_name, idx):
        pass

    # def get_tissue_mask(self):
    #     tissue_seg_model_path = self.tissue_seg_model_path
    #     tissue_seg_method = self.tissue_seg_method
    #     if tissue_seg_method is None:
    #         tissue_seg_method = DEEP
    #     if not tissue_seg_model_path or len(tissue_seg_model_path) == 0:
    #         tissue_seg_method = INTENSITY
    #     ss_dna_tissue_cut = SingleStrandDNATissueCut(
    #         src_img_path=self.img_path,
    #         model_path=tissue_seg_model_path,
    #         dst_img_path=self.out_path,
    #         seg_method=tissue_seg_method
    #     )
    #     ss_dna_tissue_cut.tissue_seg()
    #     self.tissue_mask = ss_dna_tissue_cut.mask

    @staticmethod
    def filter_roi(props):
        filtered_props = []
        for id, p in enumerate(props):
            black = np.sum(p['intensity_image'] == 0)
            sum = p['bbox_area']
            ratio_black = black / sum
            pixel_light_sum = np.sum(np.unique(p['intensity_image']) > 128)
            if ratio_black < 0.75 and pixel_light_sum > 10:
                filtered_props.append(p)
        return filtered_props

    def get_roi(self):
        pass

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
                    tissue_bbox_temp[1]: tissue_bbox_temp[3]]  # noqa
                ).astype(np.uint8)
                label_filter_list.append(label_filter)
            tissue_cell_label_filter.append(label_filter_list)
        return tissue_cell_label_filter

    def tissue_cell_infer(self):
        pass

    def mosaic(self, tissue_cell_label_filter):
        """mosaic tissue into original mask"""
        for idx, label_list in enumerate(tissue_cell_label_filter):
            tissue_bbox = self.tissue_bbox[idx]
            cell_mask = np.zeros((self.img_list[idx].shape), dtype=np.uint8)
            for i in range(self.tissue_num[idx]):
                tissue_bbox_temp = tissue_bbox[i]
                cell_mask[tissue_bbox_temp[0]: tissue_bbox_temp[2], tissue_bbox_temp[1]: tissue_bbox_temp[3]] = \
                    label_list[i]
            self.cell_mask.append(cell_mask)
        return self.cell_mask

    def watershed_score(self, cell_mask):
        pass

    def save_tissue_mask(self):
        for idx, tissue_thumb in enumerate(self.tissue_mask_thumb):
            tifffile.imsave(join(self.out_path, self.file_name[idx] + r'_tissue_cut.tif'), tissue_thumb)

    def mkdir_subpkg(self):
        """
        make new dir while image is large
        """
        assert self.is_list == False  # noqa
        file = self.file[0]
        file_name = splitext(file)[0]

        mask_outline_name = r'_watershed_outline' if self.is_water else r'_outline'
        mask_name = r'_watershed' if self.is_water else r''

        self.subpkg_mask = join(self.out_path, file_name + mask_name)
        self.subpkg_mask_outline = join(self.out_path, file_name + mask_outline_name)
        self.subpkg_score = join(self.out_path, file_name + r'_score')

        if not os.path.exists(self.subpkg_mask):
            os.mkdir(self.subpkg_mask)

        if not os.path.exists(self.subpkg_mask_outline):
            os.mkdir(self.subpkg_mask_outline)

        if not os.path.exists(self.subpkg_score):
            os.mkdir(self.subpkg_score)

    @abstractmethod
    def run(self):
        pass
