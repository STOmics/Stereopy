import glog
# import image
import os
import time
from skimage import measure
from os.path import join, splitext, exists, split
import tifffile
import cv2
import numpy as np

import seg_utils.grade as grade
import seg_utils.utils as utils
from multiprocessing import Process
from multiprocessing import Queue
import matplotlib.pyplot as plt


class CellSegPipe(object):

    def __init__(self, model_path, img_path, out_path, is_water, DEEP_CROP_SIZE=20000, OVERLAP=100):
        self.deep_crop_size = DEEP_CROP_SIZE
        self.overlap = OVERLAP
        self.model_path = model_path
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
            os.mkdir(out_path)
            glog.info('Create new dir : %s' % out_path)
        self.__is_water = is_water
        t0 = time.time()
        self.__trans16to8()
        t1 = time.time()
        glog.info('Transform 16bit to 8bit : %.2f' % (t1 - t0))
        self.tissue_mask = []
        self.tissue_mask_thumb = []
        self.tissue_num = []  # tissue num in each image
        self.tissue_bbox = []  # tissue roi bbox in each image
        self.img_filter = []  # image filtered by tissue mask
        self.cell_mask = []
        self.post_mask_list = []
        self.score_mask_list = []

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
                glog.info('Image %s convert to gray!' % self.__file[idx])
                self.img_list[idx] = img[:, :, 0]

    def __trans16to8(self):

        for idx, img in enumerate(self.img_list):
            assert img.dtype in ['uint16', 'uint8']
            if img.dtype != 'uint8':
                glog.info('%s transfer to 8bit' % self.__file[idx])
                self.img_list[idx] = utils.transfer_16bit_to_8bit(img)


    def tissue_cell_infer(self, q):
        import seg_utils.cell_infer as cell_infer

        """cell segmentation in tissue area by neural network"""
        tissue_cell_label = []
        for idx, img in enumerate(self.img_list):
            label_list = cell_infer.cellInfer(self.model_path, img, self.deep_crop_size, self.overlap)
            tissue_cell_label.append(label_list)
        q.put(tissue_cell_label)


    def watershed_score(self, cell_mask):

        """watershed and score on cell mask by neural network"""
        for idx, cell_mask in enumerate(cell_mask):
            cell_mask = np.squeeze(cell_mask)
            cell_mask_tile, x_list, y_list = utils.split(cell_mask, self.deep_crop_size)
            img_tile, _, _ = utils.split(self.img_list[idx], self.deep_crop_size)
            input_list = [[cell_mask_tile[id], img] for id, img in enumerate(img_tile)]
            if self.__is_water:
                post_list_tile = grade.watershed_multi(input_list, 15)
            else:
                post_list_tile = grade.score_multi(input_list, 15)

            post_mask_tile = [label[0] for label in post_list_tile]
            score_mask_tile = [label[1] for label in post_list_tile]  # grade saved
            post_mask = utils.merge(post_mask_tile, x_list, y_list, cell_mask.shape)
            score_mask = utils.merge(score_mask_tile, x_list, y_list, cell_mask.shape)
            self.post_mask_list.append(post_mask)
            self.score_mask_list.append(score_mask)


    def __mkdir_subpkg(self):

        """make new dir while image is large"""
        assert self.__is_list == False
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

        """save mask"""
        self.save_cell_mask()

    def run(self):

        t1 = time.time()

        q = Queue()
        t = Process(target=self.tissue_cell_infer, args=(q,))
        t.start()

        tissue_cell_label = q.get()
        t.join()
        # tissue_cell_label = self.tissue_cell_infer()
        t2 = time.time()
        glog.info('Cell inference : %.2f' % (t2 - t1))


        ###post process###
        # self.watershed_score(cell_mask)
        self.watershed_score(tissue_cell_label)
        t5 = time.time()
        glog.info('Post-processing : %.2f' % (t5 - t2))

        self.save_result()
        glog.info('Result saved : %s ' % (self.__out_path))
