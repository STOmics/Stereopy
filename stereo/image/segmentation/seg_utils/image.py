import numpy as np
import tifffile
import cv2
import glog
from os.path import join, splitext, split
import seg_utils.utils as utils


SPLIT_SIZE = 20000


class Image(object):

    def __init__(self, path):

        self.__file = split(path)[-1]
        self.__file_name = splitext(path)[0]
        self.__suffix = splitext(path)[-1]
        self.__img = self.__imload(path)
        self.__convert_gray()
        self.__trans16to8()
        self.__dtype = self.__img.dtype
        self.__shape = self.__img.shape
        self.__is_split = np.sum(np.array(self.__shape) > SPLIT_SIZE)
        self.tisue_mask = []
        self.tisue_mask_thumb = []
        self.tissue_num = []  # tissue num in each image
        self.tissue_bbox = []  # tissue roi bbox in each image
        self.img_filter = []  # image filtered by tissue mask
        self.cell_mask = []
        self.cell_mask_water = []
        self.score_mask = []


    def __imload(self, path):

        assert self.__suffix in ['.tif', '.png', '.jpg']
        if self.__suffix == '.tif':
            img = tifffile.imread(path)
        else:
            img = cv2.imread(path, -1)
        return img


    def __convert_gray(self):

        if len(self.__img.shape) == 3:
            glog.info('Image %s convert to gray!'%self.__file)
            self.__img = self.__img[:, :, 0]


    def __trans16to8(self):

        assert self.__dtype in ['uint16', 'uint8']
        if self.__dtype != 'uint8':
            glog.info('%s transfer to 8bit'%self.__file)
            self.__img = utils.transfer_16bit_to_8bit(self.__img)