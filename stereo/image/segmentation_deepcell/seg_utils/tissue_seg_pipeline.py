import cv2
from . import utils as utils
from skimage import measure
import numpy as np
from PIL import Image
import multiprocessing as mp



class TissueSegPipe(object):

    def __init__(self, image, scale = 5):
        self.image = image
        self.scale = scale
        self.img_thumb = []
        self.mask = []
        self.mask_thumb = []
        pass


    def down_sample(self):
        ori_image = Image.fromarray(self.image.__img)
        shape = self.image.__shape
        image_thumb = ori_image.resize((shape[1] // self.scale, shape[0] // self.scale), Image.NEAREST)
        self.img_thumb = np.array(image_thumb).astype(np.uint8)


    def up_sample(self, image):
        mask_thumb = Image.fromarray(image)
        ori_shape = self.image.__shape
        marker = mask_thumb.resize((ori_shape[1], ori_shape[0]), Image.NEAREST)
        marker = np.array(marker).astype(np.uint8)
        self.mask = marker
        return marker


    def tissueSeg(self):

        # downsample ori_image
        self.down_sample()

        # binary
        ret1, mask_thumb = cv2.threshold(self.img_thumb, 125, 255, cv2.THRESH_OTSU)
        if mask_thumb.dtype != 'uint8':
            mask_thumb = utils.transfer_16bit_to_8bit(mask_thumb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  # 椭圆结构
        mask_thumb = cv2.morphologyEx(mask_thumb, cv2.MORPH_CLOSE, kernel, iterations=8)

        # choose tissue prop
        label_image = measure.label(mask_thumb, connectivity=2)
        props = measure.regionprops(label_image, intensity_image=mask_thumb)

        def getArea(elem):
            return elem.area

        props.sort(key=getArea, reverse=True)
        areas = [p['area'] for p in props]
        label_num = int(np.sum(areas > np.mean(areas)))
        result = np.zeros((self.img_thumb.shape)).astype(np.uint8)
        for i in range(label_num):
            prop = props[i]
            result += np.where(label_image != prop.label, 0, 1).astype(np.uint8)

        result = utils.hole_fill(result)
        result_thumb = cv2.dilate(result, kernel, iterations=10)

        # upsample
        marker = self.up_sample(result_thumb)
        marker = np.uint8(marker > 0)

        return marker, np.uint8(result_thumb > 0)







def tissue_seg_multi(self, input_list, processes):
    with mp.Pool(processes=processes) as p:
        pre_tissue = p.map(self.tissueSeg, input_list)
    return pre_tissue



