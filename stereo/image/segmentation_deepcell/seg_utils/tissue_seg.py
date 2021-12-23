import cv2
import numpy as np
from skimage import measure
from PIL import Image
import multiprocessing as mp

from . import utils as utils  #


def down_sample(img, scale=5):

    shape = img.shape
    ori_image = Image.fromarray(img)
    image_thumb = ori_image.resize((shape[1] // scale, shape[0] // scale), Image.NEAREST)
    image_thumb = np.array(image_thumb).astype(np.uint8)
    return image_thumb


def up_sample(image, ori_shape):

    mask_thumb = Image.fromarray(image)
    marker = mask_thumb.resize((ori_shape[1], ori_shape[0]), Image.NEAREST)
    marker = np.array(marker).astype(np.uint8)
    return marker


def hole_fill(binary_image):
    ''' 孔洞填充 '''
    hole = binary_image.copy()  ## 空洞填充
    hole = cv2.copyMakeBorder(hole, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0])  # 首先将图像边缘进行扩充，防止空洞填充不完全
    hole2 = hole.copy()
    cv2.floodFill(hole, None, (0, 0), 255)  # 找到洞孔
    hole = cv2.bitwise_not(hole)
    binary_hole = cv2.bitwise_or(hole2, hole)[1:-1, 1:-1]
    return binary_hole


def getArea(elem):
    return elem.area


def tissueSeg(ori_image_list):

    if not isinstance(ori_image_list, list):
        ori_image_list = [ori_image_list]

    result_list = []
    for idx, ori_image in enumerate(ori_image_list):
        shapes = ori_image.shape
        # downsample ori_image
        image_thumb = down_sample(ori_image)

        # binary
        ret1, mask_thumb = cv2.threshold(image_thumb, 125, 255, cv2.THRESH_OTSU)
        if mask_thumb.dtype != 'uint8':
            mask_thumb = utils.transfer_16bit_to_8bit(mask_thumb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  # 椭圆结构
        mask_thumb = cv2.morphologyEx(mask_thumb, cv2.MORPH_CLOSE, kernel, iterations=8)

        # choose tissue prop
        label_image = measure.label(mask_thumb, connectivity=2)
        props = measure.regionprops(label_image, intensity_image=mask_thumb)
        props.sort(key=getArea, reverse=True)
        areas = [p['area'] for p in props]
        if np.std(areas) * 10 < np.mean(areas):
            label_num = len(areas)
        else:
            label_num = int(np.sum(areas >= np.mean(areas)))
        result = np.zeros((image_thumb.shape)).astype(np.uint8)
        for i in range(label_num):
            prop = props[i]
            result += np.where(label_image != prop.label, 0, 1).astype(np.uint8)


        result = hole_fill(result)
        result_thumb = cv2.dilate(result, kernel, iterations=10)

        # upsample
        marker = up_sample(result_thumb, shapes)
        marker = np.uint8(marker > 0)

        result_list.append([marker, np.uint8(result_thumb > 0)])

    if len(result_list) == 1:
        result_list = result_list[0]

    return result_list


def tissue_seg_multi(input_list, processes):
    with mp.Pool(processes=processes) as p:
        pre_tissue = p.map(tissueSeg, input_list)
    return pre_tissue


# if __name__=='__main__':
#     import tifffile
#     img = tifffile.imread(r'D:\limin\img\issue_img\21SD-GJ-006-C-ssDNA-B5-16bit_registered.tif')
#     tissue = tissueSeg(img)
#     tifffile.imsave(r'D:\limin\img\issue_img\21SD-GJ-006-C-ssDNA-B5-16bit_registered_tissue_mask.tif', tissue[0])