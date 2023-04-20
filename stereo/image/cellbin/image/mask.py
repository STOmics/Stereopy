import cv2
import numpy as np


def f_fill_all_hole(mask_in):
    """
    fill all holes in the mask

    :param mask_in: np.array np.uint8
    :return: np.array np.uint8
    """
    ''' 对二值图像进行孔洞填充 '''
    im_floodfill = cv2.copyMakeBorder(mask_in, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0])
    # im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill[2:-2, 2:-2])

    # Combine the two images to get the foreground.
    return mask_in | im_floodfill_inv


def f_instance2semantics(ins):
    """
    update by cenweixuan on 2023/3/07
    :param ins:
    :return:
    """
    h, w = ins.shape[:2]
    tmp0 = ins[1:, 1:] - ins[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins[1:, :w - 1] - ins[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins[ind1] = 0
    ins[ind0] = 0
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)


def iou(a, b, epsilon=1e-5):
    """
    add by jqc on 2023/04/10
    Args:
        a ():
        b ():
        epsilon ():

    Returns:

    """
    # 首先将a和b按照0/1的方式量化
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)

    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)

    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)

    # 计算IoU
    iou = intersection / (union + epsilon)

    return iou
