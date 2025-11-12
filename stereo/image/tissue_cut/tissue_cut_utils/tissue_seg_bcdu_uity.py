import copy

import cv2
import numpy as np
from PIL import Image


def up_sample(img, shape=(1024, 2048)):
    mask_thumb = Image.fromarray(img)
    marker = mask_thumb.resize((shape[1], shape[0]), Image.NEAREST)
    marker = np.array(marker).astype(np.uint8)
    return marker


def down_sample(img, shape=(1024, 2048)):
    ori_image = Image.fromarray(img)
    image_thumb = ori_image.resize((shape[1], shape[0]), Image.NEAREST)
    image_thumb = np.array(image_thumb).astype(np.uint8)
    return image_thumb


def ij_16_to_8(img):
    dst = copy.deepcopy(img)
    p_max = np.max(dst)
    p_min = np.min(dst)
    scale = 256.0 / (p_max - p_min + 1)
    dst = np.int16(dst)
    dst = (dst & 0xffff) - p_min
    dst[dst < 0] = 0
    dst = dst * scale + 0.5
    dst[dst > 255] = 255
    dst = np.uint8(dst)
    return dst


def ij_auto_contrast(img):
    limit = img.size / 10
    threshold = img.size / 5000
    if img.dtype != 'uint8':
        bit_max = 65536
    else:
        bit_max = 256
    hist, _ = np.histogram(img.flatten(), 256, [0, bit_max])
    hmin = 0
    hmax = bit_max - 1
    for i in range(1, len(hist) - 1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(len(hist) - 2, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break
    dst = copy.deepcopy(img)
    if hmax > hmin:
        hmax = int(hmax * bit_max / 256)
        hmin = int(hmin * bit_max / 256)
        dst[dst < hmin] = hmin
        dst[dst > hmax] = hmax
        cv2.normalize(dst, dst, 0, bit_max - 1, cv2.NORM_MINMAX)
    return dst
