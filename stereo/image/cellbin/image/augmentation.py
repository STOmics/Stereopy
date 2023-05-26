import copy

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from PIL import Image
import cv2


def f_rgb2gray(img, need_not=False):
    """
    rgb2gray

    :param img: (CHANGE) np.array
    :param need_not: if need bitwise_not
    :return: np.array
    """
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
            img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if need_not:
            img = cv2.bitwise_not(img)
    return img


def f_gray2bgr(img):
    """
    gray2bgr

    :param img: (CHANGE) np.array
    :return: np.array
    """

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def f_padding(img, top, bot, left, right, mode='constant', value=0):
    """
    update by dengzhonghan on 2023/2/23
    1. support 3d array padding.
    2. not support 1d array padding.

    Args:
        img (): numpy ndarray (2D or 3D).
        top (): number of values padded to the top direction.
        bot (): number of values padded to the bottom direction.
        left (): number of values padded to the left direction.
        right (): number of values padded to the right direction.
        mode (): padding mode in numpy, default is constant.
        value (): constant value when using constant mode, default is 0.

    Returns:
        pad_img: padded image.

    """

    if mode == 'constant':
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode, constant_values=value)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode, constant_values=value)
    else:
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode)
    return pad_img


def f_resize(img, shape=(1024, 2048), mode="NEAREST"):
    """
    resize img with pillow

    :param img: (CHANGE) np.array
    :param shape: tuple
    :param mode: An optional resampling filter. This can be one of Resampling.NEAREST,
     Resampling.BOX, Resampling.BILINEAR, Resampling.HAMMING, Resampling.BICUBIC or Resampling.LANCZOS.
     If the image has mode “1” or “P”, it is always set to Resampling.NEAREST.
     If the image mode specifies a number of bits, such as “I;16”, then the default filter is Resampling.NEAREST.
     Otherwise, the default filter is Resampling.BICUBIC
    :return:np.array
    """
    imode = Image.NEAREST
    if mode == "BILINEAR":
        imode = Image.BILINEAR
    elif mode == "BICUBIC":
        imode = Image.BICUBIC
    elif mode == "LANCZOS":
        imode = Image.LANCZOS
    elif mode == "HAMMING":
        imode = Image.HAMMING
    elif mode == "BOX":
        imode = Image.BOX
    if img.dtype != 'uint8':
        imode = Image.NEAREST
    img = Image.fromarray(img)
    img = img.resize((shape[1], shape[0]), resample=imode)
    img = np.array(img).astype(np.uint8)
    return img


def f_percentile_threshold(img, percentile=99.9):
    """
    Threshold an image to reduce bright spots

    :param img: (CHANGE) numpy array of image data
    :param percentile: cutoff used to threshold image
    :return: np.array: thresholded version of input image
    """

    # non_zero_vals = img[np.nonzero(img)]
    non_zero_vals = img[img > 0]

    # only threshold if channel isn't blank
    if len(non_zero_vals) > 0:
        img_max = np.percentile(non_zero_vals, percentile)

        # threshold values down to max
        threshold_mask = img > img_max
        img[threshold_mask] = img_max

    return img


def f_equalize_adapthist(img, kernel_size=None):
    """
    Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    :param img: (CHANGE) (numpy.array): numpy array of phase image data.
    :param kernel_size: (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    :return: numpy.array:Pre-processed image
    """
    return equalize_adapthist(img, kernel_size=kernel_size)


def f_histogram_normalization(img):
    """
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    :param img: (CHANGE) (numpy.array): numpy array of phase image data.
    :return: numpy.array:image data with dtype float32.
    """

    img = img.astype('float32')
    sample_value = img[(0,) * img.ndim]
    if (img == sample_value).all():
        return np.zeros_like(img)
    img = rescale_intensity(img, out_range=(0.0, 1.0))

    return img


def f_ij_16_to_8(img, chunk_size=1000):
    """
    16 bits img to 8 bits

    :param img: (CHANGE) np.array
    :param chunk_size: chunk size (bit)
    :return: np.array
    """

    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = copy.deepcopy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst


def enhance(arr, mode, thresh):
    """
    Only support 2D array

    Args:
        arr (): 2D numpy array
        mode (): enhance mode
        thresh (): threshold

    Returns:

    """
    data = arr.ravel()
    min_v = np.min(data)
    data_ = data[np.where(data <= thresh)]
    if len(data_) == 0:
        return 0, 0
    if mode == 'median':
        var_ = np.std(data_)
        thr = np.median(data_)
        max_v = thr + var_
    elif mode == 'hist':
        freq_count, bins = np.histogram(data_, range(min_v, int(thresh + 1)))
        count = np.sum(freq_count)
        freq = freq_count / count
        thr = bins[np.argmax(freq)]
        max_v = thr + (thr - min_v)
    else:
        raise Exception('Only support median and histogram')

    return min_v, max_v


def encode(arr, min_v, max_v):
    """
    Encode image with min and max pixel value

    Args:
        arr (): 2D numpy array
        min_v (): min value obtained from enhance method
        max_v (): max value

    Returns:
        mat: encoded mat

    """
    if min_v >= max_v:
        arr = arr.astype(np.uint8)
        return arr
    mat = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    v_w = max_v - min_v
    mat[arr < min_v] = 0
    mat[arr > max_v] = 255
    pos = (arr >= min_v) & (arr <= max_v)
    mat[pos] = (arr[pos] - min_v) * (255 / v_w)
    return mat


def f_ij_auto_contrast(img):
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
    if hmax > hmin:
        hmax = int(hmax * bit_max / 256)
        hmin = int(hmin * bit_max / 256)
        img[img < hmin] = hmin
        img[img > hmax] = hmax
        cv2.normalize(img, img, 0, bit_max - 1, cv2.NORM_MINMAX)
    return img

