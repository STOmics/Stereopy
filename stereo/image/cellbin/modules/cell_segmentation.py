import os

import numpy as np
from tifffile import tifffile
import matplotlib.pyplot as plt

from . import CellBinElement
from ..dnn.cseg.cell_trace import get_trace as get_t
from ..dnn.cseg.detector import Segmentation
from ...tissue_cut import SingleStrandDNATissueCut, DEEP, INTENSITY


class CellSegmentation(CellBinElement):
    def __init__(self, model_path, gpu="-1", num_threads=0):
        super(CellSegmentation, self).__init__()

        self._MODE = "onnx"
        self._NET = "bcdu"
        self._WIN_SIZE = (256, 256)
        self._INPUT_SIZE = (256, 256, 1)
        self._OVERLAP = 16

        self._gpu = gpu
        self._model_path = model_path
        self._num_threads = num_threads

        self._cell_seg = Segmentation(
            net=self._NET,
            mode=self._MODE,
            gpu=self._gpu,
            num_threads=self._num_threads,
            win_size=self._WIN_SIZE,
            intput_size=self._INPUT_SIZE,
            overlap=self._OVERLAP
        )
        self._cell_seg.f_init_model(model_path=self._model_path)

    def run(self, img):
        mask = self._cell_seg.f_predict(img)
        return mask

    @staticmethod
    def get_trace(mask):
        return get_t(mask)


def _get_tissue_mask(img_path, model_path, method, dst_img_path):
    if method is None:
        method = DEEP
    if not model_path or len(model_path) == 0:
        method = INTENSITY
    ssDNA_tissue_cut = SingleStrandDNATissueCut(
        src_img_path=img_path,
        model_path=model_path,
        dst_img_path=dst_img_path,
        seg_method=method
    )
    ssDNA_tissue_cut.tissue_seg()
    return ssDNA_tissue_cut.mask[0]


def _get_img_filter(img, tissue_mask):
    """get tissue image by tissue mask"""
    img_filter = np.multiply(img, tissue_mask)
    return img_filter


def cell_seg_v3(
        model_path: str,
        img_path: str,
        out_path: str,
        gpu="-1",
        num_threads=0,
        need_tissue_cut=True,
        tissue_seg_model_path: str = None,
        tissue_seg_method: str = None,
        tissue_seg_dst_img_path=None,

):
    """
    Implement cell segmentation v3 by deep learning model.

    Parameters
    -----------------
    model_path
        the path to deep learning model.
    img_path
        the path to image file.
    out_path
        the path to output mask result.
    gpu
        set gpu id, if `'-1'`, use cpu for prediction.
    num_threads
        multi threads num of the model reading process
    need_tissue_cut
        whether cut image as tissue before cell segmentation
    tissue_seg_model_path
        the path of deep learning model of tissue segmentation, if set it to None, it would use OpenCV to process.
    tissue_seg_method
        the method of tissue segmentation, 1 is based on deep learning and 0 is based on OpenCV.
    tissue_seg_dst_img_path
        default to the img_path's directory.
    Returns
    ------------
    None

    """
    cell_bcdu = CellSegmentation(
        model_path=model_path,
        gpu=gpu,
        num_threads=num_threads
    )
    if img_path.split('.')[-1] == "tif":
        img = tifffile.imread(img_path)
    elif img_path.split('.')[-1] == "png":
        img = plt.imread(img_path)
        if img.dtype == np.float32:
            img.astype('uint32')
            img = transfer_32bit_to_8bit(img)
    else:
        raise Exception("cell seg only support tif and png")

    # img must be 16 bit ot 8 bit, and 16 bit image finally will be transferred to 8 bit
    assert img.dtype == np.uint16 or img.dtype == np.uint8, f'{img.dtype} is not supported'
    if img.dtype == np.uint16:
        img = transfer_16bit_to_8bit(img)
    if need_tissue_cut:
        if tissue_seg_dst_img_path is None:
            tissue_seg_dst_img_path = os.path.dirname(img_path)
        tissue_mask = _get_tissue_mask(img_path, tissue_seg_model_path, tissue_seg_method, tissue_seg_dst_img_path)
        img = _get_img_filter(img, tissue_mask)
    mask = cell_bcdu.run(img)
    CellSegmentation.get_trace(mask)
    tifffile.imwrite(out_path, mask)


def transfer_16bit_to_8bit(image_16bit):
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit


def transfer_32bit_to_8bit(image_32bit):
    min_32bit = np.min(image_32bit)
    max_32bit = np.max(image_32bit)
    image_8bit = np.array(
        np.rint(255 * ((image_32bit - min_32bit) / (max_32bit - min_32bit))), dtype=np.uint8
    )
    return image_8bit
