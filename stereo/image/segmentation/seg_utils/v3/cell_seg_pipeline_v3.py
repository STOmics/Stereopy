# import image
import os

import matplotlib.pyplot as plt
import numpy as np
from cellbin.image import Image
from cellbin.modules.cell_segmentation import CellSegmentation
from tifffile import tifffile

from stereo.image.segmentation.seg_utils.base_cell_seg_pipe.cell_seg_pipeline import CellSegPipe
from stereo.log_manager import logger
from stereo.tools.tools import make_dirs


class CellSegPipeV3(CellSegPipe):

    def _get_img_filter(self, img, tissue_mask):
        """get tissue image by tissue mask"""
        img_filter = np.multiply(img, tissue_mask)
        return img_filter

    def run(self):
        logger.info('Start do cell mask, the method is v3, this will take some minutes.')
        num_threads = self.kwargs.get('num_threads', 0)
        if num_threads <= 0:
            from multiprocessing import cpu_count
            num_threads = cpu_count()
        cell_seg = CellSegmentation(
            model_path=self.model_path,
            gpu=self.gpu,
            num_threads=num_threads,
        )
        logger.info(f"Load {self.model_path}) finished.")
        if self.img_path.split('.')[-1] == "tif":
            img = tifffile.imread(self.img_path)
        elif self.img_path.split('.')[-1] == "png":
            img = plt.imread(self.img_path)
            if img.dtype == np.float32:
                img.astype('uint32')
                img = self.transfer_32bit_to_8bit(img)
        else:
            raise Exception("cell seg only support tif and png")

        # img must be 16 bit ot 8 bit, and 16 bit image finally will be transferred to 8 bit
        assert img.dtype == np.uint16 or img.dtype == np.uint8, f'{img.dtype} is not supported'
        if img.dtype == np.uint16:
            img = self.transfer_16bit_to_8bit(img)

        # if self.kwargs.get('need_tissue_cut', None):
            # self.get_tissue_mask()
        if self.tissue_mask is not None:
            img = self._get_img_filter(img, self.tissue_mask[0])

        # Run cell segmentation
        mask = cell_seg.run(img)
        self.mask = mask
        self.save_cell_mask()

    def save_cell_mask(self):
        make_dirs(self.out_path)
        cell_mask_path = os.path.join(self.out_path, f"{self.file_name[-1]}_mask.tif")
        Image.write_s(self.mask, cell_mask_path, compression=True)
        logger.info('Result saved : %s ' % (cell_mask_path))
