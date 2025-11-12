#######################
# intensity seg
# network infer
#######################

import copy
import os
from typing import Optional, Union, List

import cv2
import numpy as np
import tifffile
from skimage import measure
import glob

from functools import partial
from multiprocessing import cpu_count
from cellbin.modules.tissue_segmentation import TissueSegmentation

from stereo.log_manager import logger


class SingleStrandDNATissueCut(object):
    def __init__(
            self,
            src_img_path: Union[str, List[str]] = None,
            staining_type: str = None,
            model_path: str = None,
            dst_img_path: str = None,
            gpu: Union[int, str] = -1,
            num_threads: int = -1
        ):
        """
        Tissue Segmentation based on regist.tif

        :param src_img_path: the path of regist.tif, defaults to None
        :param staining_type: the staining type of regist.tif, available values include 'ssDNA', 'dapi',  'HE' and 'mIF', defaults to None
        :param model_path: the path of model, it has to be matched with staining type, defaults to None
        :param dst_img_path: the path of directory to save result mask.tif, defaults to save into the directory of regist.tif.
        :param gpu: the gpu on which the model will work, -1 means working on cpu.
        :param num_threads: the number of threads when model work on cpu, -1 means using all the cores.
        """

        self._check_staining_type(staining_type)

        if num_threads <= 0:
            num_threads = cpu_count()

        self.seg_instance = TissueSegmentation(
            model_path=model_path,
            stype=staining_type,
            gpu=gpu,
            num_threads=num_threads
        )
        self.images_data, self.files_dir, self.files_prefix = self.load_images(src_img_path)
        if dst_img_path is not None:
            os.makedirs(dst_img_path, exist_ok=True)
        self.dst_img_path = dst_img_path
        self.mask = []
        self.mask_paths = []

    def _check_image_type(self, image_path_or_name: str):
        return image_path_or_name.endswith('tif') or \
                image_path_or_name.endswith('tiff') or \
                image_path_or_name.endswith('TIF') or \
                image_path_or_name.endswith('TIFF')
    
    def _check_staining_type(self, staining_type: str):
        if staining_type is None:
            raise ValueError("staining_type didn't be gave.")
        
        if staining_type.lower() not in ('ssdna', 'dapi',  'he', 'mif'):
            raise ValueError("staining_type only can be 'ssDNA', 'dapi',  'HE' or 'mIF'.")

    def get_image_paths(self, image_path):
        if isinstance(image_path, str):
            if not os.path.isfile(image_path) and not os.path.isdir(image_path):
                raise ValueError(f'{image_path} is not a path of file or directory.')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'{image_path} is not Found.')
            if os.path.isfile(image_path):
                if not self._check_image_type(image_path):
                    raise TypeError(f'Error type image ({image_path}).')
                image_path_list = [image_path]
            else:
                image_path_list = []
                for path in glob.glob(os.path.join(image_path, f'*')):
                    if os.path.isfile(path):
                        if not self._check_image_type(path):
                            raise TypeError(f'Error type image ({path}).')
                        image_path_list.append(path)
                    elif os.path.isdir(path):
                        image_path_list.extend(self.get_image_paths(path))
        elif isinstance(image_path, list):
            image_path_list = []
            for path in image_path:
                image_path_list.extend(self.get_image_paths(path))
        else:
            raise ValueError('src_img_path must be type of string or list.')
        if len(image_path_list) == 0:
            raise Exception('There are no images can be processed.')
        return image_path_list
    
    def load_images(self, image_path):
        image_paths = self.get_image_paths(image_path)
        images_data = []
        files_dir = []
        files_prefix = []
        for path in image_paths:
            image_data = tifffile.imread(path)
            file_dir = os.path.dirname(path)
            file_name = os.path.basename(path)
            file_prefix, _ = os.path.splitext(file_name)
            images_data.append(image_data)
            files_dir.append(file_dir)
            files_prefix.append(file_prefix)
        return images_data, files_dir, files_prefix

        

    def tissue_seg(self):
        for image_data, file_dir, file_prefix in zip(self.images_data, self.files_dir, self.files_prefix):
            mask = self.seg_instance.run(image_data)
            # mask = np.multiply(mask, 255).astype(np.uint8)
            self.mask.append(mask)
            save_file_name = f"{file_prefix}_tissue_cut.tif"
            if self.dst_img_path is not None:
                save_file_path = os.path.join(self.dst_img_path, save_file_name)
            else:
                save_file_path = os.path.join(file_dir, save_file_name)
            tifffile.imwrite(save_file_path, mask)
            self.mask_paths.append(save_file_path)
        return self.mask_paths

class RNATissueCut(SingleStrandDNATissueCut):
    
    def __init__(
            self,
            dst_img_path: Optional[str] = None,
            gef_path: Optional[str] = None,
            gem_path: Optional[str] = None,
            bin_size: int = 1,
            model_path: str = None,
            gpu: Union[int, str] = -1,
            num_threads: int = -1
        ):
        """
        Tissue Segmentation based on raw.gef/raw.gem.

        :param dst_img_path: the path of directory to save result mask.tif, defaults to save into the directory of regist.tif.
        :param gef_path: choose one of `gef_path` and `gem_path`.
        :param gem_path: just like `gef_path`.
        :param bin_size: set 1 mean `bin1` for high quality, or use `bin100` for efficiency.
        :param model_path: the path of model.
        :param gpu: the gpu on which the model will work, -1 means working on cpu.
        :param num_threads: the number of threads when model work on cpu, -1 means using all the cores.
        """
        # Don't need source image type, this class will read data from gef/gem(txt)
        # super().__init__(src_img_path="", src_img_type=RNA, seg_method=INTENSITY, dst_img_path=dst_img_path)
        if gef_path is not None and gem_path is not None:
            raise Exception("only one of the gef_path and gem_path can be input.")
        if gef_path is None and gem_path is None:
            raise Exception("one of the gef_path and gem_path must be input.")
        
        # self.bin_size = bin_size

        if gef_path is not None:
            self.load_images = partial(self.get_img_from_x2tif_gef, bin_size=bin_size)
        else:
            self.load_images = self.get_img_from_x2tif_gem

        super(RNATissueCut, self).__init__(
            src_img_path=gef_path  if gef_path is not None else gem_path,
            dst_img_path=dst_img_path,
            model_path=model_path,
            staining_type='RNA',
            gpu=gpu,
            num_threads=num_threads
        )
        
        # if gef_path:
        #     self.get_img_from_x2tif_gef(gef_path, bin_size)
        # elif gem_path:
        #     self.get_img_from_x2tif_gem(gem_path)

    # def _preprocess_file(self, path):
    #     pass

    # def tissue_seg(self):
    #     self.tissue_seg_intensity()
    #     self.save_tissue_mask()

    # def get_thumb_img(self):
    #     logger.info('image loading and preprocessing...')

    #     self.img_from_x2tif = np.squeeze(self.img_from_x2tif)
    #     if len(self.img_from_x2tif.shape) == 3:
    #         self.img_from_x2tif = self.img_from_x2tif[:, :, 0]

    #     self.img.append(self.img_from_x2tif)
    #     self.shape.append(self.img_from_x2tif.shape)
        
    def _check_staining_type(self, staining_type: str):
        pass

    def get_img_from_x2tif_gef(self, gef_path, bin_size=1):
        from stereo.image.x2tif.x2tif import gef2image
        # self.img_from_x2tif = gef2image(gef_path, bin_size=bin_size)
        # self.file = [os.path.split(gef_path)[-1]]
        # self.file_name = [os.path.splitext(self.file[0])[0]]
        images_data = [gef2image(gef_path, bin_size=bin_size)]
        files_dir = [os.path.dirname(gef_path)]
        file_name = os.path.basename(gef_path)
        files_prefix = [os.path.splitext(file_name)[0]]
        return images_data, files_dir, files_prefix

    def get_img_from_x2tif_gem(self, gem_path):
        from stereo.image.x2tif.x2tif import txt2image
        # self.img_from_x2tif = txt2image(gem_path)
        # self.file = [os.path.split(gem_path)[-1]]
        # self.file_name = [os.path.splitext(self.file[0])[0]]
        images_data = [txt2image(gem_path)]
        files_dir = [os.path.dirname(gem_path)]
        file_name = os.path.basename(gem_path)
        files_prefix = [os.path.splitext(file_name)[0]]
        return images_data, files_dir, files_prefix
    
    # def load_images(self, gef_gem_path):
    #     if 'gef' in gef_gem_path:
    #         return self.get_img_from_x2tif_gef(gef_gem_path, self.bin_size)
    #     else:
    #         return self.get_img_from_x2tif_gem(gef_gem_path)
