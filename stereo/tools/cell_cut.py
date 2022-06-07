import os
from gefpy import cgef_writer_cy
from .cell_segment import CellSegment
from ..log_manager import logger

class CellCut(object):
    def __init__(self, cgef_out_dir):
        """generate cgef from bgef and mask or bgef and ssdna image

        :param cgef_out_dir: the path of the directory to save generated cgef
        """
        self.cgef_out_dir = cgef_out_dir
        # self.cgef_out_parent_path = os.path.dirname(cgef_out_path)
        if not os.path.exists(self.cgef_out_dir):
            os.makedirs(self.cgef_out_dir)
    
    def cell_cut(self, bgef_path,
                mask_path=None,
                image_path=None,
                model_path=None,
                mask_save=True,
                model_type='deep-learning',
                depp_cro_size=20000,
                overlap=100,
                gen_mask_on_gpu='-1'):
        """generate cgef by bgef and mask or bgef and ssdna image

        :param bgef_path: the path of bgef
        :param mask_path: the path of mask, if None, need to specify the path of ssdn image by parameter image_path to generate it, defaults to None
        :param image_path: the path of ssdn image, if there is no mask, must to input it, defaults to None
        :param model_path: the path of model use to generate mask, defaults to None
        :param mask_save: if True, save the mask after generating cgef, if False, don't to save, defaults to True
        :param model_type: the model type of the model use to generate mask, deep-learning or deep-cell, defaults to 'deep-learning'
        :param depp_cro_size: deep crop size, parameter for generating mask, defaults to 20000
        :param overlap: the size of overlap, parameter for generating mask, defaults to 100
        :param gen_mask_on_gpu: specify the gpu id if calculated on gpu when generate mask, if -1, calculate on cpu, defaults to '-1'
        :return: the path of the generated cgef
        """
        if mask_path is None and image_path is None:
            raise Exception("must to input the mask or ssdn image")
        
        if mask_path is None:
            logger.info(f"there is no mask file, generate it by model {model_path}")
            cell_segment = CellSegment(image_path, gen_mask_on_gpu, self.cgef_out_dir)
            cell_segment.generate_mask(model_path, model_type, depp_cro_size, overlap)
            mask_path = cell_segment.get_mask_files()[0]
            logger.info(f"the generated mask file {mask_path}")

        file_name = os.path.basename(bgef_path)
        file_prefix = os.path.splitext(file_name)[0]
        file_name = f"{file_prefix}.cellbin.gef"
        cgef_out_path = os.path.join(self.cgef_out_dir, file_name)
        cgef_writer_cy.generate_cgef(cgef_out_path, bgef_path, mask_path, [256, 256])
        if not mask_save:
            cell_segment.remove_all_mask_files()
        return cgef_out_path