import os
from gefpy import cgef_writer_cy, bgef_writer_cy
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

    def get_file_name(self, source_file_path, ext=None):
        ext = ext.lstrip('.') if ext is not None else ""
        file_name = os.path.basename(source_file_path)
        file_prefix = os.path.splitext(file_name)[0]
        if ext == "":
            return file_prefix
        else:
            return f"{file_prefix}.{ext}"
    
    def generate_bgef(self, gem_path, threads=10):
        file_name = self.get_file_name(gem_path, 'bgef')
        bgef_path = os.path.join(self.cgef_out_dir, file_name)
        logger.info(f"start to generate bgef({bgef_path})")
        if os.path.exists(bgef_path):
            os.remove(bgef_path)
        bgef_writer_cy.generate_bgef(gem_path, bgef_path, n_thread=threads, bin_sizes=[1])
        logger.info(f"generate bgef finished")
        return bgef_path
    
    def cell_cut(self,
                bgef_path=None,
                gem_path=None,
                mask_path=None,
                image_path=None,
                model_path=None,
                mask_save=True,
                model_type='deep-learning',
                depp_cro_size=20000,
                overlap=100,
                gen_mask_on_gpu='-1'):
        """generate cgef by bgef and mask or bgef and ssdna image

        :param bgef_path: the path of bgef, if None, need to specify the path of gem by parameter gem_path to generate it, defaults to None
        :param gem_path: the path of gem, if there is no bgef, must to input it to convert to bgef, defaults to None
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
        if bgef_path is None and gem_path is None:
            raise Exception("must to input the path of bgef or the path of gem")

        if mask_path is None and image_path is None:
            raise Exception("must to input the mask or ssdn image")
        
        if bgef_path is None and gem_path is not None:
            bgef_path = self.generate_bgef(gem_path)
        
        do_mask_generating = False
        if mask_path is None:
            from .cell_segment import CellSegment
            logger.info(f"there is no mask file, generate it by model {model_path}")
            cell_segment = CellSegment(image_path, gen_mask_on_gpu, self.cgef_out_dir)
            cell_segment.generate_mask(model_path, model_type, depp_cro_size, overlap)
            mask_path = cell_segment.get_mask_files()[0]
            logger.info(f"the generated mask file {mask_path}")
            do_mask_generating = True

        file_name = self.get_file_name(bgef_path, 'cellbin.gef')
        cgef_out_path = os.path.join(self.cgef_out_dir, file_name)
        cgef_writer_cy.generate_cgef(cgef_out_path, bgef_path, mask_path, [256, 256])
        if not mask_save and do_mask_generating:
            cell_segment.remove_all_mask_files()
        return cgef_out_path