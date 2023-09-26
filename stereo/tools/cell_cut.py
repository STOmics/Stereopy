import os

from gefpy import bgef_writer_cy
from gefpy import cgef_writer_cy

from ..log_manager import logger


class CellCut(object):
    def __init__(self, cgef_out_dir):
        """generate cgef from bgef and mask or bgef and ssdna image

        :param cgef_out_dir: the path of the directory to save generated cgef
        """
        self.cgef_out_dir = cgef_out_dir
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
        logger.info("generate bgef finished")
        return bgef_path

    def cell_cut(self,
                 bgef_path: str = None,
                 gem_path: str = None,
                 mask_path: str = None,
                 image_path: str = None,
                 model_path: str = None,
                 mask_save: bool = True,
                 model_type: str = 'deep-learning',
                 depp_cro_size: int = 20000,
                 overlap: int = 100,
                 gen_mask_on_gpu: str = '-1',
                 tissue_seg_model_path: str = None,
                 tissue_seg_method: str = None,
                 post_processing_workers: int = 10
                 ):
        """
        Generate CGEF resutl via following combinations:
            * BGEF and mask
            * BGEF and ssDNA image

        :param bgef_path: the path to BGEF file.
        :param gem_path: the path to GEM file.
        :param mask_path: the path to mask file.
        :param image_path: the path to ssDNA image file.
        :param model_path: the path to model file.
        :param mask_save: whether to save mask file after correction, generated from ssDNA image.
        :param model_type: the type of model to generate mask, whcih only could be set to deep learning model and deep cell model.
        :param depp_cro_size: deep crop size.
        :param overlap: overlap size.
        :param gen_mask_on_gpu: specify gpu id to predict when generate mask, if `'-1'`, use cpu for prediction.
        :param tissue_seg_model_path: the path of deep-learning model of tissue segmentation, if set it to None, it would use OpenCV to process.
        :param tissue_seg_method: the method of tissue segmentation, 0 is deep-learning and 1 is OpenCV.
        :param post_processing_workers: the number of processes for post-processing.

        :return: Path to CGEF result.
        """  # noqa
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
