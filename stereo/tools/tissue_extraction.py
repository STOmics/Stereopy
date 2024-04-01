"""
Tissue Extraction from bGEF
"""
from typing import Optional, Union

import numpy as np
from gefpy.bgef_creater_cy import BgefCreater
from scipy.sparse import csr_matrix

from ..core import StereoExpData
from ..core.cell import Cell
from ..core.gene import Gene
from ..image.tissue_cut import (
    # ssDNA,
    # RNA,
    # DEEP,
    SingleStrandDNATissueCut,
    RNATissueCut
)
from ..log_manager import logger


def tissue_extraction_to_bgef(
        src_gef_path: Optional[str],
        dst_bgef_path: Optional[str],
        dst_mask_dir_path: Optional[str],
        staining_type: Optional[str] = None,
        model_path: Optional[str] = "",
        src_img_path: Optional[str] = "",
        rna_tissue_cut_bin_size: Optional[int] = 1,
        gpu: Union[int, str] = '-1',
        num_threads: int = -1
):
    """
    :param src_gef_path: the gef will be extracted by mask.tif from tissue cut.
    :param dst_bgef_path: result bgef path, contains file name.
    :param dst_mask_dir_path: save tissue cut result mask.tif to this directory.
    :param staining_type: the staining type of regist.tif, available values include 'ssDNA', 'dapi',  'HE', 'mIF' and 'RNA',
                     except 'RNA' staining type, `src_img_path` is required,
                     while setting to 'RNA', the mask.tif will be generated from src gef.
    :param model_path: the path if model used to generated mask.tif, it has to be matched with staining type.
    :param src_img_path: the path of regist.tif which is used to generated mask.tif.
    :param rna_tissue_cut_bin_size: fixed value 1.
    :param gpu: the gpu on which the model will work, -1 means working on cpu.
    :param num_threads: the number of threads when model work on cpu, -1 means using all the cores.

    :return: the dst_bgef_path
    """
    if staining_type is None:
        raise ValueError("staining_type didn't be gave.")
    
    if staining_type.lower() not in ('ssdna', 'dapi',  'he', 'mif', 'rna'):
        raise ValueError("staining_type only can be 'ssDNA', 'dapi',  'HE', 'mIF' or 'RNA'.")
    
    staining_type = staining_type.lower()
    if staining_type == 'rna':
        tissue_cut_obj = RNATissueCut(
            dst_img_path=dst_mask_dir_path,
            gef_path=src_gef_path,
            bin_size=rna_tissue_cut_bin_size,
            model_path=model_path,
            gpu=gpu,
            num_threads=num_threads
        )
        extract_bin_size = rna_tissue_cut_bin_size
    else:
        tissue_cut_obj = SingleStrandDNATissueCut(
            # seg_method=seg_method,
            src_img_path=src_img_path,
            dst_img_path=dst_mask_dir_path,
            staining_type=staining_type,
            model_path=model_path,
            gpu=gpu,
            num_threads=num_threads
        )
        extract_bin_size = 1
    # else:
    #     logger.error(f'{src_type} is not a tissue-cut-src_type')
    #     raise Exception

    # real do the image transforming
    tissue_cut_obj.tissue_seg()

    # TODO: mask file is dump to disk, which should be a option for user to choose
    mask_file_path = tissue_cut_obj.mask_paths[-1]
    logger.info(f'tissue_cut finish, mask file is saved at {mask_file_path}')

    # gef extraction
    bc = BgefCreater()
    bc.create_bgef(src_gef_path, extract_bin_size, mask_file_path, dst_bgef_path)
    logger.info(f'gef extraction finish, extracted gef save at {dst_bgef_path}')
    # if save_result_bgef:
    #     bc.create_bgef(src_gef_path, extract_bin_size, mask_file_path, dst_bgef_path)
    #     logger.info(f'gef extraction finish, extracted gef save at {dst_bgef_path}')
    # else:
    #     # TODO: has not finished, need recheck
    #     data = StereoExpData(bin_size=1)
    #     uniq_cells, uniq_genes, count, cell_ind, gene_ind = bc.get_stereo_data(
    #         src_gef_path, extract_bin_size, mask_file_path)
    #     logger.info(f'the matrix has {len(uniq_cells)} cells, and {len(uniq_genes)} genes.')

    #     data.position = np.array(
    #         list((zip(np.right_shift(uniq_cells, 32), np.bitwise_and(uniq_cells, 0xffff))))).astype('uint32')
    #     data.offset_x = data.position[0].min()
    #     data.offset_y = data.position[1].min()
    #     data.attr = {
    #         'minX': data.offset_x,
    #         'minY': data.offset_y,
    #         'maxX': data.position[0].max(),
    #         'maxY': data.position[1].max(),
    #         'minExp': count.min(),
    #         'maxExp': count.max(),
    #         'resolution': 0,
    #     }
    #     data.cells = Cell(cell_name=uniq_cells)
    #     data.genes = Gene(gene_name=uniq_genes)
    #     data.exp_matrix = csr_matrix(
    #         (count, (cell_ind, gene_ind)),
    #         shape=(len(uniq_cells), len(uniq_genes)),
    #         dtype=np.uint32
    #     )
    #     return data
    return dst_bgef_path
