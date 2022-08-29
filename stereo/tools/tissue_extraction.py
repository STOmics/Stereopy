"""
Tissue Extraction from bGEF
"""
from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from gefpy.bgef_creater_cy import BgefCreater

from ..core.cell import Cell
from ..core.gene import Gene
from ..core import StereoExpData
from ..log_manager import logger
from ..image.tissue_cut import ssDNA, RNA, DEEP, SingleStrandDNATissueCut, RNATissueCut


def tissue_extraction_to_bgef(
        src_gef_path: Optional[str],
        dst_bgef_path: Optional[str],
        dst_mask_dir_path: Optional[str],
        src_type: Optional[int] = ssDNA,
        seg_method: Optional[int] = DEEP,
        model_path: Optional[str] = "",
        src_img_path: Optional[str] = "",
        rna_tissue_cut_bin_size: Optional[int] = 20,
        save_result_bgef: Optional[bool] = True,
):
    """
    :param src_gef_path: the gef will be extracted by mask tif from tissue cut
    :param dst_bgef_path: result bgef path
    :param dst_mask_dir_path: save tissue cut result mask tif to this directory
    :param src_type: default to ssDNA with deep learn method, which means `src_img_path` `model_path` are mandatory
    :param seg_method: default to `DEEP`, see desc in `src_type`
    :param model_path: see desc in `src_type`
    :param src_img_path: see desc in `src_type`
    :param rna_tissue_cut_bin_size: default to 20, using bin20 for better performance but less mask fineness
    :param save_result_bgef: True return None, but save gef to disk; False for returning a `StereoExpData`
    :return: None or StereoExpData
    """
    if src_type == ssDNA:
        tissue_cut_obj = SingleStrandDNATissueCut(
            seg_method=seg_method,
            src_img_path=src_img_path,
            dst_img_path=dst_mask_dir_path,
            model_path=model_path
        )
        extract_bin_size = 1
    elif src_type == RNA:
        tissue_cut_obj = RNATissueCut(
            dst_img_path=dst_mask_dir_path,
            gef_path=src_gef_path,
            bin_size=rna_tissue_cut_bin_size
        )
        extract_bin_size = rna_tissue_cut_bin_size
    else:
        logger.error(f'{src_type} is not a tissue-cut-src_type')
        raise Exception

    # real do the image transforming
    tissue_cut_obj.tissue_seg()

    # TODO: mask file is dump to disk, which should be a option for user to choose
    mask_file_path = tissue_cut_obj.dst_img_file_path[-1]
    logger.info(f'tissue_cut finish, mask file is saved at {mask_file_path}')

    # gef extraction
    bc = BgefCreater()
    if save_result_bgef:
        bc.create_bgef(src_gef_path, extract_bin_size, mask_file_path, dst_bgef_path)
        logger.info(f'gef extraction finish, extracted gef save at {dst_bgef_path}')
    else:
        # TODO: has not finished, need recheck
        data = StereoExpData(bin_size=1)
        uniq_cells, uniq_genes, count, cell_ind, gene_ind = bc.get_stereo_data(src_gef_path, extract_bin_size, mask_file_path)
        logger.info(f'the matrix has {len(uniq_cells)} cells, and {len(uniq_genes)} genes.')

        data.position = np.array(list((zip(np.right_shift(uniq_cells, 32), np.bitwise_and(uniq_cells, 0xffff))))).astype('uint32')
        data.offset_x = data.position[0].min()
        data.offset_y = data.position[1].min()
        data.attr = {
            'minX': data.offset_x,
            'minY': data.offset_y,
            'maxX': data.position[0].max(),
            'maxY': data.position[1].max(),
            'minExp': count.min(),
            'maxExp': count.max(),
            'resolution': 0,
        }
        data.cells = Cell(cell_name=uniq_cells)
        data.genes = Gene(gene_name=uniq_genes)
        data.exp_matrix = csr_matrix((count, (cell_ind, gene_ind)), shape=(len(uniq_cells), len(uniq_genes)), dtype=np.uint32)
        return data
