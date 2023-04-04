# @FileName : cell_correct.py
# @Time     : 2022-05-26 14:14:27
# @Author   : TanLiWei
# @Email    : tanliwei@genomics.cnW
import os
import time
from typing import Union
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
import numba
from ..algorithm.cell_correction import CellCorrection
from ..algorithm import cell_correction_fast
from ..algorithm import cell_correction_fast_by_mask
from ..algorithm.draw_contours import DrawContours
from ..io import read_gem, read_gef
from ..log_manager import logger
from gefpy import cgef_writer_cy, bgef_writer_cy, cgef_adjust_cy, gef_to_gem_cy
from ..utils.time_consume import TimeConsume, log_consumed_time
from .gem_filter import gem_filter

@log_consumed_time
@numba.njit(cache=True, parallel=True, nogil=True)
def generate_cell_and_dnb(adjusted_data: np.ndarray):
    # ['x', 'y', 'UMICount', 'label', 'geneid']
    cells_list = adjusted_data[:, 3]
    cells_idx_sorted = np.argsort(cells_list)
    adjusted_data = adjusted_data[cells_idx_sorted]
    cell_data = []
    dnb_data = []
    last_cell = -1
    cellid = -1
    offset = -1
    count = -1
    for i, row in enumerate(adjusted_data):
        current_cell = row[3]
        if current_cell != last_cell:
            if last_cell >= 0:
                cell_data.append((cellid, offset, count))
            cellid, offset, count = current_cell, i, 1
            last_cell = current_cell
        else:
            count += 1
        dnb_data.append((row[0], row[1], row[2], row[4]))
    cell_data.append((cellid, offset, count))
    return cell_data, dnb_data

class CellCorrect(object):

    def __init__(self, gem_path=None, bgef_path=None, raw_cgef_path=None, mask_path=None, tissue_mask_path=None, out_dir=None,):
        self.tc = TimeConsume()
        self.gem_path = gem_path
        self.bgef_path = bgef_path
        self.raw_cgef_path = raw_cgef_path
        self.mask_path = mask_path
        self.new_mask_path = None
        self.tissue_mask_path = tissue_mask_path
        self.out_dir = out_dir
        self.cad = cgef_adjust_cy.CgefAdjust()
        self.gene_names = None
        self.check_input()
        self.sn = self.get_sn()

    def check_input(self):
        if self.bgef_path is None and self.gem_path is None:
            raise Exception("must to input gem file or bgef file")

        if self.out_dir is None:
            now = time.strftime("%Y%m%d%H%M%S")
            self.out_dir = f"./cell_correct_result_{now}"
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
        if self.bgef_path is None:
            self.bgef_path = self.generate_bgef()
    
    def get_sn(self):
        if self.bgef_path is not None:
            file_name = os.path.basename(self.bgef_path)
        else:
            file_name = os.path.basename(self.gem_path)
        return file_name.split('.')[0]

    def get_file_name(self, ext=None):
        ext = ext.lstrip('.') if ext is not None else ""
        if self.bgef_path is not None:
            file_name = os.path.basename(self.bgef_path)
        else:
            file_name = os.path.basename(self.gem_path)
        file_prefix = file_name.split('.')[0]
        if ext == "":
            return file_prefix
        else:
            return f"{file_prefix}.{ext}"


    @log_consumed_time
    def generate_bgef(self, threads=10):
        file_name = self.get_file_name('bgef')
        bgef_path = os.path.join(self.out_dir, file_name)
        if os.path.exists(bgef_path):
            os.remove(bgef_path)
        bgef_writer_cy.generate_bgef(self.gem_path, bgef_path, n_thread=threads, bin_sizes=[1])
        return bgef_path
    
    def generate_cgef_with_mask(self, mask_path, ext_in_ext):
        file_name = self.get_file_name(f'{ext_in_ext}.cellbin.gef')
        cgef_path = os.path.join(self.out_dir, file_name)
        logger.info(f"start to generate cellbin gef ({cgef_path})")
        if os.path.exists(cgef_path):
            os.remove(cgef_path)
        tk = self.tc.start()
        cgef_writer_cy.generate_cgef(cgef_path, self.bgef_path, mask_path, [256, 256])
        logger.info(f"generate cellbin gef finished, time consumed : {self.tc.get_time_consumed(key=tk, restart=False)}")
        return cgef_path

    def get_data_from_bgef_and_cgef(self, bgef_path, cgef_path, sample_n=-1):
        tk = self.tc.start()
        logger.info("start to get data from bgef and cgef")
        genes, data = self.cad.get_cell_data(bgef_path, cgef_path)
        logger.info(f"get data finished, time consumed : {self.tc.get_time_consumed(tk)}")
        genes = pd.DataFrame(genes, columns=['geneID']).reset_index().rename(columns={'index': 'geneid'})
        data = pd.DataFrame(data.tolist(), dtype='int32').rename(columns={'midcnt': 'UMICount', 'cellid': 'label'})
        data = pd.merge(data, genes, on=['geneid'])[['geneID', 'x', 'y', 'UMICount', 'label', 'geneid']]
        if sample_n > 0:
            logger.info(f"sample {sample_n} from raw data")
            data = data.sample(sample_n, replace=False)
        logger.info(f"merged genes to data, time consumed : {self.tc.get_time_consumed(tk)}")

        return genes, data
    
    @log_consumed_time
    def generate_raw_data(self, sample_n=-1):
        if self.raw_cgef_path is None:
            self.raw_cgef_path = self.generate_cgef_with_mask(self.mask_path, 'raw')
        
        logger.info("start to generate raw data")
        genes, raw_data = self.get_data_from_bgef_and_cgef(self.bgef_path, self.raw_cgef_path, sample_n=sample_n)
        # self.generate_raw_gem(raw_data)
        return genes, raw_data

    @log_consumed_time
    def generate_raw_gem(self, raw_data: pd.DataFrame):
        file_name = self.get_file_name('raw.gem')
        raw_gem_path = os.path.join(self.out_dir, file_name)
        raw_data.to_csv(raw_gem_path, sep="\t", index=False, columns=['geneID', 'x', 'y', 'UMICount', 'label'])

    @log_consumed_time
    def generate_adjusted_cgef(self, adjusted_data: pd.DataFrame, outline_path):
        adjusted_data_np = adjusted_data[['x', 'y', 'UMICount', 'label', 'geneid']].to_numpy(dtype=np.uint32)
        cell_data, dnb_data = generate_cell_and_dnb(adjusted_data_np)
        cell_type = np.dtype({'names':['cellid', 'offset', 'count'], 'formats':[np.uint32, np.uint32, np.uint32]}, align=True)
        dnb_type = np.dtype({'names':['x', 'y', 'count', 'gene_id'], 'formats':[np.int32, np.int32, np.uint16, np.uint32]}, align=True)
        cell = np.array(cell_data, dtype=cell_type)
        dnb = np.array(dnb_data, dtype=dnb_type)
        file_name = self.get_file_name('adjusted.cellbin.gef')
        cgef_file_adjusted = os.path.join(self.out_dir, file_name)
        if os.path.exists(cgef_file_adjusted):
            os.remove(cgef_file_adjusted)
        if outline_path is not None:
            self.cad.write_cgef_adjustdata(cgef_file_adjusted, cell, dnb, outline_path)
        else:
            self.cad.write_cgef_adjustdata(cgef_file_adjusted, cell, dnb)
        logger.info(f"generate adjusted cellbin gef finished ({cgef_file_adjusted})")
        return cgef_file_adjusted
    
    @log_consumed_time
    def generate_adjusted_gem(self, adjusted_data: pd.DataFrame):
        file_name = self.get_file_name("adjusted.gem")
        gem_file_adjusted = os.path.join(self.out_dir, file_name)
        columns=['geneID', 'x', 'y', 'UMICount', 'label']
        if 'tag' in adjusted_data.columns:
            columns.append('tag')
        adjusted_data.to_csv(gem_file_adjusted, sep="\t", index=False, columns=columns)
        logger.info(f"generate adjusted gem finished ({gem_file_adjusted})")
        return gem_file_adjusted
    
    @log_consumed_time
    def cgef_to_gem(self, cgef_path):
        file_name = self.get_file_name("adjusted.gem")
        gem_file_adjusted = os.path.join(self.out_dir, file_name)
        obj = gef_to_gem_cy.gefToGem(gem_file_adjusted, self.sn)
        obj.cgef2gem(cgef_path, self.bgef_path)
        return gem_file_adjusted

    @log_consumed_time
    def bgef_to_gem(self, mask_path):
        file_name = self.get_file_name("adjusted.gem")
        gem_file_adjusted = os.path.join(self.out_dir, file_name)
        obj = gef_to_gem_cy.gefToGem(gem_file_adjusted, self.sn)
        obj.bgef2cgem(mask_path, self.bgef_path)

    def __set_processes_count(self, process_count, fast):
        if process_count is not None:
            if not isinstance(process_count, int):
                raise TypeError("the type of prameter 'process_count' must be int.")
            
        if isinstance(fast, bool) and (not fast):
            if process_count is None or process_count == 0:
                process_count = 10 if cpu_count() > 10 else cpu_count()
            elif process_count < 0 or process_count > cpu_count():
                process_count = cpu_count()
        else:
            if isinstance(fast, bool) and fast:
                fast = 'v2'
            if fast == 'v1':
                process_count = 1
            elif fast == 'v2':
                if process_count is None or process_count == 0:
                    process_count = 1
                elif process_count < 0 or process_count > cpu_count():
                    process_count = cpu_count()
            else:
                process_count = 1 if process_count is None else process_count
        return process_count
    
    def __set_contours_fitting(self, contours_fitting, fast):
        if contours_fitting is not None:
            if not isinstance(contours_fitting, bool):
                raise TypeError("the type of parameter 'contours_fitting' must be bool.")

        if isinstance(fast, bool) and (not fast):
            if contours_fitting is None:
                contours_fitting = True
        else:
            if isinstance(fast, bool) and fast:
                fast = 'v2'
            if fast == 'v1':
                if contours_fitting is None:
                    contours_fitting = True
            elif fast == 'v2':
                if contours_fitting is None:
                    contours_fitting = False
            else:
                contours_fitting = True if contours_fitting is None else contours_fitting
        return contours_fitting


    @log_consumed_time
    def correcting(self, threshold=20, process_count=None, only_save_result=False, sample_n=-1, fast=True, **kwargs):
        if isinstance(fast, bool) and fast:
            fast = 'v2'
        elif isinstance(fast, str):
            fast = fast.lower()
        process_count = self.__set_processes_count(process_count, fast)
        if fast is False or fast == 'v1':
            genes, raw_data = self.generate_raw_data(sample_n)
            if self.tissue_mask_path is not None:
                raw_data = gem_filter(self.tissue_mask_path, raw_data)
        if fast is False:
            correction = CellCorrection(self.mask_path, raw_data, threshold, process_count, err_log_dir=self.out_dir)
            adjusted_data = correction.cell_correct()
        elif fast == 'v1':
            adjusted_data = cell_correction_fast.cell_correct(raw_data, self.mask_path)
        elif fast == 'v2':
            new_kwargs = {}
            if 'n_split_data_jobs' in kwargs:
                new_kwargs['n_split_data_jobs'] = kwargs['n_split_data_jobs']
                del kwargs['n_split_data_jobs']
            if 'distance' in kwargs:
                new_kwargs['distance'] = kwargs['distance']
                del kwargs['distance']
            new_mask_path = cell_correction_fast_by_mask.main(self.mask_path, n_jobs=process_count, out_path=self.out_dir, **new_kwargs)
            cgef_file_adjusted = self.generate_cgef_with_mask(new_mask_path, 'adjusted')
            gem_file_adjusted = self.cgef_to_gem(cgef_file_adjusted)
            # gem_file_adjusted = self.bgef_to_gem(new_mask_path)
            # genes, adjusted_data_dirty = self.get_data_from_bgef_and_cgef(self.bgef_path, new_cgef_path)
            # logger.info(f"adjusted_data_dirty.shape = {adjusted_data_dirty.shape}")
            # adjusted_data = adjusted_data_dirty[adjusted_data_dirty['label'] != 0]
            # logger.info(f"adjusted_data.shape = {adjusted_data.shape}")
        else:
            raise ValueError(f"Unexpected value({fast}) for parameter fast, available values include [False, 'v1', 'v2'].")

        if fast == 'v2':
            pass
        else:
            gem_file_adjusted = self.generate_adjusted_gem(adjusted_data)
            dc = DrawContours(adjusted_data, self.out_dir)
            outline_path = dc.get_contours()
            cgef_file_adjusted = self.generate_adjusted_cgef(adjusted_data, outline_path)

        if not only_save_result:
            return read_gef(cgef_file_adjusted, bin_type='cell_bins')
        else:
            return cgef_file_adjusted

@log_consumed_time    
def cell_correct(out_dir: str,
                threshold: int=20,
                gem_path: str=None,
                bgef_path: str=None,
                raw_cgef_path: str=None,
                mask_path: str=None,
                image_path: str=None,
                model_path: str=None,
                mask_save: bool=True,
                model_type: str='deep-learning',
				gpu: str='-1',
                process_count: int=None,
                only_save_result: bool=False,
                fast: Union[bool, str]='v2',
                tissue_mask_path: str=None,
                **kwargs
):

    """
    Correct cells using one of file conbinations as following:
        * GEM and mask
        * GEM and ssDNA image
        * BGEF and mask
        * BGEF and ssDNA image
        * GEM and raw CGEF (not have been corrected)
        * BGEF and raw CGEF (not have been corrected)

    :param out_dir: the path to save intermediate result, like mask (if generate from ssDNA image), 
        BGEF (generate from GEM), CGEF (generate from GEM and mask), etc. and final corrected result.
    :param threshold: threshold size, default to 20.
    :param gem_path: the path to GEM file.
    :param bgef_path: the path to BGEF file.
    :param raw_cgef_path: the path to CGEF file which not has been corrected.
    :param mask_path: the path to mask file.
    :param image_path: the path to ssDNA image file.
    :param model_path: the path to model file.
    :param mask_save: whether to save mask file after correction, generated from ssDNA image.
    :param model_type: the type of model to generate mask, whcih only could be set to deep learning model and deep cell model.
	:param gpu: specify gpu id to predict when generate mask, if `'-1'`, cpu is used.
    :param process_count: the count of the process will be started when correct cells, defaults to None
                by default, it will be set to 10 when `fast` is set to False and will be set to 1 when `fast` is set to True, v1 or v2.
                if it is set to -1, all of the cores will be used.
	:param only_save_result: if `True`, only save result to disk; if `False`, return an StereoExpData object.
    :param fast: specify the version of algorithm, available values include [False, v1, v2], defaults to v2.
                    False: the oldest and slowest version, it will uses multiprocessing if set `process_count` to more than 1.
                    v1: the first fast version, it olny uses single process and single threading.
                    v2: default and recommended algorithm, the latest fast version, faster and more accurate than v1, it will uses multithreading if set `process_count` to more than 1.
    :param tissue_mask_path: the path of tissue mask, default to None.
                    if it is set, the data will be filterd by tissue_mask, unavailable if set the `fast` to v2.
    :param deep_cro_size: deep crop size.
    :param overlap: overlap size.

    :return: An StereoExpData object if `only_save_result` is set to `False`, otherwise none.
    """
    do_mask_generating = False
    if mask_path is None and image_path is not None:
        from .cell_segment import CellSegment
        do_mask_generating = True
        deep_cro_size = kwargs.get('deep_cro_size', 20000)
        overlap = kwargs.get('overlap', 100)
        if 'deep_cro_size' in kwargs:
            del kwargs['deep_cro_size']
        if 'overlap' in kwargs:
            del kwargs['overlap']
        cell_segment = CellSegment(image_path, gpu, out_dir)
        logger.info(f"there is no cell mask file, generate it by model {model_path}")
        cell_segment.generate_mask(model_path, model_type, deep_cro_size, overlap)
        mask_path = cell_segment.get_mask_files()[0]
        logger.info(f"the generated cell mask file {mask_path}")
        
    cc = CellCorrect(gem_path=gem_path, bgef_path=bgef_path, raw_cgef_path=raw_cgef_path, mask_path=mask_path, tissue_mask_path=tissue_mask_path, out_dir=out_dir)
    adjusted_data = cc.correcting(threshold=threshold, process_count=process_count, only_save_result=only_save_result, fast=fast, **kwargs)
    if do_mask_generating and not mask_save:
        cell_segment.remove_all_mask_files()
    return adjusted_data