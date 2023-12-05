# @FileName : cell_correct.py
# @Time     : 2022-05-26 14:14:27
# @Author   : TanLiWei
# @Email    : tanliwei@genomics.cnW
import os
import time
from multiprocessing import cpu_count
from typing import Literal

import numba
import numpy as np
import pandas as pd
from gefpy import (
    cgef_writer_cy,
    bgef_writer_cy,
    cgef_adjust_cy,
    gef_to_gem_cy
)

from ..algorithm import cell_correction_fast
from ..algorithm import cell_correction_fast_by_mask
from ..algorithm.cell_correction import CellCorrection
from ..algorithm.draw_contours import DrawContours
from ..io import read_gef
from ..log_manager import logger
from ..utils.time_consume import TimeConsume
from ..utils.time_consume import log_consumed_time


@log_consumed_time
@numba.njit(cache=True, parallel=True, nogil=True)
def generate_cell_and_dnb(adjusted_data: np.ndarray):
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

    def __init__(self, gem_path=None, bgef_path=None, raw_cgef_path=None, mask_path=None, out_dir=None):
        self.tc = TimeConsume()
        self.gem_path = gem_path
        self.bgef_path = bgef_path
        self.raw_cgef_path = raw_cgef_path
        self.mask_path = mask_path
        self.new_mask_path = None
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
        logger.info(
            f"generate cellbin gef finished, time consumed : {self.tc.get_time_consumed(key=tk, restart=False)}")
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
        cell_type = np.dtype({
            'names': ['cellid', 'offset', 'count'],
            'formats': [np.uint32, np.uint32, np.uint32]
        }, align=True)
        dnb_type = np.dtype({
            'names': ['x', 'y', 'count', 'gene_id'],
            'formats': [np.int32, np.int32, np.uint16, np.uint32]
        }, align=True)
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
        columns = ['geneID', 'x', 'y', 'UMICount', 'label']
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

    def __set_processes_count(self, process_count, method):
        if process_count is not None:
            if not isinstance(process_count, int):
                raise TypeError("the type of prameter 'process_count' must be int.")

        if method == 'GMM':
            if process_count is None or process_count == 0:
                process_count = 10 if cpu_count() > 10 else cpu_count()
            elif process_count < 0 or process_count > cpu_count():
                process_count = cpu_count()
        elif method == 'FAST':
            process_count = 1
        elif method == 'EDM':
            if process_count is None or process_count == 0:
                process_count = 1
            elif process_count < 0 or process_count > cpu_count():
                process_count = cpu_count()
        else:
            pass
        return process_count

    @log_consumed_time
    def correcting(self,
                   threshold=20,
                   process_count=None,
                   only_save_result=False,
                   sample_n=-1,
                   method='EDM',
                   distance=10,
                   **kwargs
                   ):
        if method is None:
            method = 'EDM'
        method = method.upper()
        process_count = self.__set_processes_count(process_count, method)
        if method in ('GMM', 'FAST'):
            genes, raw_data = self.generate_raw_data(sample_n)
        if method == 'GMM':
            correction = CellCorrection(self.mask_path, raw_data, threshold, process_count, err_log_dir=self.out_dir)
            adjusted_data = correction.cell_correct()
        elif method == 'FAST':
            adjusted_data = cell_correction_fast.cell_correct(raw_data, self.mask_path)
        elif method == 'EDM':
            n_split_data_jobs = kwargs.get('n_split_data_jobs', -1)
            new_mask_path = cell_correction_fast_by_mask.main(
                self.mask_path,
                n_jobs=process_count,
                distance=distance,
                out_path=self.out_dir,
                n_split_data_jobs=n_split_data_jobs
            )
            cgef_file_adjusted = self.generate_cgef_with_mask(new_mask_path, 'adjusted')
        else:
            raise ValueError(
                f"Unexpected value({method}) for parameter method, available values include ['GMM', 'FAST', 'EDM'].")

        if method in ('GMM', 'FAST'):
            dc = DrawContours(adjusted_data, self.out_dir)
            outline_path = dc.get_contours()
            cgef_file_adjusted = self.generate_adjusted_cgef(adjusted_data, outline_path)

        if not only_save_result:
            return read_gef(cgef_file_adjusted, bin_type='cell_bins')
        else:
            return cgef_file_adjusted


@log_consumed_time
def cell_correct(out_dir: str,
                 threshold: int = 20,
                 gem_path: str = None,
                 bgef_path: str = None,
                 raw_cgef_path: str = None,
                 mask_path: str = None,
                 process_count: int = None,
                 only_save_result: bool = False,
                 method: Literal['GMM', 'FAST', 'EDM'] = 'EDM',
                 distance: int = 10,
                 **kwargs
                 ):
    """
    Correct cells using one of file conbinations as following:
        * GEM and mask
        * BGEF and mask
        * GEM and raw CGEF (not corrected)
        * BGEF and raw CGEF (not corrected)

    :param out_dir: the path to save intermediate result, like mask (if generated from ssDNA image),
        BGEF (generated from GEM), CGEF (generated from GEM and mask), etc. and final corrected result.
    :param threshold: threshold size, default to 20.
    :param gem_path: the path to GEM file.
    :param bgef_path: the path to BGEF file.
    :param raw_cgef_path: the path to CGEF file which not has been corrected.
    :param mask_path: the path to mask file.
    :param process_count: the count of the processes or threads will be started when correct cells, defaults to None
                by default, it will be set to 10 when `method` is set to 'GMM' and will be set to 1 when `method` is set to 'FAST' or 'EDM'.
                if it is set to -1, all of the cores will be used.
	:param only_save_result: if `True`, only save result to disk; if `False`, return an StereoExpData object.
    :param method: correct in different method if `method` is set, otherwise `EDM`.
    :param distance: outspread distance based on cellular contour of cell segmentation image, in pixels, only available for 'EDM' method.

    :return: An StereoExpData object if `only_save_result` is set to `False`, 
                otherwise the path of corrected CGEF file.
    """  # noqa

    cc = CellCorrect(gem_path=gem_path, bgef_path=bgef_path, raw_cgef_path=raw_cgef_path, mask_path=mask_path,
                     out_dir=out_dir)
    return cc.correcting(threshold=threshold, process_count=process_count, only_save_result=only_save_result,
                         method=method, distance=distance, **kwargs)
