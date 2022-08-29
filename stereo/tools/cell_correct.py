# @FileName : cell_correct.py
# @Time     : 2022-05-26 14:14:27
# @Author   : TanLiWei
# @Email    : tanliwei@genomics.cnW
import multiprocessing
import os
import time
import pandas as pd
import numpy as np
from ..algorithm.cell_correction import CellCorrection
from ..algorithm import cell_correction_fast
from .cell_segment import CellSegment
from ..io import read_gem, read_gef
from ..log_manager import logger
from gefpy import cgef_writer_cy, bgef_writer_cy, cgef_adjust_cy


class CellCorrect(object):

    def __init__(self, gem_path=None, bgef_path=None, raw_cgef_path=None, mask_path=None, out_dir=None):
        self.gem_path = gem_path
        self.bgef_path = bgef_path
        self.raw_cgef_path = raw_cgef_path
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.cad = cgef_adjust_cy.CgefAdjust()
        self.gene_names = None
        self.check_input()

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
    
    def get_file_name(self, ext=None):
        ext = ext.lstrip('.') if ext is not None else ""
        if self.bgef_path is not None:
            file_name = os.path.basename(self.bgef_path)
            file_prefix = os.path.splitext(file_name)[0]
        else:
            file_name = os.path.basename(self.gem_path)
            file_prefix = os.path.splitext(file_name)[0]
        if ext == "":
            return file_prefix
        else:
            return f"{file_prefix}.{ext}"


    def generate_bgef(self, threads=10):
        t0 = time.time()
        file_name = self.get_file_name('bgef')
        bgef_path = os.path.join(self.out_dir, file_name)
        logger.info(f"start to generate bgef({bgef_path})")
        if os.path.exists(bgef_path):
            os.remove(bgef_path)
        bgef_writer_cy.generate_bgef(self.gem_path, bgef_path, n_thread=threads, bin_sizes=1)
        t1 = time.time()
        logger.info(f"generate bgef finished : {t1 - t0}")
        return bgef_path
    
    def generate_raw_data(self, sample_n=-1):
        t0 = time.time()
        logger.info("start to generate raw data")
        if self.raw_cgef_path is None:
            file_name = self.get_file_name('raw.cellbin.gef')
            self.raw_cgef_path = os.path.join(self.out_dir, file_name)
            logger.info(f"start to generate raw cgef ({self.raw_cgef_path})")
            if os.path.exists(self.raw_cgef_path):
                os.remove(self.raw_cgef_path)
            cgef_writer_cy.generate_cgef(self.raw_cgef_path, self.bgef_path, self.mask_path, [256, 256])
        
        genes, raw_data = self.cad.get_cell_data(self.bgef_path, self.raw_cgef_path)
        genes = pd.DataFrame(genes, columns=['geneID']).reset_index().rename(columns={'index': 'geneid'})
        raw_data = pd.DataFrame(raw_data.tolist(), dtype='int32').rename(columns={'midcnt': 'UMICount', 'cellid': 'label'})
        raw_data = pd.merge(raw_data, genes, on=['geneid']).reset_index()[['geneID', 'x', 'y', 'UMICount', 'label']]
        if sample_n > 0:
            logger.info(f"sample {sample_n} from raw data")
            raw_data = raw_data.sample(sample_n, replace=False)
        t1 = time.time()
        logger.info(f"generate raw cgef finished : {t1 - t0}")
        return genes, raw_data
    
    def generate_adjusted_cgef(self, adjusted_data, genes):
        t0 = time.time()
        logger.info("start to generate adjusted cgef")
        data = adjusted_data.drop(labels='tag', axis=1).rename(columns={"label": "cellid", "UMICount": "count"})
        data = pd.merge(data, genes, on=['geneID']).sort_values("cellid").reset_index().drop("index", axis=1).reset_index()
        cell_min_ids = data.groupby("cellid").min().reset_index()[["cellid", "index"]].rename(columns={"index": "offset"})
        cell_count = data.groupby("cellid").count().reset_index()[["cellid", "geneID"]].rename(columns={"geneID": "count"})
        cell = pd.merge(cell_min_ids, cell_count, on=['cellid'])
        dnb = data.drop(['index', 'geneID', 'cellid'], axis=1)
        cell_data = list(map(tuple, cell.to_dict("split")['data']))
        dnb_data = list(map(tuple, dnb.to_dict("split")['data']))
        cell_type = np.dtype({'names':['cellid', 'offset', 'count'], 'formats':[np.uint32, np.uint32, np.uint32]})
        dnb_type = np.dtype({'names':['x', 'y', 'count', 'gene_id'], 'formats':[np.int32, np.int32, np.uint16, np.uint16]})
        cell = np.array(cell_data, dtype=cell_type)
        dnb = np.array(dnb_data, dtype=dnb_type)
        file_name = self.get_file_name('adjusted.cellbin.gef')
        adjust_cgef_file = os.path.join(self.out_dir, file_name)
        if os.path.exists(adjust_cgef_file):
            os.remove(adjust_cgef_file)
        self.cad.write_cgef_adjustdata(adjust_cgef_file, cell, dnb)
        t1 = time.time()
        logger.info(f"generate adjusted cgef finished ({adjust_cgef_file}) : {t1 - t0}")
        return adjust_cgef_file
        
    def generate_adjusted_gem(self, adjusted_data):
        file_name = self.get_file_name("adjusted.gem")
        gem_file_adjusted = os.path.join(self.out_dir, file_name)
        adjusted_data.to_csv(gem_file_adjusted, sep="\t", index=False)
        return gem_file_adjusted

    def correcting(self, threshold=20, process_count=10, only_save_result=False, sample_n=-1, fast=False):
        genes, raw_data = self.generate_raw_data(sample_n)
        if not fast:
            correction = CellCorrection(self.mask_path, raw_data, threshold, process_count, err_log_dir=self.out_dir)
            adjusted_data = correction.cell_correct()
        else:
            adjusted_data = cell_correction_fast.cell_correct(raw_data, self.mask_path)
        gem_file_adjusted = self.generate_adjusted_gem(adjusted_data)
        cgef_file_adjusted = self.generate_adjusted_cgef(adjusted_data, genes)
        if not only_save_result:
            return read_gef(cgef_file_adjusted, bin_type='cell_bins')
        else:
            return cgef_file_adjusted
    
def cell_correct(out_dir,
                threshold=20,
                gem_path=None,
                bgef_path=None,
                raw_cgef_path=None,
                mask_path=None,
                image_path=None,
                model_path=None,
                mask_save=True,
                model_type='deep-learning',
                depp_cro_size=20000,
                overlap=100, 
                gpu='-1', 
                process_count=10,
                only_save_result=False,
                sample_n=-1,
                fast=True):
    """correct cells from gem and mask or gem and ssdna image or bgef and mask or bgef and raw cgef(the cgef without correcting)

    :param out_dir: the path of the directory to save some intermediate result like mask(if generate from ssdna image),
                    bgef(generate from gem),cgef(generate from gem and mask) etc. and finally corrected result
    :param threshold: default to 20
    :param gem_path: the path of gem file, if None, need to input bgef_path, defaults to None
    :param bgef_path: the path of bgef file, if None, need to input gem_path and then generate from gem, defaults to None
    :param raw_cgef_path: the path of cgef file contains the data without correcting, if None, generate from bgef and mask, defaults to None
    :param mask_path: the path of mask, if None, need to input the ssdna image by image_path and then generate from ssdna image, defaults to None
    :param image_path: the path of ssdna image , if None, need to input mask_path, defaults to None
    :param model_path: the path of the model used to generate mask, defaults to None
    :param mask_save: if generated mask from ssdna image, set it to True to save mask file after correcting, defaults to True
    :param model_type: the type of model used to generate mask, only can be set to deep-learning or deep-cell, defaults to 'deep-learning'
    :param depp_cro_size: deep crop size, defaults to 20000
    :param overlap: the size of overlap, defaults to 100
    :param gpu: specify the gpu id to predict on gpu when generate mask, if -1, predict on cpu, defaults to '-1'
    :param process_count: the count of the process will be started when correct cells, defaults to 10
    :param only_save_result: if True, only save result to disk; if False, return an object of StereoExpData, defaults to False
    :param fast: if True, it will run more faster and only run by single process, defaults to True
    :return: an object of StereoExpData if only_save_result was set to False or the path of the correct result if only_save_result was set to True
    """
    do_mask_generating = False
    if mask_path is None and image_path is not None:
        do_mask_generating = True
        cell_segment = CellSegment(image_path, gpu, out_dir)
        logger.info(f"there is no mask file, generate it by model {model_path}")
        cell_segment.generate_mask(model_path, model_type, depp_cro_size, overlap)
        mask_path = cell_segment.get_mask_files()[0]
        logger.info(f"the generated mask file {mask_path}")
        
    cc = CellCorrect(gem_path=gem_path, bgef_path=bgef_path, raw_cgef_path=raw_cgef_path, mask_path=mask_path, out_dir=out_dir)
    adjusted_data = cc.correcting(threshold=threshold, process_count=process_count, only_save_result=only_save_result, sample_n=sample_n, fast=fast)
    if do_mask_generating and not mask_save:
        cell_segment.remove_all_mask_files()
    return adjusted_data