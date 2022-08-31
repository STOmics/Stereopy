import sys
from unittest import result
from warnings import filterwarnings
from cv2 import log
filterwarnings('ignore')
import tifffile as tifi
import cv2
import os
import math
import time
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from multiprocessing import Process, Manager, Queue, Lock
from concurrent.futures import ProcessPoolExecutor
import gzip
from ..log_manager import logger

def parse_head(gem):
    """
    %prog <stereomics-seq data>
    return number of header lines
    """
    if gem.endswith('.gz'):
        f = gzip.open(gem, 'rb')
    else:
        f = open(gem, 'rb')

    num_of_header_lines = 0
    for i, l in enumerate(f):
        l = l.decode("utf-8") # read in as binary, decode first
        if not l.startswith('#'): # header lines always start with '#'
            break
        num_of_header_lines += 1

    return num_of_header_lines


def creat_cell_gxp(maskFile, geneFile ,transposition=False):
    """
    %prog <CellMask><Gene expression matrix> <output Path>

    return gene expression matrix under each cell
    """

    logger.info("Loading mask file...")
    mask = tifi.imread(maskFile)

    if transposition:
        mask = mask.T

    _, maskImg = cv2.connectedComponents(mask)

    logger.info("Reading data..")
    type_column = {
        "geneID": 'str',
        "x": np.uint32,
        "y": np.uint32,
        "values": np.uint32,
        "UMICount": np.uint32,
        "MIDCount": np.uint32,
        "MIDCounts": np.uint32
    }

    header = parse_head(geneFile)
    genedf = pd.read_csv(geneFile, header=header, sep='\t', dtype=type_column)
    if "UMICount" in genedf.columns:
        genedf = genedf.rename(columns={'UMICount':'MIDCount'})
    if "MIDCounts" in genedf.columns:
        genedf = genedf.rename(columns={'MIDCounts':'MIDCount'})

    tissuedf = pd.DataFrame()
    dst = np.nonzero(maskImg)

    logger.info("Dumping results...")
    tissuedf['x'] = dst[1] + genedf['x'].min()
    tissuedf['y'] = dst[0] + genedf['y'].min()
    tissuedf['label'] = maskImg[dst]

    res = pd.merge(genedf, tissuedf, on=['x', 'y'], how='left').fillna(0) # keep background data
    return res


class CellCorrection(object):

    def __init__(self, mask_file, gem_file, threshold, process, err_log_dir=None):
        self.mask_file = mask_file
        self.gem_file = gem_file
        self.threshold = threshold
        self.process = process
        self.radius = 50
        self.progress_update_interval = 10
        self.err_log_dir = err_log_dir
        if self.err_log_dir is not None and not os.path.exists(self.err_log_dir):
            os.makedirs(self.err_log_dir)

    def __creat_gxp_data(self):
        if isinstance(self.gem_file, pd.DataFrame):
            data = self.gem_file
        elif isinstance(self.gem_file, str) and os.path.isfile(self.gem_file):
            data = creat_cell_gxp(self.mask_file, self.gem_file, transposition=False)
        else:
            raise Exception("error gem data")
        
        logger.info(f"data size : rows {data.shape[0]}, cols {data.shape[1]}")

        if 'MIDCounts' in data.columns:
            data = data.rename(columns={'MIDCounts': 'UMICount'})
        if 'MIDCount' in data.columns:
            data = data.rename(columns={'MIDCount': 'UMICount'})

        assert 'UMICount' in data.columns
        assert 'x' in data.columns
        assert 'y' in data.columns
        assert 'geneID' in data.columns

        cell_data = data[data.label != 0].copy()
        cell_coor = cell_data.groupby('label').mean()[['x', 'y']].reset_index()

        return data, cell_data, cell_coor

    def gmm_score_func(self, data, data_idx, cell_coor, radius, threshold, p_num, queue, lock, err_log_fp=None):
        # t0 = time.time()
        p_data = []
        count = len(data_idx)
        err_logs = []
        for idx, i in enumerate(data_idx):
            try:
                clf = GaussianMixture(n_components=3, covariance_type='spherical')
                # Gaussian Mixture Model GPU version
                cell_test = data[(data.x < cell_coor.loc[i].x + radius) & (data.x > cell_coor.loc[i].x - radius) 
                                & (data.y > cell_coor.loc[i].y - radius) & (data.y < cell_coor.loc[i].y + radius)]
                # fit GaussianMixture Model
                clf.fit(cell_test[cell_test.label == cell_coor.loc[i].label][['x', 'y', 'UMICount']].values)
                # cell_test_bg = cell_test[cell_test.label == 0]
                # # threshold 20
                # score = pd.Series(-clf.score_samples(cell_test_bg[['x', 'y', 'UMICount']].values))
                cell_test_bg_ori = cell_test[cell_test.label == 0]
                bg_group = cell_test_bg_ori.groupby(['x', 'y']).agg(UMI_max=('UMICount', 'max')).reset_index()
                cell_test_bg = pd.merge(cell_test_bg_ori, bg_group, on=['x', 'y'])		
                # threshold 20
                score = pd.Series(-clf.score_samples(cell_test_bg[['x', 'y', 'UMI_max']].values))
                cell_test_bg['score'] = score.values
                # threshold = self.threshold
                cell_test_bg['label'] = np.where(score < threshold, cell_coor.loc[i].label, 0)
                # used multiprocessing have to save result to file
                p_data.append(cell_test_bg)
            except Exception as e:
                if err_log_fp is not None:
                    err_msg = f"proc {p_num}, cell id {cell_coor.loc[i].label}, {e}\n"
                    err_logs.append(err_msg)

            if len(err_logs) >= 10 or ((idx + 1) == count and len(err_logs) > 0):
                try:
                    lock.acquire()
                    err_log_fp.writelines(err_logs)
                    err_log_fp.flush()
                finally:
                    lock.release()
                    err_logs.clear()

            mod = (idx + 1) % self.progress_update_interval
            if (mod == 0) or (idx + 1 == count):
                c = self.progress_update_interval if mod == 0 else mod
                queue.put((False, p_num, c))

        out = pd.concat(p_data)
        out.drop('UMI_max', axis=1, inplace=True)
        queue.put((True, p_num, out))
        # return out

    def gmm_score(self, data, cell_coor):
        # queues = []
        queue = Queue()
        bars = []
        processes = []
        finished = [False for i in range(self.process)]
        bg_adjust_label = [None for i in range(self.process)]
        qs = math.ceil(len(cell_coor.index) / int(self.process))
        lock = Lock()
        err_log_fp = None
        if self.err_log_dir is not None:
            err_log_path = os.path.join(self.err_log_dir, 'err.log')
            try:
                err_log_fp = open(err_log_path, 'w')
            except:
                pass
        for i in range(self.process):
            idx = np.arange(i * qs, min((i + 1) * qs, len(cell_coor.index)))
            if len(idx) == 0: continue
            # q = Queue()
            # queues.append(q)
            p = Process(target=self.gmm_score_func, args=(data, idx, cell_coor, self.radius, self.threshold, i, queue, lock, err_log_fp))
            p.start()
            processes.append(p)
            bar = tqdm(total=len(idx), desc=f"correcting process-{i}", position=i)
            bars.append(bar)

        while True:
            try:
                res = queue.get()
                flag = res[0]
                pid = res[1]
                result = res[2]
                if flag is True:
                    finished[pid] = True
                    bg_adjust_label[pid] = result
                else:
                    bars[pid].update(result)
            except:
                pass
            if all(finished):
                break
        
        [p.join() for p in processes]
        [bar.close() for bar in bars]
        if err_log_fp:
            err_log_fp.close()
        return bg_adjust_label
    
    def gmm_correction(self, cell_data, bg_adjust_label):
        bg_data = []
        for tmp in bg_adjust_label:
            bg_data.append(tmp[tmp.label != 0])
        adjust_data = pd.concat(bg_data).sort_values('score')
        adjust_data = adjust_data.drop_duplicates(subset=['geneID', 'x', 'y', 'UMICount'], keep='first').rename(columns={'score':'tag'})
        adjust_data['tag'] = 'adjust'
        cell_data['tag'] = 'raw'
        adjust_data['label'] = adjust_data['label'].astype('uint32')
        cell_data['label'] = cell_data['label'].astype('uint32')
        result = pd.concat([adjust_data, cell_data])
        return result

    def cell_correct(self):
        logger.info("start to correct cells!!!")
        t0 = time.time()
        data, cell_data, cell_coor = self.__creat_gxp_data()
        t1 = time.time()
        logger.info(f'Load data :{t1 - t0}')
        bg_adjust_label = self.gmm_score(data, cell_coor)
        t2 = time.time()
        logger.info(f'Calc score :{t2 - t1}')
        result = self.gmm_correction(cell_data, bg_adjust_label)
        t3 = time.time()
        logger.info(f'Correct :{t3 - t2}')
        logger.info(f'Total :{t3 - t0}')
        return result