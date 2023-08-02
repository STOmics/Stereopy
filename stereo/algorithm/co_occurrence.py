# python core module
from typing import Union
from collections import defaultdict
import time
from natsort import natsorted
from copy import deepcopy
from multiprocessing import cpu_count

# third part module
import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import networkx as nx
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# module in self project
import stereo as st # only used in test function to read data
from stereo.core.stereo_exp_data import StereoExpData, AnnBasedStereoExpData
from stereo.log_manager import logger
from stereo.algorithm.algorithm_base  import AlgorithmBase, ErrorCode


##----------------------------------------------##
## please try the notebook demo in pull requese ##
##----------------------------------------------##

@nb.njit(cache=True, nogil=True)
def _cal_distance(point: np.ndarray, points: np.ndarray):
    return np.sqrt(np.sum((points - point)**2, axis=1))

@nb.njit(cache=True, nogil=True, parallel=True)
def _cal_pairwise_distances(points_a: np.ndarray, points_b: np.ndarray):
    points_count_a = points_a.shape[0]
    points_count_b = points_b.shape[0]
    distance = np.zeros((points_count_a, points_count_b), dtype=np.float64)
    for i in nb.prange(points_count_a):
        distance[i] = _cal_distance(points_a[i], points_b)
    return distance


@nb.njit(cache=True, nogil=True, parallel=True)
def _coo_stereopy_calculator(
        data_position: np.ndarray,
        group_codes: np.ndarray,
        groups: np.ndarray,
        groups_idx: np.ndarray,
        thresh: np.ndarray,
        genelist: np.ndarray = None,
        gene_exp_matrix: np.ndarray = None,
        gene_thresh: float = 0 
    ):
    count_list = np.zeros((thresh.size - 1, group_codes.size), dtype=np.uint64)
    if genelist is None:
        ret_list = np.zeros((thresh.size - 1, group_codes.size, group_codes.size), dtype=np.uint64)
        out = np.zeros((thresh.size - 1, group_codes.size, group_codes.size), dtype=np.float64)
    else:
        ret_list = np.zeros((thresh.size - 1, group_codes.size, genelist.size), dtype=np.uint64)
        out = np.zeros((thresh.size - 1, genelist.size, group_codes.size), dtype=np.float64)

    for ep in nb.prange(thresh.size - 1):
        thresh_l, thresh_r = thresh[ep], thresh[ep+1]
        count = count_list[ep]
        ret = ret_list[ep]
        if genelist is None:
            for i, gidx1 in enumerate(groups_idx):
                dist = _cal_distance(data_position[i], data_position)
                gidx2 = np.unique(groups_idx[(dist >= thresh_l) & (dist < thresh_r)])
                ret[gidx1][gidx2] += np.uint64(1)
                count[gidx1] += np.uint64(1)
            ret = ret.T / count
            out[ep, :, :] = ret
        else:
            for i, gidx in enumerate(groups_idx):
                dist = _cal_distance(data_position[i], data_position)
                flag = np.where((dist >= thresh_l) & (dist < thresh_r), 1, 0)
                gene_exp_flag = np.where(gene_exp_matrix >= gene_thresh, 1, 0).astype(gene_exp_matrix.dtype)
                gene_exp_flag = gene_exp_matrix * flag
                gene_exp_flag = np.sum(gene_exp_flag, axis=1)
                gene_exp_flag = np.where(gene_exp_flag > 0, 1, 0)
                ret[gidx] += gene_exp_flag.astype(np.uint64)
                count[gidx] += np.uint64(1)
            ret=ret.T / count
            out[ep, :, :] = ret
    return out

@nb.njit(cache=True, nogil=True, parallel=True)
def _coo_squidpy_calculator(
    data_position: np.ndarray,
    group_codes: np.ndarray,
    groups_idx: np.ndarray,
    thresh: np.ndarray,
):
    num = group_codes.size
    out = np.zeros((num, num, thresh.shape[0] - 1))
    for ep in nb.prange(thresh.shape[0] - 1):
        co_occur = np.zeros((num, num))
        thresh_l, thresh_r = thresh[ep], thresh[ep+1]
        for x in range(data_position.shape[0]):
            dist = _cal_distance(data_position[x], data_position)
            i = groups_idx[x]
            y = groups_idx[(dist > thresh_l) & (dist <= thresh_r)]
            for j in y:
                co_occur[i, j] += 1

        probs_matrix = co_occur / np.sum(co_occur)
        probs = np.sum(probs_matrix, axis=1)

        probs_con = (co_occur.T / np.sum(co_occur, axis=1) / probs).T

        out[:, :, ep] = probs_con
    return out

class CoOccurrence(AlgorithmBase):
    """
    docstring for CoOccurence
    :param 
    :return: 
    """

    def main(
        self,
        cluster_res_key,
        method='stereopy',
        dist_thres=300,
        steps=10,
        genelist=None,
        gene_thresh=0,
        n_jobs=-1,
        res_key='co_occurrence'
    ):
        """
        Co-occurence calculates the score or probability of two or more cell types in spatial.  
        Stereopy provided two method for co-occurence, 'squidpy' for method in squidpy, 'stereopy' for method in stereopy by default.


        :param cluster_res_key: The key of the cluster or annotation result of cells stored in `data.tl.result` which ought to be equal to cells in length.
        :param method: The method to calculate co-occurence choose from `['stereopy', 'squidpy']`, `'stereopy'` by default.
        :param dist_thres: The max distance to measure co-occurence. Only used when `method='stereopy'`.
        :param steps: The steps to generate threshold to measure co-occurence, use along with dist_thres, i.e. default params
                        will generate [30,60,90......,270,300] as threshold. Only used when `method='stereopy'`.
        :param genelist: Calculate co-occurence between clusters in cluster_res_key & genelist if provided, otherwise calculate between clusters 
                        in cluster_res_key. Only used when `method='stereopy'`.
        :param gene_thresh: Threshold to determine whether a cell expresses targeted gene. Only used when `method='stereopy'`.
        :param n_jobs: The number of threads to calculate co-occurence, default to all cores of the machine.
        :param res_key: The key to store the result in `data.tl.result`.


        :return: StereoExpData object with co_occurrence result in `data.tl.result`.
        """
        if n_jobs <= 0 or n_jobs > cpu_count():
            n_jobs = cpu_count()
        
        current_jobs = nb.get_num_threads()
        nb.set_num_threads(n_jobs)

        try:
            if method == 'stereopy':
                res = self.co_occurrence(self.stereo_exp_data, cluster_res_key, dist_thres = dist_thres, steps = steps, genelist = genelist, gene_thresh = gene_thresh)
            elif method == 'squidpy':
                res = self.co_occurrence_squidpy(self.stereo_exp_data, cluster_res_key)
            else:
                raise ValueError("unavailable value for method, it only can be choosed from ['stereopy', 'squidpy'].")
            self.pipeline_res[res_key] = res
            self.stereo_exp_data.tl.reset_key_record('co_occurrence', res_key)
        finally:
            nb.set_num_threads(current_jobs)
        
        return self.stereo_exp_data


    def co_occurrence_squidpy(
        self,
        data: Union[StereoExpData, AnnBasedStereoExpData],
        use_col: str
    ):
        """
        Squidpy mode to calculate co-occurence, result same as squidpy
        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :return: co_occurrence result, also written in data.tl.result['co-occur']
        """

        thresh_min, thresh_max = self._find_min_max(data.position)
        thresh = np.linspace(thresh_min, thresh_max, num=50)
        if use_col in data.cells:
            groups:pd.Series = data.cells[use_col].astype('category')
        else:
            groups:pd.Series = self.pipeline_res[use_col]['group'].astype('category')
        group_codes = groups.cat.categories.to_numpy().astype('U')
        out = _coo_squidpy_calculator(
            data.position,
            group_codes,
            groups.cat.codes.to_numpy(),
            thresh,
        )
        ret = {}
        for i, j in enumerate(group_codes):
            tmp = pd.DataFrame(out[i]).T
            tmp.columns = group_codes
            tmp.index = thresh[1:]
            ret[j] = tmp
        return ret

    def _find_min_max(self, spatial):
        '''
        Helper to calculate distance threshold in squidpy mode
        param: spatial: the cell position of data
        return: thres_min, thres_max for minimum & maximum of threshold
        '''
        coord_sum = np.sum(spatial, axis=1)
        min_idx, min_idx2 = np.argpartition(coord_sum, 2)[:2]
        max_idx = np.argmax(coord_sum)
        # fmt: off
        thres_max = _cal_pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1))[0, 0] / 2.0
        thres_min = _cal_pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1))[0, 0]
        # fmt: on
        return thres_min, thres_max

    def co_occurrence(
        self,
        data: Union[StereoExpData, AnnBasedStereoExpData],
        use_col,
        dist_thres = 300,
        steps = 10,
        genelist = None,
        gene_thresh = 0
    ):
        '''
        Stereopy mode to calculate co-occurence, the score of result['A']['B'] represent the probablity of 'B' occurence around 
        'A' in distance of threshold

        :param data: An instance of StereoExpData, data.position & data.tl.result[use_col] will be used.
        :param use_col: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :param method: The metrics to calculate co-occurence choose from ['stereopy', 'squidpy'], 'squidpy' by default.
        :param dist_thres: The max distance to measure co-occurence. 
        :param steps: The steps to generate threshold to measure co-occurence, use along with dist_thres, i.e. default params 
                        will generate [30,60,90......,270,300] as threshold. 
        :param genelist: Calculate co-occurence between use_col & genelist if provided, otherwise calculate between clusters 
                        in use_col. 
        :param gene_thresh: Threshold to determine whether a cell express the gene. 
        :return: co_occurrence result, also written in data.tl.result['co-occur']
        '''
        #from collections import defaultdict
        #from scipy import sparse
        # dist_ori = pairwise_distances(data.position, data.position, metric='euclidean')
        # distance = _cal_pairwise_distances(data.position, data.position)
        if isinstance(genelist, np.ndarray):
            genelist = list(genelist)
        elif isinstance(genelist, list):
            genelist = genelist
        elif isinstance(genelist, str):
            genelist = [genelist]
        elif isinstance(genelist, int):
            genelist = [genelist]

        thresh = np.linspace(0, dist_thres, num=steps+1)
        if use_col in data.cells:
            groups:pd.Series = data.cells[use_col].astype('category')
        else:
            groups:pd.Series = self.pipeline_res[use_col]['group'].astype('category')
        group_codes = groups.cat.categories.to_numpy().astype('U')
        gene_exp_matrix = None
        if genelist is not None:
            genelist = np.array(genelist, dtype='U')
            gene_idx = [np.argwhere(data.gene_names == gene_name)[0][0] for gene_name in genelist]
            gene_exp_matrix = data.exp_matrix[:, gene_idx].toarray() if data.issparse() else data.exp_matrix[:, gene_idx]
            gene_exp_matrix = gene_exp_matrix.T
        out = _coo_stereopy_calculator(
            data.position,
            group_codes,
            groups.to_numpy().astype('U'),
            groups.cat.codes.to_numpy(),
            thresh,
            genelist,
            gene_exp_matrix,
            gene_thresh
        )
        ret = {}
        ret_key_list = group_codes if genelist is None else genelist
        for i, ret_key in enumerate(ret_key_list):
            tmp = {}
            for j, th in enumerate(thresh[1:]):
                tmp[th] = out[j][i]
            ret[ret_key] = pd.DataFrame(tmp, index=group_codes).T
        return ret


    def ms_co_occur_integrate(self, ms_data, scope, use_col, res_key='co_occurrence'):
        from collections import Counter, defaultdict
        if use_col not in ms_data.obs:
            tmp_list = []
            for data in ms_data:
                tmp_list.extend(list(data.cells[use_col]))
            ms_data.obs[use_col]=tmp_list
        ms_data.obs[use_col] =  ms_data.obs[use_col].astype('category')

        slice_groups = scope.split('|')
        if len(slice_groups) == 1:
            slices = slice_groups[0].split(",")
            ct_count = {}
            for x in slices:
                ct_count[x] = dict(Counter(ms_data[x].cells[use_col]))

            ct_count = pd.DataFrame(ct_count)
            ct_ratio = ct_count.div(ct_count.sum(axis=1), axis=0)
            ct_ratio = ct_ratio.loc[ms_data.obs[use_col].cat.categories]
            merge_co_occur_ret = ms_data[slices[0]].tl.result[res_key].copy()
            merge_co_occur_ret = {x:y[ms_data.obs[use_col].cat.categories] *0 for x, y  in merge_co_occur_ret.items()}
            for ct in merge_co_occur_ret:
                for x in slices:
                    merge_co_occur_ret[ct] += ms_data[x].tl.result[res_key][ct] * ct_ratio[x]

        elif len(slice_groups) == 2:
            ret = []
            for tmp_slice_groups in slice_groups:
                slices = tmp_slice_groups.split(",")
                ct_count = {}
                for x in slices:
                    ct_count[x] = dict(Counter(ms_data[x].cells[use_col]))

                ct_count = pd.DataFrame(ct_count)
                ct_ratio = ct_count.div(ct_count.sum(axis=1), axis=0)
                ct_ratio = ct_ratio.loc[ms_data.obs[use_col].cat.categories]
                merge_co_occur_ret = ms_data[slices[0]].tl.result[res_key].copy()
                merge_co_occur_ret = {x:y[ms_data.obs[use_col].cat.categories] *0 for x, y  in merge_co_occur_ret.items()}
                for ct in merge_co_occur_ret:
                    for x in slices:
                        merge_co_occur_ret[ct] += ms_data[x].tl.result[res_key][ct] * ct_ratio[x]
                ret.append(merge_co_occur_ret)

            merge_co_occur_ret = {ct:ret[0][ct]-ret[1][ct] for ct in merge_co_occur_ret}

        else:
            print('co-occurrence only compare case and control on two groups')
            merge_co_occur_ret = None

        return merge_co_occur_ret
