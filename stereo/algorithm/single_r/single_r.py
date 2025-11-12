import time
from collections import defaultdict
from operator import itemgetter

import numba
import numpy as np
import pandas as pd
import scipy
from joblib import (
    Parallel,
    delayed,
    cpu_count
)
from tqdm import tqdm

from stereo.core.stereo_exp_data import StereoExpData
from stereo.log_manager import logger
from .utils import (
    corr_spearman,
    apply_along_axis
)
from ..algorithm_base import AlgorithmBase


class _TestData(object):

    def __init__(self, exp_matrix, cell_names, gene_names):
        self.exp_matrix = exp_matrix
        self.cell_names = cell_names
        self.gene_names = gene_names

    def to_df(self):
        return pd.DataFrame(self.exp_matrix.toarray(), index=self.cell_names, columns=self.gene_names)


@numba.jit(forceobj=True, nogil=True, parallel=True)
def _spearman_parallel(label, ranked_mat_ref, y, quantile):
    ranked_mat_qry = apply_along_axis(y)
    sim = corr_spearman(ranked_mat_ref, ranked_mat_qry)
    return label, np.percentile(sim, quantile, axis=1)


def _get_label_genes_set(top_labels, trained_data, de_n_len=0):
    de_n = np.round(500 * (2 / 3) ** np.log2(de_n_len if de_n_len else len(top_labels)))
    genes_filtered = []
    for i in top_labels:
        for j in top_labels:
            if i == j:
                continue
            s = trained_data[i][j]
            s = s.loc[s[0] > 0]
            genes_filtered.append(s.index[:int(de_n)])
    return genes_filtered


class SingleR(AlgorithmBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_jobs = None
        self.quantile = None
        self.ref_exp_data = None
        self.fine_tune_times = None
        self.group_data_frame = None
        self.fine_tune_threshold = None

    def main(
            self,
            ref_exp_data: StereoExpData,
            ref_use_col: str = "celltype",
            cluster_res_key: str = None,
            quantile: int = 80,
            fine_tune_threshold: float = 0.05,
            fine_tune_times: int = 0,
            n_jobs: int = int(cpu_count() / 2),
            res_key: str = 'annotation',
            method: str = 'default',
            gpuid: int = 0
    ):
        """
        Single-cell recognition is a tool to automatically annotate a test sample by a reference sample.

        :param ref_exp_data: a `StereoExpData` as reference data.
        :param ref_use_col: `ref_use_col` mean cluster-like or annotation result key in ref's `StereoExpData.tl.result`.
        :param cluster_res_key: test's cluster-like key in `StereoExpData.tl.result`, If cluster_res_key is set, CPU will be used by default regardless of whether gpu acceleration is set or not.
        :param quantile: quantile will influence scoring and fine_tune result.
        :param fine_tune_threshold: while in fine_tuning, if result greater than `max(result) - fine_tune_threshold`, will be filtered.
        :param fine_tune_times: default to 0, meaning that it will fine_tune until results decreasing to only 1.
                                If it is set to num(eg: 5), it will only loop only 5 times, and choose the first one.
        :param n_jobs: `joblib` parameter, will create `n_jobs` num of threads to work.
        :param res_key: default to `annotation`, means the result will be stored as key `annotation` in the `tl.result`.
        :param methods: whether to use GPU acceleration, if methods is `rapids`, it means using, It is not used by default.
        :param gpuid: slots used by gpu, default to 0.

        :return: `pandas.DataFrame`
        """  # noqa
        temp_res = set(self.stereo_exp_data.gene_names) & set(ref_exp_data.gene_names)
        interact_genes = [gene for gene in self.stereo_exp_data.gene_names.tolist() if gene in temp_res]
        assert interact_genes, "no gene of `test_exp_data.gene_names` in `ref_exp_data.gene_names`"

        total_start_time = time.time()
        test_exp_data = self.stereo_exp_data.sub_by_name(gene_name=interact_genes)
        ref_exp_data = ref_exp_data.sub_by_name(gene_name=interact_genes)

        if not cluster_res_key and method == 'rapids':
            try:
                from cusingler import cusingler
            except ImportError:
                logger.warning("The cusingler dosen't be installed, it will use cpu to process!")
                method = 'default'
        if not cluster_res_key and method == 'rapids':
            from cusingler import cusingler
            self.check_gpu_env()

            qry_data = test_exp_data.exp_matrix.data.tolist()
            qry_indices = test_exp_data.exp_matrix.indices.tolist()
            qry_indptr = test_exp_data.exp_matrix.indptr.tolist()
            ref_geneidx = ref_exp_data.gene_names.tolist()
            temp_ref_exp_data = ref_exp_data.array2sparse()
            ref_data = temp_ref_exp_data.data.tolist()
            ref_indices = temp_ref_exp_data.indices.tolist()
            ref_indptr = temp_ref_exp_data.indptr.tolist()
            ref_height, ref_width = temp_ref_exp_data.shape
            qry_height, qry_width = test_exp_data.shape
            cellnames = test_exp_data.cell_names.tolist()
            qry_geneidx = test_exp_data.gene_names.tolist()
            codes, celltypes = self.load_cell_types(ref_exp_data, ref_use_col)
            res = np.array(
                cusingler.run(
                    n_jobs, gpuid, ref_height, ref_width, ref_data, ref_indices, ref_indptr,
                    qry_height, qry_width, qry_data, qry_indices, qry_indptr,
                    codes, celltypes, cellnames, ref_geneidx, qry_geneidx,
                    quantile=quantile, finetune_thre=fine_tune_threshold, finetune_times=fine_tune_times
                )
            )
            bins = res[:, 0].tolist()
            labels_array = res[:, 1].tolist()
            ret_labels = res[:, 2].tolist()
        else:
            assert ref_use_col in ref_exp_data.tl.result
            self.group_data_frame = ref_exp_data.tl.result[ref_use_col]
            self.ref_gene_names = ref_exp_data.to_df().columns
            self.test_gene_names = {name: i for i, name in enumerate(test_exp_data.gene_names.tolist())}
            self.group_data_frame = self.group_data_frame.reset_index()
            self.group_data_frame['bins'] = self.group_data_frame.index
            self._group_data_frame_checker()

            self.ref_exp_data = _TestData(
                ref_exp_data.exp_matrix if scipy.sparse.issparse(ref_exp_data.exp_matrix) else scipy.sparse.csr_matrix(
                    ref_exp_data.exp_matrix),
                np.array(range(len(ref_exp_data.cells.cell_name))),
                np.array(range(len(ref_exp_data.genes.gene_name)))
            )

            self.n_jobs = n_jobs
            self.quantile = quantile
            self.fine_tune_times = fine_tune_times
            self.fine_tune_threshold = fine_tune_threshold

            logger.info(f'start single-r with n_jobs={n_jobs} fine_tune_times={fine_tune_times}')

            logger.info('start training ref...')
            start_time = time.time()
            trained_data, common_gene = self._train_ref()
            logger.info(f'training ref finished, cost {time.time() - start_time} seconds')

            if cluster_res_key:
                tmp_exp_matrix = pd.DataFrame(
                    test_exp_data.exp_matrix.todense() if scipy.sparse.issparse(
                        test_exp_data.exp_matrix) else test_exp_data.exp_matrix,
                    index=test_exp_data.cell_names,
                    columns=test_exp_data.gene_names
                )
                test_cluster_result = test_exp_data.tl.result[cluster_res_key]
                test_group_data_frame = test_cluster_result['group']
                if 'bins' in test_cluster_result:
                    test_group_data_frame.index = test_cluster_result['bins'].values
                tmp_exp_matrix = tmp_exp_matrix.groupby(test_group_data_frame).sum()
                test_data = _TestData(
                    scipy.sparse.csr_matrix(tmp_exp_matrix.values),
                    tmp_exp_matrix.index,
                    np.array(range(len(test_exp_data.gene_names)))
                )
            else:
                test_data = _TestData(
                    test_exp_data.exp_matrix,
                    np.array(range(len(test_exp_data.cell_names))),
                    np.array(range(len(test_exp_data.gene_names)))
                )

            logger.info('start scoring test_data...')
            start_time = time.time()
            output, labels_array = self._score_test_data(test_data, common_gene)
            logger.info(f'scoring test_data finished, cost {time.time() - start_time} seconds')

            logger.info('start fine-tuning...')
            start_time = time.time()
            ret_labels = self._fine_tune(test_data, output, trained_data)
            logger.info(f'fine-tuning finished, cost {time.time() - start_time} seconds')
            bins = test_data.cell_names if cluster_res_key else test_exp_data.cell_names

        res = pd.DataFrame(columns=['bins', 'group', 'first_labels'])
        res['bins'] = bins
        res['group'] = ret_labels
        res['first_labels'] = labels_array
        res.index = res['bins'].values
        logger.info(f'single-r finished, cost {time.time() - total_start_time} seconds')
        if cluster_res_key:
            res_single_r = pd.DataFrame(res.loc[test_cluster_result['group']])
            bins = test_cluster_result['bins'] if 'bins' in test_cluster_result else test_cluster_result.index
            res_single_r['bins'].astype(bins.dtype)
            res_single_r['bins'] = bins.values
            res_single_r.index = res_single_r['bins']
            self.pipeline_res[res_key] = res_single_r
            return res_single_r
        else:
            self.pipeline_res[res_key] = res
            return res

    @staticmethod
    def check_gpu_env():
        try:
            import cudf  # noqa
            import cugraph  # noqa
        except ImportError:
            raise ImportError(
                "Your env don't have GPU related RAPIDS packages, if you want to run this option, follow the "
                "guide at https://stereopy.readthedocs.io/en/latest/Tutorials/clustering_by_gpu.html")

    @staticmethod
    def load_cell_types(ref_exp_data, ref_use_col):
        if ref_use_col not in ref_exp_data._ann_data.obs:
            raise IndexError("No celltype found.")

        group = ref_exp_data._ann_data.obs[ref_use_col]
        codes = group.cat.categories.to_list()
        celltypes = [int(celltype) for celltype in group.cat.codes]
        return codes, celltypes

    def _train_ref(self):
        dict_of_median_exp = dict()
        for label, y in self.group_data_frame.groupby('group'):
            cells_bool_list = np.isin(self.ref_exp_data.cell_names, y['bins'].values)
            dict_of_median_exp[label] = pd.DataFrame(
                np.median(self.ref_exp_data.exp_matrix[cells_bool_list].toarray(), axis=0))
        median_exp = pd.concat(dict_of_median_exp, axis=1)
        median_exp.index = self.ref_gene_names

        ret_all = defaultdict(dict)
        ret_gene = defaultdict(dict)

        ct = self.group_data_frame['group'].astype('category').cat.categories
        de_n = np.round(500 * (2 / 3) ** np.log2(len(ct)))
        for i in ct:
            for j in ct:
                if i == j:
                    continue

                ret_all[i][j] = median_exp[i] - median_exp[j]
                ret_tmp = pd.DataFrame(ret_all[i][j]).round(6)
                ret_tmp['arg'] = [x for x in range(ret_tmp.shape[0])][::-1]
                ret_all[i][j] = ret_tmp.sort_values([0, 'arg'], ascending=False)
                ret_gene[i][j] = ret_all[i][j].index[:int(de_n)]

        return ret_all, np.unique([y for x in ret_gene.values() for y in x.values()])

    def _score_test_data(self, test_data, common_gene):
        original_exp = {}
        tmp_gene_names = self.ref_gene_names.tolist()
        test_data_length = test_data.exp_matrix.shape[1]
        common_gene_index = []
        for com in common_gene:
            index = tmp_gene_names.index(com)
            if index < test_data_length:
                common_gene_index.append(index)

        for x, y in self.group_data_frame.groupby('group'):
            y_cell_bool_list = np.isin(self.ref_exp_data.cell_names, y['bins'].values)
            original_exp[x] = self.ref_exp_data.exp_matrix[y_cell_bool_list][:, common_gene_index].toarray()
        test_mat = test_data.exp_matrix[:, common_gene_index].toarray()
        ranked_mat_ref = apply_along_axis(test_mat.T)

        res_dict = dict(Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_spearman_parallel)(label, ranked_mat_ref, y.T, self.quantile)
            for label, y in tqdm(original_exp.items())
        ))

        ret = pd.DataFrame(res_dict, index=test_data.cell_names)
        return ret, ret.columns[np.argmax(ret.values, axis=1)]

    def _fine_tune(self, test_data, output, trained_data):
        tmp = output.values.copy()
        tmp[tmp < np.array(np.max(output.values, axis=1) - self.fine_tune_threshold).reshape([-1, 1])] = 0
        tmp[tmp > 0] = 1

        ref = pd.DataFrame(
            self.ref_exp_data.exp_matrix.toarray(),
            index=self.ref_exp_data.cell_names,
            columns=self.ref_exp_data.gene_names
        )

        logger.info(f'fine-tuning with test_data(shape={test_data.exp_matrix.shape})')

        ret_labels = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self._fine_tune_parallel)(
                ref,
                output.columns[tmp[x].astype(bool)].values,
                y[1].to_frame().T,
                trained_data
            )
            for x, y in tqdm(enumerate(test_data.to_df().iterrows()))
        )
        return ret_labels

    def _fine_tune_parallel(self, ref, labels, y, trained_data):
        if self.fine_tune_times:
            try_num = 0
            while try_num < self.fine_tune_times and len(labels) > 1:
                labels = self._fine_tune_one_time(labels, ref, y, trained_data, try_num)
                try_num += 1
        else:
            while len(labels) > 1:
                labels = self._fine_tune_one_time(labels, ref, y, trained_data)
        return labels[0]

    @numba.jit(forceobj=True, nogil=True, parallel=True)
    def _fine_tune_one_time(self, top_labels, ref, test, trained_data, try_num=None):
        genes_filtered = _get_label_genes_set(top_labels, trained_data)
        genes_filtered = np.unique([y for x in genes_filtered for y in x])
        if len(genes_filtered) < 20:
            return [top_labels[0]]

        test_data_length = test.shape[1]
        ref_data_length = ref.shape[1]
        test_genes_filtered_index = []
        ref_genes_filtered_index = []
        for gen in genes_filtered:
            index = self.test_gene_names.get(gen, None)
            if index is None:
                continue
            if test_data_length > index:
                test_genes_filtered_index.append(index)
            if ref_data_length > index:
                ref_genes_filtered_index.append(index)
        test_filtered = test.loc[:, test_genes_filtered_index].values
        if np.std(test_filtered) <= 0:
            return [top_labels[0]]

        bool_list = np.isin(self.group_data_frame['group'].values, top_labels)
        ref_filtered = ref.loc[bool_list, ref_genes_filtered_index]
        group_data_frame_filtered = self.group_data_frame.loc[bool_list].reset_index()

        ranked_mat_ref = apply_along_axis(test_filtered.T)
        res_labels = {}
        for p, q in group_data_frame_filtered.groupby('group'):
            ref_filtered_by_group = ref_filtered.iloc[q.index].values.T
            ranked_mat_qry = apply_along_axis(ref_filtered_by_group)
            sim = corr_spearman(ranked_mat_ref, ranked_mat_qry)
            if not (sim.shape and sim.shape[0] and sim.shape[1]):
                continue
            res_labels[p] = np.percentile(sim, self.quantile, axis=1)[0]

        if not res_labels:
            return [top_labels[0]]

        # remove the labels lower than median
        mid_value = (min(res_labels.values()) + np.median(list(res_labels.values()))) / 2
        res_labels = {p: value for p, value in res_labels.items() if value > mid_value}
        if not res_labels:
            return [top_labels[0]]

        # remove the label lower than fine_tune_threshold
        max_value = max(res_labels.values()) - self.fine_tune_threshold
        res_labels = {p: value for p, value in res_labels.items() if value >= max_value}
        if not res_labels:
            return [top_labels[0]]

        if try_num and try_num == self.fine_tune_times - 1:
            return [sorted(res_labels.items(), key=itemgetter(1))[-1][0]]

        return list(res_labels.keys())

    @staticmethod
    def test_rank():
        mat = np.random.sample((5, 5)).astype(np.float32)
        res_self = apply_along_axis(mat)

        from scipy.stats import rankdata
        res_sci = rankdata(mat, axis=0).astype(np.float32)

        assert np.all(res_self == res_sci), 'self define rank is different with the SciPy\'s versions'
        logger.info('test_rank success')

    def _group_data_frame_checker(self):
        assert 'group' in self.group_data_frame.columns
        if 'bins' not in self.group_data_frame.columns:
            self.group_data_frame = self.group_data_frame.reset_index().rename(
                columns={'index': 'bins', 'CellID': 'bins'}
            )
        assert 'bins' in self.group_data_frame.columns

    @staticmethod
    def test_group_data_frame():
        s1 = SingleR(None)
        # cluster result data like
        s1.group_data_frame = pd.DataFrame(index=['cell1', 'cell2'], columns=['bins', 'group'])
        s1._group_data_frame_checker()

        s2 = SingleR(None)
        # h5ad with cell_type data like
        s2.group_data_frame = pd.DataFrame(index=['cell1', 'cell2'], columns=['CellID', 'group'])
        s2._group_data_frame_checker()

        s3 = SingleR(None)
        # h5ad with cell_type data like
        s3.group_data_frame = pd.DataFrame(index=['cell1', 'cell2'], columns=['group'])
        s3._group_data_frame_checker()
        logger.info('test_group_data_frame success')
