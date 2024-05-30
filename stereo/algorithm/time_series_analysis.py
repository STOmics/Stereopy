import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.constant import AlternativeType
from stereo.constant import FEATURE_P
from stereo.constant import FUZZY_C_RESULT
from stereo.constant import FUZZY_C_WEIGHT
from stereo.constant import GREATER_P
from stereo.constant import GREATER_PVALUE
from stereo.constant import LESS_P
from stereo.constant import LESS_PVALUE
from stereo.constant import LOG_FC
from stereo.constant import PValCombinationType
from stereo.constant import RunMethodType
from stereo.constant import SCORES
from stereo.constant import UseColType


class TimeSeriesAnalysis(AlgorithmBase):

    def main(
            self,
            run_method=RunMethodType.tvg_marker.value,
            use_col=UseColType.timepoint.value,
            branch=None,
            p_val_combination=PValCombinationType.fdr.value,
            cluster_number=6,
            **kwargs
    ):
        """
        :param run_method: the model type when the algorithm is run, default = `tvg_marker`.
                        `tvg_marker`: Calculate time variable gene based on expression of celltypes in branch
                        `other`: Use fuzzy C means cluster method to cluster genes based on 1-p_value of celltypes in branch
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param p_val_combination: p_value combination method to use, choosing from ['fisher', 'mean', 'FDR'], default to 'FDR',
                        only for `tvg_marker` run_method
        :param cluster_number: number of cluster, defaults to 6, only for `other` run_method

        other parameters:

        .. note::

            All the parameters below are key word arguments and only for `tvg_marker` run_method.
        
        :param statistical_test_method: one tail statistical test method to be used for up or down regulated examination between time point,
                        choosing from ['t-test', 'permutation', 'wilcoxon'], default to 't-test']
        :param permutation_batch: permutation_batch to control memory consumption if statistical_test_method == 'permutation',
                        the higher permutation_batch require higher memory consumption. default to 100. 
        :param permutation_n_resamples: the times of resamples if statistical_test_method == 'permutation', default to 999.

        .. note::
            
            All the parameters below are key word arguments and only for `other` run_method.

        :param spatial_weight: the weight to combine spatial feature, defaults to 1.
        :param n_spatial_feature: n top features to combine of spatial feature, defaults to 2.
        :param temporal_mean_threshold: filter out genes of which mean absolute temporal feature <= temporal_mean_threshold, defaults to 0.85.
        :param temporal_top_threshold: filter out genes of which top absolute temporal feature < temporal_top_threshold, defaults to 1.
        :param cluster_method: method to cluster gene based on spatial and temporal feature,
                        choose from ['fuzzy_C_means', 'gaussian_mixture'], default to fuzzy_C_means.
        :param Epsilon: max value to finish iteration if cluster_method=='fuzzy_C_means', defaults to 1e-7.
        :param w_size: window size to rasterizing spatial expression, default to 20.
        :param use_col: the col in obs representing celltype or clustering, default to None.
        :param branch: celltypes order in use_col, default to None.
        :param seed: fix seed in numpy to keep output constant in every run.

        """  # noqa
        if run_method == RunMethodType.tvg_marker.value:
            self.TVG_marker(
                use_col=use_col,
                branch=branch,
                p_val_combination=p_val_combination,
                **kwargs
            )
        else:
            self.fuzzy_C_gene_pattern_cluster(
                cluster_number=cluster_number,
                branch=branch,
                use_col=use_col,
                **kwargs
            )
            
    def _statistic_mean_diff(self, y, x, axis=0):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)

    def TVG_marker(self, use_col, branch, p_val_combination=PValCombinationType.fisher.value, statistical_test_method = 't-test', permutation_batch=100, permutation_n_resamples=999):
        """
        Calculate time variable gene based on expression of celltypes in branch
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param p_val_combination: p_value combination method to use, choosing from ['fisher', 'mean', 'FDR']
        :param statistical_test_method: one tail statistical test method to be used for up or down regulated examination between time point. choosing from ['t-test', 'permutation', 'wilcoxon'], default as 't-test'
        :param permutation_batch: permutation_batch to control memory consumption if statistical_test_method = 'permutation', the higher permutation_batch require higher memory consumption. default as 100. 
        :param permutation_n_resamples: the times of resamples if statistical_test_method = 'permutation', default as 999.
        :return: stereo_exp_data contains Time Variabel Gene marker result
        """
        from scipy.stats import ttest_ind
        from scipy.stats import permutation_test
        from scipy.stats import ranksums
        from scipy import sparse
        from scipy import stats
        label2exp = {}
        for x in branch:
            cell_list = self.stereo_exp_data.cells.to_df().loc[self.stereo_exp_data.cells[use_col] == x,].index
            test_exp_data = self.stereo_exp_data.sub_by_name(cell_name=cell_list.to_list())
            if sparse.issparse(test_exp_data.exp_matrix):
                label2exp[x] = test_exp_data.exp_matrix.todense()
            else:
                label2exp[x] = np.mat(test_exp_data.exp_matrix)

        logFC = []
        less_pvalue = []
        greater_pvalue = []
        scores = []
        if statistical_test_method == 't-test':
            for i in range(len(branch)-1):
                score, pvalue = ttest_ind(label2exp[branch[i+1]], label2exp[branch[i]], axis=0, alternative='less')
                #np.nan_to_num(score, nan=0, copy = False)
                less_pvalue.append(np.nan_to_num(pvalue, nan=1, copy = False))
                score, pvalue = ttest_ind(label2exp[branch[i+1]], label2exp[branch[i]], axis=0, alternative='greater')
                greater_pvalue.append(np.nan_to_num(pvalue, nan=1, copy = False))
                logFC.append(np.array(np.log2((np.mean(label2exp[branch[i+1]], axis=0)+1e-9)/(np.mean(label2exp[branch[i]], axis=0)+1e-9)))[0]) 
                scores.append(score)
        elif statistical_test_method == 'permutation':
            #print(statistical_test_method)
            for i in range(len(branch)-1):
                tmp_permutation = permutation_test((label2exp[branch[i+1]], label2exp[branch[i]]), self._statistic_mean_diff, 
                                                    n_resamples=permutation_n_resamples, batch=permutation_batch, 
                                                    vectorized=True, alternative='less',random_state=42, axis=0)
                score = tmp_permutation.statistic
                #pvalue = tmp_permutation.pvalue
                null_distribution = tmp_permutation.null_distribution
                pvalue = np.sum(score >= null_distribution, axis=0)/permutation_n_resamples
                #np.nan_to_num(score, nan=0, copy = False)
                less_pvalue.append(np.nan_to_num(pvalue, nan=1, copy = False))
                pvalue = np.sum(score <= null_distribution, axis=0)/permutation_n_resamples
                greater_pvalue.append(np.nan_to_num(pvalue, nan=1, copy = False))
                logFC.append(np.array(np.log2((np.mean(label2exp[branch[i+1]], axis=0)+1e-9)/(np.mean(label2exp[branch[i]], axis=0)+1e-9)))[0]) 
                scores.append(score)
        elif statistical_test_method == 'wilcoxon':
            for i in range(len(branch)-1):
                score, pvalue = ranksums(label2exp[branch[i+1]], label2exp[branch[i]], axis=0, alternative='less')
                #np.nan_to_num(score, nan=0, copy = False)
                less_pvalue.append(np.nan_to_num(pvalue, nan=1, copy = False))
                score, pvalue = ranksums(label2exp[branch[i+1]], label2exp[branch[i]], axis=0, alternative='greater')
                greater_pvalue.append(np.nan_to_num(pvalue, nan=1, copy = False))
                logFC.append(np.array(np.log2((np.mean(label2exp[branch[i+1]], axis=0)+1e-9)/(np.mean(label2exp[branch[i]], axis=0)+1e-9)))[0]) 
                scores.append(score)
        else:
            raise ValueError("statistical_test_method should be selected from ['t-test', 'permutation', 'wilcoxon']")
        self.stereo_exp_data.genes_matrix[SCORES] = np.array(scores).T
        self.stereo_exp_data.genes_matrix[SCORES] = np.nan_to_num(self.stereo_exp_data.genes_matrix[SCORES])
        self.stereo_exp_data.genes_matrix[GREATER_P] = np.array(greater_pvalue).T
        self.stereo_exp_data.genes_matrix[LESS_P] = np.array(less_pvalue).T
        logFC = np.array(logFC).T

        self.stereo_exp_data.genes_matrix[LOG_FC] = logFC
        logFC = np.mean(logFC, axis=1)

        if p_val_combination == PValCombinationType.mean.value:
            less_pvalue = np.mean(np.array(less_pvalue), axis=0)
            greater_pvalue = np.mean(np.array(greater_pvalue), axis=0)
        elif p_val_combination == PValCombinationType.fisher.value:
            tmp = self.stereo_exp_data.genes_matrix[LESS_P].copy()
            tmp[tmp == 0] = np.min(tmp[tmp != 0])
            tmp = np.sum(-2 * np.log(tmp), axis=1)
            less_pvalue = 1 - stats.chi2.cdf(tmp, self.stereo_exp_data.genes_matrix[LESS_P].shape[1])
            tmp = self.stereo_exp_data.genes_matrix[GREATER_P].copy()
            tmp[tmp == 0] = np.min(tmp[tmp != 0])
            tmp = np.sum(-2 * np.log(tmp), axis=1)
            greater_pvalue = 1 - stats.chi2.cdf(tmp, self.stereo_exp_data.genes_matrix[GREATER_P].shape[1])
        elif p_val_combination == PValCombinationType.fdr.value:
            less_pvalue = 1 - np.prod(1 - self.stereo_exp_data.genes_matrix[LESS_P], axis=1)
            greater_pvalue = 1 - np.prod(1 - self.stereo_exp_data.genes_matrix[GREATER_P], axis=1)
        self.stereo_exp_data.genes[LESS_PVALUE] = less_pvalue
        self.stereo_exp_data.genes[GREATER_PVALUE] = greater_pvalue
        self.stereo_exp_data.genes[LOG_FC] = logFC

    def fuzzy_C(self, data, cluster_number, MAX=10000, m=2, Epsilon=1e-7, seed=20240523):
        """
        fuzzy C means algorithm to cluster, helper function used in fuzzy_C_gene_pattern_cluster
        :param data: pd.DataFrame object for fuzzy C means cluster, each col represent a feature, each row represent a obsversion # noqa
        :param seed: fix seed in numpy to keep output constant in every run.
        :param cluster_number: number of cluster
        :param MAX: max value to random initialize
        :param m: degree of membership, default = 2
        :param Epsilon: max value to finish iteration
        :return: fuzzy C means cluster result
        """
        np.random.seed(seed)
        assert m > 1
        import time
        import copy
        from scipy import spatial
        U = np.random.randint(1, int(MAX), (len(data), cluster_number))
        U = U / np.sum(U, axis=1, keepdims=True)
        epoch = 0
        tik = time.time()
        while True:
            epoch += 1
            U_old = copy.deepcopy(U)
            U1 = U ** m
            U2 = np.expand_dims(U1, axis=2)
            U2 = np.repeat(U2, data.shape[1], axis=2)
            data1 = np.expand_dims(data, axis=1)
            data1 = np.repeat(data1, cluster_number, axis=1)
            dummy_sum_num = np.sum(U2 * data1, axis=0)
            dummy_sum_dum = np.sum(U1, axis=0)
            C = (dummy_sum_num.T / dummy_sum_dum).T

            # initializing distance matrix
            distance_matrix = spatial.distance_matrix(data, C)

            # update U
            distance_matrix_1 = np.expand_dims(distance_matrix, axis=1)
            distance_matrix_1 = np.repeat(distance_matrix_1, cluster_number, axis=1)
            distance_matrix_2 = np.expand_dims(distance_matrix, axis=2)
            distance_matrix_2 = np.repeat(distance_matrix_2, cluster_number, axis=2)

            U = np.sum((distance_matrix_1 / distance_matrix_2) ** (2 / (m - 1)), axis=1)
            U = 1 / U
            if epoch % 100 == 0:
                print('epoch {} : time cosumed{:.4f}s, loss:{}'.format(epoch, time.time() - tik,
                                                                       np.max(np.abs(U - U_old))))
                tik = time.time()
            if np.max(np.abs(U - U_old)) < Epsilon:
                break
        return U

    def gene_spatial_feature(self, w_size=20):
        """
        Use pca on rasterizing spatial expression to represent gene spatial feature
        :param w_size: window size to rasterizing spatial expression,
        :return: stereoexpdata.genes_matrix['spatial_info'] # a gene matrix to represent spatial feature
        """
        from scipy import sparse
        from stereo.algorithm import dim_reduce
        from stereo.algorithm import normalization
        from stereo.algorithm import scale

        loc = self.stereo_exp_data.position.copy()
        loc = (loc / w_size).astype('int').astype('str')
        loc = np.array(['_'.join(x) for x in loc])
        Exp_matrix = self.stereo_exp_data.exp_matrix
        if not sparse.issparse(Exp_matrix):
            Exp_matrix = sparse.csr_array(Exp_matrix)
        ci, gi = Exp_matrix.nonzero()
        values = Exp_matrix.data
        cl = loc[ci]
        cl = pd.Series(cl).astype('category')
        X = sparse.csr_matrix((values, (gi, cl.cat.codes.to_numpy())), shape=(np.max(gi) + 1, len(cl.cat.categories)))
        X = normalization.normalize_total(X, target_sum=1e4)
        X = np.log1p(X)
        X = scale.scale(X, zero_center=True, max_value=None)
        X_pca = dim_reduce.pca(X, min(50 , X.shape[1] -1))['x_pca']

        self.stereo_exp_data.tl.result['spatial_feature'] = X_pca
        self.stereo_exp_data.genes_matrix['spatial_feature'] = X_pca

    def fuzzy_C_gene_pattern_cluster(self, cluster_number, spatial_weight=1, n_spatial_feature=2, cluster_method = 'fuzzy_C_means',
                                     temporal_mean_threshold=0.85, temporal_top_threshold=1, Epsilon=1e-7, w_size=None,
                                     use_col=None, branch=None, seed=20240523):
        """
        Use fuzzy C means cluster method to cluster genes based on 1-p_value of celltypes in branch
        :param cluster_number: number of cluster
        :param spatial_weight: the weight to combine spatial feature
        :param n_spatial_feature: n top features to combine of spatial feature
        :param cluster_method: method to cluster gene based on spatial and temporal feature. choose from ['fuzzy_C_means', 'gaussian_mixture'], default as fuzzy_C_means.
        :param temporal_mean_threshold: filter out genes of which mean absolute temporal feature <= temporal_mean_threshold # noqa
        :param temporal_top_threshold: filter out genes of which top absolute temporal feature < temporal_top_threshold
        :param Epsilon: max value to finish iteration
        :param w_size: window size to rasterizing spatial expression, see also data.tl.gene_spatial_feature
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param seed: fix seed in numpy to keep output constant in every run.
        :return: stereo_exp_data contains fuzzy_C_result
        """
        if (GREATER_P not in self.stereo_exp_data.genes_matrix) or (LESS_P not in self.stereo_exp_data.genes_matrix):
            if use_col == None:  # noqa
                print("greater_p and less_p not in stereo_exp_data.genes_matrix, you should run with run_method='tvg_marker' first")
            else:
                self.TVG_marker(use_col=use_col, branch=branch)
        sig = ((1 - self.stereo_exp_data.genes_matrix[GREATER_P]) >= (
                1 - self.stereo_exp_data.genes_matrix[LESS_P])).astype(
            int)
        sig[sig == 0] = -1
        self.stereo_exp_data.genes_matrix[FEATURE_P] = np.max(
            [(1 - self.stereo_exp_data.genes_matrix[GREATER_P]), (1 - self.stereo_exp_data.genes_matrix[LESS_P])],
            axis=0) * sig

        # filtter useless gene
        useful_index_1 = (
                np.mean(np.abs(self.stereo_exp_data.genes_matrix[FEATURE_P]), axis=1) > temporal_mean_threshold)
        useful_index_2 = (
                np.max(np.abs(self.stereo_exp_data.genes_matrix[FEATURE_P]), axis=1) >= temporal_top_threshold)
        useful_index = useful_index_1 & useful_index_2
        useless_index = ~useful_index

        if 'spatial_feature' not in self.stereo_exp_data.genes_matrix:
            if w_size is None:
                self.gene_spatial_feature()
            else:
                self.gene_spatial_feature(w_size)

        temporal_feature = self.stereo_exp_data.genes_matrix[FEATURE_P]
        spatial_feature = spatial_weight * self.stereo_exp_data.genes_matrix['spatial_feature'][:, :n_spatial_feature]
        merge_feature = np.concatenate([temporal_feature, spatial_feature], axis=1)
        
        if cluster_method == 'fuzzy_C_means':
            fuzzy_C_weight = self.fuzzy_C(merge_feature[useful_index], cluster_number, Epsilon=Epsilon, seed=seed)
        elif cluster_method == 'gaussian_mixture':
            gm = GaussianMixture(n_components=cluster_number, random_state=seed).fit(merge_feature[useful_index])
            gm_weight = gm.predict_proba(merge_feature[useful_index])
            fuzzy_C_weight = gm_weight
        else:
            raise ValueError("cluster_method should be selected from ['fuzzy_C_means', 'gaussian_mixture']")
            
        df = {x: fuzzy_C_weight[i] for i, x in enumerate(self.stereo_exp_data.genes.to_df().index[useful_index])}
        for x in self.stereo_exp_data.genes.to_df().index[useless_index]:
            df[x] = np.zeros(cluster_number)

        df = pd.DataFrame(df).T
        df = df.loc[self.stereo_exp_data.genes.to_df().index]

        self.stereo_exp_data.genes_matrix[FUZZY_C_WEIGHT] = df.to_numpy()

        fuzzy_c_result = np.argmax(self.stereo_exp_data.genes_matrix[FUZZY_C_WEIGHT], axis=1) + 1
        fuzzy_c_result[useless_index] = 0
        self.stereo_exp_data.genes[FUZZY_C_RESULT] = fuzzy_c_result
