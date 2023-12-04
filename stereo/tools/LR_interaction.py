from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from ..core.stereo_exp_data import StereoExpData
from ..core.tool_base import ToolBase


class LrInteraction(ToolBase):
    """calculate cci score for each LR pair and do permutation test

    Parameters
    ----------
    data : `StereoExpData`
        StereoExpData
    distance : Union[int, float], optional
        the distance between spots which are considered as neighbors , by default 5
    bin_scale : int, optional
        to scale the distance `distance = bin_scale * distance`, by default 1
    n_jobs : Optional[int], optional
        num of workers for parallel jobs, by default None
    min_exp : Union[int, float], optional
        the min expression of ligand or receptor gene when caculate reaction strength, by default 0
    min_spots : `int`, optional
        the min number of spots that score > 0, by default 20
    n_pairs : `int`, optional
        number of , by default 1000
    quantiles : `tuple`, optional
        _description_, by default (0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.9975, 0.999, 1)
    """

    def __init__(
            self,
            data: StereoExpData,
            verbose: bool = True,
            distance: Union[int, float] = 5,
            bin_scale: int = 1,
            n_jobs: Optional[int] = None,
            spot_comp=None,  # TODO
            min_exp: Union[int, float] = 0,
            min_spots: int = 20,
            n_pairs: int = 1000,
            quantiles=(0.5, 0.75, 0.85, 0.9,
                       0.95, 0.97, 0.98, 0.99,
                       0.995, 0.9975, 0.999, 1)
    ):

        super(LrInteraction, self).__init__(data=data, method='KNN')
        self.verbose = verbose
        self.distance = distance
        self.bin_scale = bin_scale
        self.n_jobs = n_jobs
        self.spot_comp = spot_comp
        self.min_exp = min_exp
        self.min_spots = min_spots
        self.result = dict()
        self.quantiles = quantiles
        self.n_pairs = n_pairs
        self.lr_paris = None

    @staticmethod
    def is_gene_name(data, gene):
        if type(gene) in [str, np.str_]:
            return gene in data.gene_names
        else:
            return False

    def calculate_score(self, lr_pairs, neighbors_key, key_add, verbose):

        if isinstance(lr_pairs, str):
            lr_pairs = [lr_pairs]

        if verbose:
            print('filtered out the lr pairs which '
                  'is not unique or has gene not in the `data.gene_names`')

        lr_pairs = np.unique(lr_pairs)
        lr_pairs = [item for item in lr_pairs if np.all([self.is_gene_name(self.data, gene)
                                                         for gene in item.split("_")])]
        lr_genes = '_'.join(lr_pairs).split('_')
        genes = np.unique('_'.join(lr_pairs).split('_'))
        lr_pairs_rev = [item.split("_")[1] + "_" + item.split("_")[0] for item in lr_pairs]
        lr_genes_rev = '_'.join(lr_pairs_rev).split('_')

        df = self.data[:, genes].copy().to_df()
        df[neighbors_key] = self.result[neighbors_key]

        if verbose:
            print('calculating cci score...')

        spot_lr1 = df[lr_genes]
        spot_lr2 = df[lr_genes_rev]

        def mean_lr2(x):
            # get lr2 expressions from the neighbour(s)
            nbs = spot_lr2.loc[df.loc[x.name, neighbors_key], :]

            if nbs.shape[0] > 0:  # if neighbour exists
                return nbs.sum() / nbs.shape[0]
            else:
                return 0

        # mean of lr2 expressions from neighbours of each spot
        if self.n_jobs is not None:
            try:
                from pandarallel import pandarallel
            except ImportError:
                raise ImportError(
                    'Please install the pandarallel, `pip3 install pandarallel')

            pandarallel.initialize(progress_bar=False, nb_workers=self.n_jobs, verbose=1)
            nb_lr2 = spot_lr2.parallel_apply(mean_lr2, axis=1)

        else:
            nb_lr2 = spot_lr2.apply(mean_lr2, axis=1)

        # keep value of nb_lr2 only when lr1 is also expressed on the spots
        spot_lr = (spot_lr1.values * (nb_lr2.values > self.min_exp) +
                   (spot_lr1.values > self.min_exp) * nb_lr2.values)

        columns = [lr_pairs[i // 2] for i in range(len(spot_lr2.columns))]

        # make multiindex so we can group it together
        columns = pd.MultiIndex.from_arrays((columns, spot_lr2.columns))
        scores = pd.DataFrame(spot_lr, index=self.data.cell_names, columns=columns)
        scores = scores.T.reset_index().groupby(['level_0']).sum().T
        scores.index.name = 'LR pairs'

        if self.min_spots is not None and isinstance(self.min_spots, int):
            lrs_bool = (scores > 0).sum(axis=0) > self.min_spots
            exp_pairs = lrs_bool.sum()
            if exp_pairs > 0:
                scores = scores.loc[:, lrs_bool]
                if verbose:
                    print(f'filtered out {len(lr_pairs) - exp_pairs} lr pairs')
            else:
                raise ValueError(f"lrs expressed on less than {self.min_spots}， try to decrease min_exp")

        if key_add is not None:
            self.result[key_add] = scores

        else:
            return scores

    @staticmethod
    def get_similar_genes(ref_quants: np.array,
                          n_genes: int,
                          candidate_quants: np.ndarray,
                          candidate_genes: np.array
                          ):
        ref_quants = ref_quants.reshape(-1, 1)
        dists = (np.fabs(ref_quants - candidate_quants) / (ref_quants + candidate_quants)).sum(axis=0)

        # remove the zero-dists since this
        # indicates they are the same gene
        dists = dists[dists > 0]
        candidate_genes = candidate_genes[dists > 0]
        order = np.argsort(dists)

        # Retrieving desired number of genes
        similar_genes = candidate_genes[order[0: n_genes]]
        return similar_genes

    @staticmethod
    def gene_rand_pairs(genes1: np.array, genes2: np.array, n_pairs: int):
        """Generates random pairs of genes."""

        rand_pairs = list()
        for _ in range(0, n_pairs):
            l_rand = np.random.choice(genes1, 1)[0]
            r_rand = np.random.choice(genes2, 1)[0]
            rand_pair = "_".join([l_rand, r_rand])
            while rand_pair in rand_pairs or l_rand == r_rand:
                l_rand = np.random.choice(genes1, 1)[0]
                r_rand = np.random.choice(genes2, 1)[0]
                rand_pair = "_".join([l_rand, r_rand])

            rand_pairs.append(rand_pair)

        return rand_pairs

    def get_lr_bg(
            self,
            neighbors_key,
            lr_,
            l_quant,
            r_quant,
            genes,
            candidate_quants,
            gene_bg_genes,
            n_genes,
            n_pairs,
            spot_comp=None,  # TODO
    ):
        """Gets the LR-specific background & bg spot indices."""
        l_, r_ = lr_.split("_")
        if l_ not in gene_bg_genes:
            l_genes = self.get_similar_genes(l_quant, n_genes, candidate_quants, genes)
            gene_bg_genes[l_] = l_genes
        else:
            l_genes = gene_bg_genes[l_]

        if r_ not in gene_bg_genes:
            r_genes = self.get_similar_genes(r_quant, n_genes, candidate_quants, genes)
            gene_bg_genes[r_] = r_genes
        else:
            r_genes = gene_bg_genes[r_]

        rand_pairs = self.gene_rand_pairs(l_genes, r_genes, n_pairs)
        background = self.calculate_score(lr_pairs=rand_pairs, neighbors_key=neighbors_key, key_add=None, verbose=False)
        return background

    def get_lr_features(self, lr_expr):  # modified

        quantiles = np.array(self.quantiles)
        # Determining indices of LR pairs #
        l_indices, r_indices = [], []
        for lr in self.lr_pairs:
            l_, r_ = lr.split("_")
            l_indices.extend(np.where(lr_expr.columns.values == l_)[0])
            r_indices.extend(np.where(lr_expr.columns.values == r_)[0])

        interpolation = "nearest"

        # Calculating the non-zero quantiles of gene expression
        def nonzero_quantile(expr):
            """Calculating the non-zero quantiles."""
            nonzero_expr = expr[expr > 0]  # all=0
            quants = np.quantile(nonzero_expr, q=quantiles, interpolation=interpolation)
            return quants

        def nonzero_median(expr):
            """Calculating the non-zero median."""
            nonzero_expr = expr[expr > 0]
            median = np.median(nonzero_expr)
            return median

        def zero_props(expr):
            """Calculating the non-zero pro."""
            gene_props = (expr == 0).sum() / len(expr)
            return gene_props

        summary = lr_expr.T.agg([nonzero_quantile, nonzero_median, zero_props, np.median], axis=1)
        l_summary = summary.iloc[l_indices, :]
        r_summary = summary.iloc[r_indices, :]

        lr_median_means = (l_summary.nonzero_median.values + r_summary.nonzero_median.values) / 2
        lr_prop_means = (l_summary.zero_props.values + r_summary.zero_props.values) / 2

        median_order = np.argsort(lr_median_means)
        prop_order = np.argsort(lr_prop_means * -1)
        median_ranks = [np.where(median_order == i)[0][0] for i in range(len(self.lr_pairs))]
        prop_ranks = [np.where(prop_order == i)[0][0] for i in range(len(self.lr_pairs))]
        mean_ranks = np.array([median_ranks, prop_ranks]).mean(axis=0)

        columns = ["nonzero-median", "zero-prop", "median_rank", "prop_rank", "mean_rank"]
        lr_features = pd.DataFrame(columns=columns, index=self.lr_pairs)
        lr_features[columns[0: 2]] = summary[['nonzero_median', 'zero_props']].values.mean(axis=0)
        lr_features[columns[2]] = median_ranks
        lr_features[columns[3]] = prop_ranks
        lr_features[columns[4]] = mean_ranks

        q_cols = [f'L_q{quantile}' for quantile in quantiles] + [f'R_q{quantile}' for quantile in quantiles]

        q_values = np.hstack([l_summary.nonzero_quantile.values.tolist(), r_summary.nonzero_quantile.values.tolist()])

        lr_features[q_cols] = q_values
        self.result['lr_features'] = lr_features
        return lr_features

    def _permutation(
            self,
            lr_scores: np.ndarray,
            lr_pairs: np.array,
            neighbors_key: list,
            het_vals: np.array = None,  # TODO
            adj_method: str = "fdr_bh",
            pval_adj_cutoff: float = 0.05,
            save_bg=False,
    ):

        """Calls significant spots by creating random gene pairs with similar
        expression to given LR pair; only generate background for spots
        which have score for given LR.
        """
        quantiles = np.array(self.quantiles)
        lr_pairs = self.lr_pairs
        n_pairs = self.n_pairs

        lr_genes = np.unique([lr_.split("_") for lr_ in lr_pairs])
        genes = np.array([gene for gene in self.data.gene_names if gene not in lr_genes])
        candidate_expr = self.data[:, genes].to_df().values

        n_genes = round(np.sqrt(n_pairs) * 2)
        if len(genes) < n_genes:
            print(
                f"Exiting since need at least {n_genes} genes to generate {n_pairs} pairs."
            )
            return

        if n_pairs < 100:
            print(
                "Exiting since `n_pairs < 100`, need much larger number of pairs to "
                "get accurate backgrounds (e.g. 1000)."
            )
            return
        lr_expr = self.data[:, lr_genes].copy().to_df()
        lr_feats = self.get_lr_features(lr_expr)
        l_quants = lr_feats.loc[lr_pairs, [col for col in lr_feats.columns if "L_" in col]].values
        r_quants = lr_feats.loc[lr_pairs, [col for col in lr_feats.columns if "R_" in col]].values

        candidate_quants = np.apply_along_axis(np.quantile, 0, candidate_expr, q=quantiles, interpolation="nearest")

        pvals = np.ones(lr_scores.shape, dtype=np.float32)
        # do permutation
        from tqdm import tqdm
        gene_bg_genes = dict()
        pbar = tqdm(lr_pairs)
        if self.verbose:
            print("Performing permutation...")

        for idx, lr_ in enumerate(pbar):
            pbar.set_description(f"LR pairs {idx + 1}")
            pbar.set_postfix(LR=lr_pairs[idx])
            background = self.get_lr_bg(
                neighbors_key=neighbors_key,
                lr_=lr_,
                l_quant=l_quants[idx, :],
                r_quant=r_quants[idx, :],
                genes=genes,
                candidate_quants=candidate_quants,
                gene_bg_genes=gene_bg_genes,
                n_genes=n_genes,
                n_pairs=n_pairs,
            )
            lr_score = lr_scores[lr_].values
            spot_indices = np.where(lr_score > 0)[0]

            if save_bg:
                self.result["lrs_to_bg"][lr_] = background

            n_greater = (background.values[spot_indices, :] >= lr_score[spot_indices].reshape(-1, 1)).sum(axis=1)

            n_greater = np.where(n_greater != 0, n_greater, 1)
            pvals[spot_indices, idx] = n_greater / background.shape[0]

        if self.verbose:
            print('adjust p value...')
            # adjust p value
        from statsmodels.stats.multitest import multipletests

        def MHT(ar, adj_method):
            return multipletests(ar, method=adj_method)[1]

        pvals_adj = np.apply_along_axis(MHT, 1, pvals, adj_method=adj_method)
        self.result['LR_Pvals'] = pvals_adj

    def fit(
            self,
            lr_pairs,
            use_raw: bool = False,
            spot_comp: pd.DataFrame = None,  # TODO
            key_add: str = 'cci_score',
            adj_method: str = "fdr_bh",
    ):
        """run

        Parameters
        ----------
        lr_pairs : Union[list, np.array]
            LR pairs
        use_raw : bool, optional
            whether to use counts in `self.data.raw.X`, by default False
        spot_comp : `pd.DataFrame`, optional
            spot component of different cells, by default None
        key_add : str, optional
            key added in `self.result`, by default 'cci_score'
        adj_method : str, optional
            adjust method of p value, by default "fdr_bh"

        Raises
        ------
        ValueError
            _description_
        """
        # TODO: add the weight of cell(bin composition)
        # TODO: capitalize the genes

        # get neighbors
        import scipy.spatial as spatial
        neighbors = []
        point_tree = spatial.cKDTree(self.data.position)
        distance = self.bin_scale * self.distance

        for idx, point in enumerate(self.data.position):
            neighbor = point_tree.query_ball_point(point, distance)  # 这里5是指距离
            self_neighbor = [self.data.cell_names[idx]]
            neighbor.remove(idx)
            other_neighbor = self.data.cell_names[neighbor]

            if distance == 0:
                neighbors.append(self_neighbor)
            elif distance > 0:
                neighbors.append(other_neighbor)
            else:
                raise ValueError("`distance` should > 0")
        neighbors_key = f'neighbors_{distance}'

        self.result[neighbors_key] = neighbors

        if use_raw:
            self.data = self.data.raw

        # calulate lr scores
        self.calculate_score(lr_pairs=lr_pairs,
                             neighbors_key=neighbors_key,
                             key_add=key_add,
                             verbose=self.verbose)

        lr_scores = self.result[key_add]
        self.lr_pairs = lr_scores.columns.tolist()

        # permutation
        self._permutation(lr_scores=lr_scores,
                          lr_pairs=lr_pairs,
                          neighbors_key=neighbors_key,
                          adj_method=adj_method
                          )

        return self.result
