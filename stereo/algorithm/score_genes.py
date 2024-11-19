from typing import Union, Tuple, List

import pandas as pd
import numpy as np
from scipy.sparse import spmatrix, issparse
from scipy import stats

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.log_manager import logger


class ScoreGenes(AlgorithmBase):

    def _expression_mean(
        self,
        exp_matrix: Union[np.ndarray, spmatrix],
        axis: int
    ) -> np.ndarray:
        if issparse(exp_matrix):
            s = exp_matrix.sum(axis=axis, dtype=np.float64)
            m = s / exp_matrix.shape[axis]
            return m.A.flatten()
        return exp_matrix.mean(axis=axis, dtype=np.float64)

    def _get_expression_subset(
        self,
        genes: np.ndarray,
        use_raw: bool
    ) -> Union[np.ndarray, spmatrix]:
        data = self.stereo_exp_data
        gene_names = data.raw.gene_names if use_raw else data.gene_names
        exp_matrix = data.raw.exp_matrix if use_raw else data.exp_matrix

        if len(genes) == len(gene_names):
            return exp_matrix
        idx = pd.Index(gene_names).get_indexer(genes)
        return exp_matrix[:, idx]

    def _check_score_genes(
        self,
        genes_used: np.ndarray,
        genes_reference: Union[np.ndarray, None],
        use_raw: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restrict `genes_used` and `genes_reference` to present genes in `data`.
        """
        data = self.stereo_exp_data
        gene_names = data.raw.gene_names if use_raw else data.gene_names
        genes_used = np.array([genes_used] if isinstance(genes_used, str) else genes_used, dtype='U')
        isin = np.isin(genes_used, gene_names)
        genes_to_ignore = genes_used[~isin]  # first get missing
        genes_used = genes_used[isin]  # then restrict to present
        if len(genes_to_ignore) > 0:
            logger.warning(f"genes are not in gene_names and ignored: {genes_to_ignore}")
        if len(genes_used) == 0:
            raise ValueError("No valid genes were passed for scoring.")

        if genes_reference is None:
            genes_reference = gene_names
        else:
            genes_reference = np.array([genes_reference] if isinstance(genes_reference, str) else genes_reference, dtype='U')
            genes_reference = np.intersect1d(genes_reference, gene_names)
        if len(genes_reference) == 0:
            raise ValueError("No valid genes are passed for reference set.")

        return genes_used, genes_reference


    def _score_genes_bins(
        self,
        genes_used: np.ndarray,
        genes_reference: np.ndarray,
        ctrl_as_ref: bool,
        ctrl_size: int,
        n_bins: int,
        use_raw: bool
    ) -> np.ndarray:
        # mean expression of genes in `genes_reference`
        exp_matrix = self._get_expression_subset(genes_reference, use_raw)
        genes_exp_mean = self._expression_mean(exp_matrix, axis=0)
        n_items = int(np.round(len(genes_exp_mean) / (n_bins - 1)))
        cells_cut = stats.rankdata(genes_exp_mean, method='min') // n_items
        keep_ctrl_in_cells_cut = np.zeros(genes_reference.size, dtype=bool) if ctrl_as_ref else np.isin(genes_reference, genes_used)

        # now pick `ctrl_size` genes from every cut
        control_genes = pd.array([], dtype="U")
        isin_used = np.isin(genes_reference, genes_used)
        cells_cut_iterable = np.unique(cells_cut[isin_used])
        for cut in cells_cut_iterable:
            r_genes = genes_reference[(cells_cut == cut) & ~keep_ctrl_in_cells_cut]
            if len(r_genes) == 0:
                msg = (
                    f"No control genes for {cut=}. You may need to increase the"
                    f"size of genes_reference (current size: {len(genes_reference)})"
                )
                logger.warning(msg)
            if ctrl_size < len(r_genes):
                r_genes = np.random.choice(r_genes, ctrl_size, replace=False)
            if ctrl_as_ref:  # otherwise `r_genes` is already filtered
                r_genes = np.setdiff1d(r_genes, genes_used)
            control_genes = np.union1d(control_genes, r_genes)
        return control_genes

    def main(
        self,
        genes_used: Union[np.ndarray, List[str], Tuple[str]],
        ctrl_as_ref: bool = True,
        ctrl_size: int = 50,
        genes_reference: Union[np.ndarray, List[str], Tuple[str], None] = None,
        n_bins: int = 25,
        random_state: Union[int, np.random.RandomState, None] = 0,
        use_raw: bool = None,
        res_key: str = "score",
    ):
        """
        Score a set of genes for each cell/bin.

        The score is the average expression of a set of genes subtracted with the
        average expression of a reference set of genes. The reference set is
        randomly sampled from the `genes_reference` for each binned expression value.

        :param genes_used: The list of gene names used for score calculation.
        :param ctrl_as_ref: Allow to use the control genes as reference, defaults to True
        :param ctrl_size: Number of reference genes to be sampled from each bin, defaults to 50,
                            you can set `ctrl_size=len(genes_used)` if the length of `genes_used` is not too short.
        :param genes_reference: Genes for sampling the reference set, default is all genes.
        :param n_bins: Number of expression level bins for sampling, defaults to 25
        :param random_state: The random seed for sampling, defaults to 0, fixed value to fixed result.
        :param use_raw: Whether to use the `data.raw`, defaults to `True` if `data.raw` is not `None`
        :param res_key: the column name of the result to be added in `data.cells`, defaults to "score"
        """
        logger.info(f"calculating score, the result will be saved in data.cells['{res_key}']")

        if random_state is not None:
            np.random.seed(random_state)

        if not isinstance(genes_used,(np.ndarray, list, tuple)):
            raise ValueError("genes_used must be a list, tuple or numpy array.")

        if isinstance(genes_used, (list, tuple)):
            genes_used = np.array(genes_used, dtype="U")

        if genes_reference is not None:
            if not isinstance(genes_reference, (np.ndarray, list, tuple, str)):
                raise ValueError("genes_reference must be a list, tuple, numpy array or string.")
            if isinstance(genes_reference, str):
                genes_reference = [genes_reference]
            if isinstance(genes_reference, (list, tuple)):
                genes_reference = np.array(genes_reference, dtype="U")

        data = self.stereo_exp_data
        if use_raw is None:
            use_raw = True if data.raw is not None else False
        else:
            use_raw = use_raw and data.raw is not None

        genes_used, genes_reference = self._check_score_genes(
            genes_used, genes_reference, use_raw
        )

        # Trying here to match the Seurat approach in scoring cells.
        # Basically we need to compare genes against random genes in a matched
        # interval of expression.
        control_genes = self._score_genes_bins(
            genes_used,
            genes_reference,
            ctrl_as_ref=ctrl_as_ref,
            ctrl_size=ctrl_size,
            n_bins=n_bins,
            use_raw=use_raw
        )

        if len(control_genes) == 0:
            msg = "No control genes found in any cut."
            if ctrl_as_ref:
                msg += " Try setting `ctrl_as_ref` to False."
            raise RuntimeError(msg)

        means_list = self._expression_mean(
            self._get_expression_subset(genes_used, use_raw), axis=1
        )
        means_control = self._expression_mean(
            self._get_expression_subset(control_genes, use_raw), axis=1
        )
        score = means_list - means_control

        self.stereo_exp_data.cells[res_key] = pd.Series(
            score, index=self.stereo_exp_data.cells.cell_name, dtype=np.float64
        )