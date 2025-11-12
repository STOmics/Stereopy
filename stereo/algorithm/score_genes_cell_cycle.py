from typing import Union, Tuple, List

import numpy as np
import pandas as pd

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.algorithm.score_genes import ScoreGenes
from stereo.log_manager import logger

class ScoreGenesCellCycle(AlgorithmBase):
    def __init__(self, stereo_exp_data, pipeline_res):
        super().__init__(stereo_exp_data=stereo_exp_data, pipeline_res=pipeline_res)
        self.score_genes = ScoreGenes(stereo_exp_data, pipeline_res)

    def main(
        self,
        s_genes: Union[np.ndarray, List[str], Tuple[str]],
        g2m_genes: Union[np.ndarray, List[str], Tuple[str]],
        **kwargs,
    ):
        """
        Score cell cycle genes.

        Given two lists of genes associated to S phase and G2M phase, calculates scores and assigns a cell cycle phase (G1, S or G2M).
        See `st.tl.score_genes <stereo.algorithm.score_genes.ScoreGenes.main.html>`_ for further information.

        :param s_genes: List of genes associated with S phase.
        :param g2m_genes: List of genes associated with G2M phase.
        :kwargs: Other parameters to be passed to `st.tl.score_genes` except `ctrl_size` and `res_key`, 
                the `ctrl_size` is set as the minimum of `len(s_genes)` and `len(g2m_genes)`.
        """
        logger.info("calculating cell cycle phase")

        if 'ctrl_size' in kwargs:
            del kwargs['ctrl_size']
        if 'res_key' in kwargs:
            del kwargs['res_key']

        ctrl_size = min(len(s_genes), len(g2m_genes))
        for genes, name in [(s_genes, "S_score"), (g2m_genes, "G2M_score")]:
            self.score_genes.main(genes, res_key=name, ctrl_size=ctrl_size, **kwargs)
        scores: pd.DataFrame = self.stereo_exp_data.cells[["S_score", "G2M_score"]]

        # default phase is S
        phase = pd.Series("S", index=scores.index)

        # if G2M is higher than S, it's G2M
        phase[scores["G2M_score"] > scores["S_score"]] = "G2M"

        # if all scores are negative, it's G1...
        phase[np.all(scores < 0, axis=1)] = "G1"

        self.stereo_exp_data.cells["phase"] = phase