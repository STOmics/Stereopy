# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 14:52
# @Author  : liuxiaobin
# @File    : analysis_helper.py
# @Version：V 0.1
# @desc :


import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from fbpca import pca
from geosketch import gs

from stereo.log_manager import logger

warnings.simplefilter("ignore", FutureWarning)


class Subsampler(object):
    def __init__(self, log: bool, num_pc: int = 100, num_cells: int = None, debug_seed: int = None):
        self.log = log
        self.num_pc = num_pc
        self.num_cells = num_cells
        np.random.seed(debug_seed)

    def subsample(self, counts: pd.DataFrame, counts_pca: pd.DataFrame = None) -> pd.DataFrame:
        input_genes = counts.shape[1]  # number of cells

        if self.num_cells is None:
            self.num_cells = int(input_genes / 3)

        logger.info('Subsampling {} to {}'.format(input_genes, self.num_cells))

        try:
            counts_t = counts.T
            if counts_pca is None:
                logger.info(f"Running PCA, pcs is {self.num_pc}")
                if self.log:
                    pca_input = np.log1p(counts_t)  # natural log, ln(x+1）
                else:
                    pca_input = counts_t

                u, s, vt = pca(pca_input.values, k=self.num_pc)
                x_dimred = u[:, :self.num_pc] * s[:self.num_pc]
            else:
                x_dimred = counts_pca.values
            sketch_index = gs(x_dimred, self.num_cells, replace=False)
            x_matrix = counts_t.iloc[sketch_index]
        except Exception as e:
            logger.warning('Subsampling failed: ignored.')
            logger.warning(str(e))
            return counts

        logger.info('Done subsampling {} to {}'.format(input_genes, self.num_cells))

        return x_matrix.T


def write_to_file(
        df: pd.DataFrame,
        filename: str,
        output_path: str,
        output_format: Optional[str] = None,
        index: bool = False
):
    _, file_extension = os.path.splitext(filename)

    if output_format is None:
        if not file_extension:
            default_format = 'txt'
            default_extension = '.{}'.format(default_format)

            separator = get_separator(default_extension)
            filename = '{}{}'.format(filename, default_extension)
        else:
            separator = get_separator(file_extension)
    else:
        selected_extension = '.{}'.format(output_format)

        if file_extension != selected_extension:
            separator = get_separator(selected_extension)
            filename = '{}{}'.format(filename, selected_extension)

            if file_extension:
                logger.warning(
                    'Selected extension missmatches output filename ({}, {}): It will be added => {}'.format(
                        selected_extension, file_extension, filename))
        else:
            separator = get_separator(selected_extension)

    df.to_csv('{}/{}'.format(output_path, filename), sep=separator, index=index)


def get_separator(mime_type_or_extension: str) -> str:
    extensions = {
        '.csv': ',',
        '.tsv': '\t',
        '.txt': '\t',
        '.tab': '\t',
        'text/csv': ',',
        'text/tab-separated-values': '\t',
    }
    default_separator = ','

    return extensions.get(mime_type_or_extension.lower(), default_separator)


def mouse2human(genes, ref_path):
    """
    Transfer mouse genes to their human homologous genes.
    If one-to-many, use the first. If mouse gene not in the reference data, mark it as 'NotAvailable'.

    :param genes: list of mouse genes (hgnc symbols)
    :param ref_path: path of the reference homologous gene relationship file.
    :return: list of human homologous
    """

    ref_separator = get_separator(os.path.splitext(ref_path)[-1])
    ref_df = pd.read_csv(ref_path, sep=ref_separator)

    ref_df = ref_df.drop_duplicates(subset='genesymbol_source', keep='first')
    ref_dict = dict(ref_df.loc[:, ['genesymbol_source', 'genesymbol_target']].values)
    genes_human = [ref_dict[x] if x in ref_dict.keys() else 'NotAvailable' for x in genes]
    human_genes_to_mouse = {ref_dict[x]: x for x in genes if x in ref_dict}
    return genes_human, human_genes_to_mouse
