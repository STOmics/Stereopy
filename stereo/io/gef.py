# coding: utf-8
import gc

import h5py
import pandas as pd
from scipy.sparse import csr_matrix

import numpy as np
from ..core.cell import Cell
from ..core.gene import Gene
from ..core.stereo_exp_data import StereoExpData
from ..log_manager import logger


class GEF(object):
    def __init__(self, file_path: str, bin_size: int = 100, is_sparse: bool = True):
        self.file_path = file_path
        self.bin_size = bin_size
        self.is_sparse = is_sparse
        self.df_exp = None
        self.df_gene = None
        self.genes = None
        self.cells = None
        self.cell_num = 0
        self.gene_num = 0
        self._init()

    def _init(self):
        with h5py.File(self.file_path, mode='r') as h5f:
            bin_tag = 'bin{}'.format(self.bin_size)
            if bin_tag not in h5f['geneExp'].keys():
                raise Exception('The bin size {} info is not in the GEF file'.format(self.bin_size))

            h5exp = h5f['geneExp'][bin_tag]['expression']
            h5gene = h5f['geneExp'][bin_tag]['gene']
            self.df_gene = pd.DataFrame(h5gene['gene', 'offset', 'count'])
            self.df_exp = pd.DataFrame(h5exp['x', 'y', 'count'])

    def build(self, gene_lst: list = None, region: list = None):
        if gene_lst is not None:
            self._restrict_to_genes(gene_lst)
        if region is not None:
            self._restrict_to_region(region)
        if gene_lst is None and region is None:
            self.genes = self.df_gene['gene'].values
            self.gene_num = len(self.genes)
            cols = np.zeros((self.df_exp.shape[0],), dtype='uint32')
            gene_index = 0
            exp_index = 0
            for count in self.df_gene['count']:
                for i in range(count):
                    cols[exp_index] = gene_index
                    exp_index += 1
                gene_index += 1
            self.df_exp['gene_index'] = cols

        self.df_exp['cell_id'] = np.bitwise_or(
            np.left_shift(self.df_exp['x'].astype('uint64'), 32), self.df_exp['y'])
        self.cells = self.df_exp['cell_id'].unique()
        self.cell_num = len(self.cells)
        rows = np.zeros((self.df_exp.shape[0],), dtype='uint32')
        grp = self.df_exp.groupby('cell_id').groups
        i = 0
        for cell_id in self.cells:
            for j in grp[cell_id]:
                rows[j] = i
            i += 1
        self.df_exp['cell_index'] = rows
        del grp
        gc.collect()

    def _restrict_to_region(self, region):
        logger.info(f'restrict to region [{region[0]} <= x <= {region[1]}] and [{region[2]} <= y <= {region[3]}]')
        gene_col = []
        for row in self.df_gene.itertuples():
            for i in range(getattr(row, 'count')):
                gene_col.append(getattr(row, 'gene'))

        self.df_exp['gene'] = gene_col
        self.df_exp = self.df_exp.query(f'{region[0]} <= x <= {region[1]} and {region[2]} <= y <= {region[3]}')

        self.genes = self.df_exp['gene'].unique()
        self.df_gene = None
        self.gene_num = len(self.genes)
        genes_dict = dict(zip(self.genes, range(0, self.gene_num)))
        self.df_exp['gene_index'] = self.df_exp['gene'].map(genes_dict)
        self.df_exp.drop(columns=['gene'])
        self.df_exp = self.df_exp.reset_index(drop=True)

    def _restrict_to_genes(self, gene_lst):
        logger.info('restrict to gene_lst')
        cols = np.zeros((self.df_exp.shape[0],), dtype='uint32')
        offset_indexes = np.zeros((self.df_exp.shape[0],), dtype='uint32')
        self.df_gene = self.df_gene.set_index('gene').loc[gene_lst].reset_index()
        self.genes = self.df_gene['gene'].values
        self.gene_num = len(self.genes)

        gene_index = 0
        exp_index = 0
        for row in self.df_gene.itertuples():
            for i in range(getattr(row, 'count')):
                cols[exp_index] = gene_index
                offset_indexes[exp_index] = getattr(row, 'offset') + i
                exp_index += 1
            gene_index += 1

        self.df_exp = self.df_exp.loc[offset_indexes[:exp_index]]
        self.df_exp['gene_index'] = cols[:exp_index]
        self.df_exp = self.df_exp.reset_index(drop=True)

    def to_stereo_exp_data(self) -> StereoExpData:
        data = StereoExpData(file_path=self.file_path)
        logger.info(f'the martrix has {self.cell_num} cells, and {self.gene_num} genes.')
        data.position = self.df_exp.loc[:, ['x', 'y']].drop_duplicates().values
        exp_matrix = csr_matrix((self.df_exp['count'], (self.df_exp['cell_index'], self.df_exp['gene_index'])),
                                shape=(self.cell_num, self.gene_num), dtype=np.int)
        data.cells = Cell(cell_name=self.cells)
        data.genes = Gene(gene_name=self.genes)
        data.exp_matrix = exp_matrix if self.is_sparse else exp_matrix.toarray()
        return data
