#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: test.py
@time: 2023/Mar/09
@description: test file for inference gene regulatory networks module
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI
"""

import sys
import json
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from multiprocessing import cpu_count
from regulatory_network import InferenceRegulatoryNetwork, PlotRegulatoryNetwork


if __name__ == '__main__':
    #
    tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/tfs/test_mm_mgi_tfs.txt'
    database_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/database/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
    motif_anno_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/motifs/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl'

    # clustering output for stereopy data]
    meta_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/meta_mousebrain.csv'
    
    # h5ad data
    #fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/MouseBrainCellbin.h5ad'
    #data = InferenceRegulatoryNetwork.load_anndata_by_cluster(fn, 'psuedo_class', ['HBGLU9', 'TEINH12', 'HBSER4'])

    # alternative: small h5ad for testing
    fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/test_grn_MouseBrainCellbin.h5ad'
    data = InferenceRegulatoryNetwork.read_file(fn)

    # create grn
    grn = InferenceRegulatoryNetwork(data)
    grn.main(database_fn, motif_anno_fn, tfs_fn, num_workers=cpu_count(), save=False)

    grn_plot = PlotRegulatoryNetwork(data)

    meta = pd.read_csv(meta_fn)
    rdict = json.load(open('regulons.json'))
    #rdict = grn.regulon_dict

    grn_plot.dotplot_stereo(data, meta, rdict, ['Zfp354c'], ['VLMC2','ABC','ACNT1'], 'cell_type')
    grn_plot.dotplot_anndata(data, grn.gene_names, cluster_label='psuedo_class')
    grn_plot.auc_heatmap(grn.auc_mtx)
    grn_plot.plot_2d_reg_h5ad(data, 'spatial', grn.auc_mtx, 'Zfp354c')
