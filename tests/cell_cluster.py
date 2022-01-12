#!/usr/bin/env python3
# coding: utf-8
"""
@file: cell_cluster.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2022/01/05  create file.
"""

import stereo as st
import argparse
from scipy.sparse import issparse


def cell_cluster(gef_file, bin_size):
    """
    run the cluster using stereopy.

    :param gef_file: gef file path.
    :param bin_size: bin size.
    :return: StereoExpData.
    """
    data = st.io.read_gef(gef_file, bin_size)
    data.tl.cal_qc()
    data.tl.filter_cells(min_gene=0)
    data.tl.raw_checkpoint()
    if issparse(data.exp_matrix):
        data.exp_matrix = data.exp_matrix.toarray()
    data.tl.normalize_total()
    data.tl.log1p()
    data.tl.highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='raw_highly_variable_genes')
    data.tl.pca(use_highly_genes=True, hvg_res_key='raw_highly_variable_genes', n_pcs=20, res_key='pca')
    data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors')
    data.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap')
    data.tl.leiden(neighbors_res_key='neighbors', res_key='louvain')
    return data


def stereo2anndata(data, out_file=None):
    """
    transform the StereoExpData object into Anndata object.

    :param data: StereoExpData
    :param out_file: h5ad output path of Anndata object
    :return:
    """
    return st.io.stereo_to_anndata(data, output=out_file)


def args_parse():
    """
    get the args.

    :return:
    """
    usage = 'Usage: python ./cell_cluster.py -i ./gef.h5 -o ./cluster.h5ad -s 100.'
    ap = argparse.ArgumentParser(usage=usage)
    ap.add_argument('-s', '--bin_size', action='store', type=int, default=50, required=True,
                    help='The bin size or max bin szie that to combine the dnbs. default=50')
    ap.add_argument('-i', '--gef_file', action='store', required=True, help='gef file path.')
    ap.add_argument('-o', '--out_path', action='store', required=True, help='outfile path')
    return ap.parse_args()


def main():
    args = args_parse()
    if args.bin_size not in [1, 10, 20, 50, 100, 200, 500]:
        raise Exception('the bin size is out of range, please check. the range of gef binsize '
                        'is [1,10,20,50,100,200,500]')
    stereo_data = cell_cluster(args.gef_file, args.bin_size)
    stereo2anndata(stereo_data, args.out_path)


if __name__ == '__main__':
    main()
