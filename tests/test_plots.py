#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_plots.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/07/12  create file.
"""
from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.qc import cal_qc
from stereo.plots.qc import plot_violin_distribution
from stereo.plots.qc import save_fig
from stereo.plots.qc import plot_genes_count
from stereo.plots.qc import plot_spatial_distribution
from stereo.io.reader import read_stereo
import sys

def make_data():
    import numpy as np
    from scipy import sparse
    genes = ['g1', 'MT-111', 'g3']
    rows = [0, 1, 0, 1, 2, 0, 1, 2]
    cols = [0, 0, 1, 1, 1, 2, 2, 2]
    cells = ['c1', 'c2', 'c3']
    v = [2, 3, 4, 5, 6, 3, 4, 5]
    exp_matrix = sparse.csr_matrix((v, (rows, cols)))
    # exp_matrix = sparse.csr_matrix((v, (rows, cols))).toarray()
    position = np.random.randint(0, 10, (len(cells), 2))
    out_path = sys.argv[0]+"/F5.h5ad"
    data = StereoExpData(bin_type='cell_bins', exp_matrix=exp_matrix, genes=np.array(genes),
                         cells=np.array(cells), position=position, output=out_path)
    return data


def plot(data,out):
    """

    :param data:
    :return:
    """
    plot_violin_distribution(data)
    save_fig(out+"/F5.vio.png")
    plot_genes_count(data)
    save_fig(out+"/F5.gene.png")
    plot_spatial_distribution(data)
    save_fig(out+"/F5.spa.png")
    # plot_genes_count(data)


# add by qdh
def plot_object():
    import pickle
    from stereo.plots.plot_collection import PlotCollection
    from stereo.preprocess.qc import cal_qc
    with open('/ldfssz1/ST_BI/USER/qindanhua/projects/st/data/test_data.pickle', 'rb') as r:
        data = pickle.load(r)
    cal_qc(data)
    plt = PlotCollection(data)


if __name__ == '__main__':
    out=sys.argv[1]
    print(out)
    #data=make_data()
    data = read_stereo(out+"/F5.gem","bins",200)
    print(type(data.gene_names))
    print(data.gene_names.dtype)
    data = cal_qc(data)
    # print(data.cells.get_property("n_genes_by_counts"))
#    print(data.cells.n_genes_by_counts)
    plot(data,out)
    #plot_genes_count(data)
#    print(data.cells.total_counts)
#    print(data.cells.n_genes_by_counts)
#    print(data.position)
