#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_reasder.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/08/13  create file.
"""
import numpy as np

import pandas as pd

from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.qc import cal_qc
from stereo.plots.qc import plot_violin_distribution
from stereo.plots.qc import save_fig
from stereo.plots.qc import plot_genes_count
from stereo.plots.qc import plot_spatial_distribution
from stereo.io.reader import read
from stereo.io.reader import read_txt
from stereo.io.reader import read_stereo
from stereo.io.reader import read_ann_h5ad
from stereo.io.writer import write
import sys
import scanpy as sc


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



def check_read_txt(out) :
    data = read_txt(out+"/F5.gem.txt",bin_type="bins",bin_size=200)
    myout = out+"/check_read_txt"
    check_out(myout)
    data=cal_qc(data)
    plot(data,myout)
    write(data, output=myout + "/F5.bin200.h5ad")
    print("finished in " + myout)
    print(data.to_df())

def check_read_stereo(out):
    data = read_stereo(out+"/check_write/F5.bin200.h5ad")
    myout = out+"/check_read_stereo"
    check_out(myout)
    data=cal_qc(data)
    plot(data,myout)
    write(data, output=myout + "/F5.bin200.h5ad")
    print("finished in " + myout)
    print(data.to_df())


def check_read_ann_h5ad(out):
    data = read_ann_h5ad(out+"/scanpy/raw_andata.bin200.h5ad")
    myout = out+"/check_read_ann_h5ad"
    check_out(myout)
    plot(data,myout)
    write(data,output=myout+"/F5.bin200.h5ad")
    print("finished in " + myout)
    print(data.to_df())


def check_stereo_to_andata(out):
    data = read(file_path=out+"/F5.gem.txt",file_format='txt',bin_type="bins",bin_size=200)
    #data=make_data()
    data = cal_qc(data)
    adata=data.to_andata()
    sc.write(out+"/check_stereo_to_andata/F5.stereo_to_andata.h5ad",adata)
    print(adata)

def check_andata_to_stereo(out):
    andata = sc.read(out+"/scanpy/raw_andata.bin200.h5ad")
    data=StereoExpData()
    data.andata_to_stereo(andata)

def check_read_h5ad(out):
    import numpy as np
    data=StereoExpData()
    data.read_h5ad_anndata(out+"/scanpy/raw_andata.bin200.h5ad")
    plot(data, out+"/check_read_h5ad")
    #print(data.cells.to_df())

def check_out(out):
    from pathlib import Path
    myout = Path(out)
    if not myout.exists():
        myout.mkdir()

if __name__ == '__main__':
    out=sys.argv[1]
    print(out)
    check_read_txt(out)
    check_read_stereo(out)
    check_read_ann_h5ad(out)
    check_stereo_to_andata(out)
    '''
    data=make_data()
    data = cal_qc(data)
    print(data.cells.to_df())
    print(data.genes.to_df())
    #data = read_stereo(out+"/F5.gem","bins",200)
    ##test andata_to_stereo
    #print(data.gene_names)
#    print(data.cells.n_genes_by_counts)
    #plot(data,out)
    #plot_genes_count(data)
#    print(data.cells.total_counts)
#    print(data.cells.n_genes_by_counts)
#    print(data.position)
'''
