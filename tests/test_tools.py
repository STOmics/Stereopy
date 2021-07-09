import numpy as np
import pandas as pd
from stereo.tools.dim_reduce import DimReduce
from stereo.tools.clustering import Clustering
from stereo.tools.find_markers import FindMarker
from stereo.tools.spatial_pattern_score import SpatialPatternScore
# from stereo.tools.spatial_lag import SpatialLag
from stereo.tools.cell_type_anno import CellTypeAnno
from stereo.core.stereo_exp_data import StereoExpData

from anndata import AnnData
from stereo.plots.clustering import plot_spatial_cluster

from stereo.plots.scatter import plot_scatter

test_gem = 'D:\projects\data\sgm.gem'
# test_exp = 'D:\projects\data\sem.csv'
# test_gem = '/ldfssz1/ST_BI/USER/qindanhua/data/sgm.gem'
# test_exp = '/ldfssz1/ST_BI/USER/qindanhua/data/sem.csv'
# data = pd.read_csv(test_exp, index_col=[0]).dropna()
# data = data.loc[:, (data != 0).any(axis=0)]
# se = StereoExpData(exp_matrix=data.values, cells=data.index, genes=data.columns)

se = StereoExpData(test_gem, 'txt', 'bins')
# dr = DimReduce(se)
# dr.sparse2array()
# data = pd.DataFrame(dr.data.exp_matrix, columns=dr.data.gene_names, index=dr.data.cell_names)
# ad = AnnData(data)
# se.read_txt()
# dim
# dr = DimReduce(se)
# dr.fit()
# dr.method = 'umap'
# dr.fit()
# cluster
# ct = Clustering(se, normalization=True)
# ct.pca_x = dr
# ct.method = 'leiden'
# ct.fit()
# maker
# fm = FindMarker(se)
# fm = FindMarker(se, groups=ct.result.matrix)
# fm.fit()
# #
# sps = SpatialPatternScore(se)
# sps.fit()

# sl = SpatialLag(se, groups=ct.result.matrix)
# sl.fit()

ref_dir = 'D:\projects\data\FANTOM5'
ca = CellTypeAnno(se, ref_dir=ref_dir)
# ca = CellTypeAnno(se)
ca.fit()

# color_list = ['violet', 'turquoise', 'tomato', 'teal','tan', 'silver', 'sienna', 'red', 'purple', 'plum', 'pink',
#               'orchid', 'orangered', 'orange', 'olive', 'navy', 'maroon', 'magenta', 'lime',
#               'lightgreen', 'lightblue', 'lavender', 'khaki', 'indigo', 'grey', 'green', 'gold', 'fuchsia',
#               'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral', 'chocolate', 'chartreuse', 'brown', 'blue', 'black',
#               'beige', 'azure', 'aquamarine', 'aqua',
#               ]
# plot_scatter(se, ct.result.matrix, color_list=color_list)

#
# se = pd.DataFrame({'gene_1': [0, 1, 2, 0, 3, 4], 'gene_2': [1, 3, 2, 0, 3, 0], 'gene_3': [0, 0, 2, 0, 3, 1]},
#                   index=['cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5', 'cell_6'])


