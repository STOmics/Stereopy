import numpy as np
import pandas as pd
from stereo.tools.dim_reduce import DimReduce
from stereo.tools.clustering import Clustering
from stereo.tools.find_markers import FindMarker
from stereo.tools.spatial_pattern_score import SpatialPatternScore
from stereo.core.stereo_exp_data import StereoExpData

test_gem = 'D:\projects\data\sgm.gem'
test_exp = 'D:\projects\data\sem.csv'
data = pd.read_csv(test_exp, index_col=[0]).dropna()
data = data.loc[:, (data != 0).any(axis=0)]
se = StereoExpData(exp_matrix=data.values, cells=data.index, genes=data.columns)

# se.read_txt()
# dim
dr = DimReduce(se)
dr.fit()
dr.method = 'umap'
dr.fit()
# cluster
ct = Clustering(se)
ct.pca_x = dr
ct.method = 'leiden'
ct.fit()
# maker
# fm = FindMarker(se)
fm = FindMarker(se, groups=ct.result.matrix)
fm.fit()
#
sps = SpatialPatternScore(se)
sps.fit()

#
# se = pd.DataFrame({'gene_1': [0, 1, 2, 0, 3, 4], 'gene_2': [1, 3, 2, 0, 3, 0], 'gene_3': [0, 0, 2, 0, 3, 1]},
#                   index=['cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5', 'cell_6'])


