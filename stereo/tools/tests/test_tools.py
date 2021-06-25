import numpy as np
import pandas as pd
from stereo.tools.dim_reduce import DimReduce
from stereo.tools.clustering import Clustering
from stereo.tools.find_markers import FindMarker
from stereo.tools.spatial_pattern_score import SpatialPatternScore
from anndata import AnnData

data = pd.read_csv('D:\projects\data\sem.csv', index_col=[0]).dropna()
data = data.loc[:, (data != 0).any(axis=0)]
at = AnnData(data)
# dim
dr = DimReduce(at)
dr.fit()
# cluster
ct = Clustering(at)
# ct.pca_x = dr.
ct.fit()
# maker
fm = FindMarker(at)
fm.fit()
#
sps = SpatialPatternScore(at)
sps.fit()

