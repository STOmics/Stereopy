import sys
from anndata import AnnData

sys.path.append('/data/workspace/st/stereopy-release')

from stereo.io.reader import read_stereo_data
import scanpy as sc
import numpy as np
import pandas as pd

path = '/data/workspace/st/stereopy-release/test/Gene_bin50_lassoleiden.h5ad'
adata = sc.read_h5ad(path)
adata = AnnData(adata.raw.X, var=pd.DataFrame(index=adata.var.index), obs=pd.DataFrame(index=adata.obs.index))
pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
pos[:, 1] = pos[:, 1] * -1
adata.obsm['spatial'] = pos

from stereo.preprocess.normalize import Normalizer


# print(not isinstance(adata, AnnData) and not isinstance(adata, pd.DataFrame))
print(Normalizer(data=adata, method='normalize_total', inplace=False, target_sum=10000).fit())
Normalizer(data=adata, method='quantile', inplace=True).fit()

print(adata.X)
