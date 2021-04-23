
#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: test_data_helper.py
@time: 2021/03/16
"""

import sys

sys.path.append('/data/workspace/st/stereopy-release')


from stereo.utils.data_helper import select_group
import pandas as pd
from anndata import AnnData
import numpy as np

np.random.seed(9)


def init_andata(genes=50, cells=200):
    gname = [f'g{i}' for i in range(genes)]
    cname = [f'c{i}' for i in range(cells)]
    groups = np.random.choice(['1', '2', '3'], cells)
    x = np.random.randint(0, 100, (cells, genes))
    var = pd.DataFrame(index=gname)
    obs = pd.DataFrame(index=cname)
    obs['marker_genes'] = groups
    andata = AnnData(x, obs=obs, var=var)
    return andata


andata = init_andata()
print(andata.obs[andata.obs['marker_genes'] == '1'])
data = select_group(andata, groups='1', clust_key='marker_genes')
print(data)
