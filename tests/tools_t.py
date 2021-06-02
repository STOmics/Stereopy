import time
import sys
import pickle

sys.path.append('/data/workspace/st/test_qdh/stereopy')

import stereo as st
from stereo.io.reader import read_stereo_data
# import pandas as pd

mouse_data_path = '/data/workspace/st/test_qdh/mouse/DP8400013846TR_F5.gem'

# def get_data:

if __name__ == '__main__':
    andata = pickle.load(open('./temp.pk', 'rb'))  # load

    # andata = read_stereo_data(mouse_data_path)  # 读取bin文件
    # andata = st.preprocess.cal_qc(andata=andata)   # 指控
    # st.preprocess.filter_cells(adata=andata, min_gene=200, n_genes_by_counts=3, pct_counts_mt=4, inplace=True)
    # normalizer = st.preprocess.Normalizer(data=andata, method='normalize_total', inplace=False,
    #                                       target_sum=10000, name='normalize_total')
    # nor_total = normalizer.fit()
    # normalizer.method = 'quantile'
    # nor_quantal = normalizer.fit()

    # dim_reduce = st.tools.DimReduce(data=andata, method='pca', n_pcs=30, min_variance=0.01, n_iter=250,
    #                                 n_neighbors=10, min_dist=0.3, inplace=False, name='dim_reduce')
    # dim_reduce.fit()
    # pca_x = dim_reduce.result.x_reduce

    # cluster = st.tools.Clustering(data=andata, method='leiden', outdir=None, dim_reduce_key='dim_reduce',
    #                               n_neighbors=30, normalize_key='cluster_normalize',
    #                               normalize_method=None, nor_target_sum=10000, name='clustering')
    # cluster.fit()
    # marker = st.tools.FindMarker(data=andata, cluster='clustering', corr_method='bonferroni',
    #                              method='t-test', name='marker_test')
    # marker.fit()

    # cell_anno = st.tools.CellTypeAnno(adata=andata)
    # cell_anno.fit() 
    # print(time.asctime(time.localtime(time.time())))
    # lag = st.tools.SpatialLag(data=andata, cluster='clustering')
    # lag.fit()
    # normalizer = st.preprocess.Normalizer(data=andata, method='zscore_disksmooth',
    #                                       inplace=False, r=20, name='zscore_disksmooth')
    # nor_total = normalizer.fit()
    # andata.X = andata.raw.X
    # spc = st.tools.SpatialPatternScore(data=andata)
    # spc.fit()
    #
    print(andata)
    pickle.dump(andata, open('./temp2.pk', 'wb'))

    # print('可视化')

    # t = st.plots.plot_spatial_distribution(andata)
    # # plt.show()
    # plt.savefig('./output/0514/spatial_distribution.png')
    #
    # st.plots.plot_violin_distribution(andata)
    # plt.savefig('./output/0514/violin_distribution.png')
    #
    # st.plots.plot_spatial_cluster(andata, obs_key=['clustering'])
    # plt.savefig('./output/0514/spatial_cluster.png')
    #
    # st.plots.plot_dim_reduce(andata, obs_key=['clustering'])
    # plt.savefig('./output/0514/dim_reduce.png')
    #
    # st.plots.plot_heatmap_maker_genes(andata, marker_uns_key='marker_test', cluster_method='clustering')
    # plt.savefig('./output/0514/heatmap.png')

    # st.plots.plot_degs(andata, key='marker_test')
    # plt.savefig('./output/0514/degs.png')
    # print('保存结果')
