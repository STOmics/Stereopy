# demo
a pipeline using mouse's stereo-seq data.

the format of input matrix is:

|  GeneID   | x | y | count |
|  ----  | ----  | ----| ----|
| Gene1  | 121 | 200 | 2 |
| Gene2  | 234 | 300 | 1 |
### 1. read data
```python
import stereo as st
mouse_data_path = './path/to/matrix'
andata = read_stereo_data(mouse_data_path)
```
about andata: *https://scanpy.readthedocs.io/en/latest/usage-principles.html#anndata*

### 2.preprocess
```python
# quality control
andata = st.preprocess.cal_qc(andata=andata)
# filter
st.preprocess.filter_cells(adata=andata, min_gene=200, n_genes_by_counts=3, pct_counts_mt=4, inplace=True)
# normalize
st.preprocess.Normalizer(data=andata, method='normalize_total', inplace=True, target_sum=10000).fit()

```
预处理主要包括质控、过滤和标准化三个部分，返回的都是处理后的andata, 也可以用inplace参数直接替之前的andata

### 3.spatial distribution visualization
```python
st.plots.plot_spatial_distribution(andata)
# plt.savefig('./data/spatial_distribution.png')
st.plots.plot_violin_distribution(andata)
```
![空间分布图](http://172.16.222.161:2333/data/spatial_distribution.png)
空间分布散点图，能够展示小鼠的组织切片在空间范围的转录表达大体情况。

![小提琴图](http://172.16.222.161:2333/data/violin_distribution.png)

小提琴图
### 4.Dimensionality reduction
```python
dim_reduce = st.tools.DimReduce(andata=andata, method='pca', n_pcs=30, min_variance=0.01, n_iter=250, n_neighbors=10, min_dist=0.3, inplace=False, name='dim_reduce')
dim_reduce.fit()
pca_x = dim_reduce.result.x_reduce
```

降维分析
### 5.cluster
```python
cluster = st.tools.Clustering(data=andata, method='leiden', outdir=None, dim_reduce_key='dim_reduce', n_neighbors=30, normalize_key='cluster_normalize', normalize_method=None, nor_target_sum=10000, name='clustering')
cluster.fit()
st.plots.plot_spatial_cluster(andata, obs_key=['clustering'])
```
![聚类分布图](http://172.16.222.161:2333/data/spatial_cluster.png)

对所有位点进行聚类后，再查看其空间分布情况

### 6.find marker
```python
marker = st.tools.FindMarker(data=andata, cluster='clustering', corr_method='bonferroni', method='t-test', name='marker_test')
marker.fit()
st.plots.plot_heatmap_maker_genes(andata, marker_uns_key='marker_test', cluster_method='clustering')
```

![聚类分布图](http://172.16.222.161:2333/data/heatmap.png)

### 7.annotation
```python
cell_anno = st.tools.CellTypeAnno(adata=andata)
cell_anno.fit()
st.plots.plot_degs(andata, key='marker_test')
```

![聚类分布图](http://172.16.222.161:2333/data/degs.png)
