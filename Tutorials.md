# demo
以小鼠stereo-seq后整理的空间组学表达矩阵为例，利用stereopy工具对小鼠的空间组学进行数据分析。

数据矩阵格式如下, x, y分别为基因在组织切片的空间位置，count为基因表达数量？

|  GeneID   | x | y | count |
|  ----  | ----  | ----| ----|
| Gene1  | 121 | 200 | 2 |
| Gene2  | 234 | 300 | 1 |

该矩阵作为初始输入，分析流程大概分为如下几步。

### 1. read data
```python
import stereo as st
mouse_data_path = './path/to/matrix'
andata = read_stereo_data(mouse_data_path, bin_size=100)
```
为了方便处理，将矩阵信息取成andata的格式，andata将数据分成三个模块存储，其详细介绍在 *https://scanpy.readthedocs.io/en/latest/usage-principles.html#anndata*

由于stereo-seq是纳米级别的空间位置测序，所以每一个位置的捕捉到的表达基因数目有限，
为了达到更好的分析效果可以扩大空间，即通过设置bin size为参数，比如将范围内的10*10（bin_size=100）个位置合并成一个位置。

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

<img src="https://raw.githubusercontent.com/molindoudou/bio_tools/main/data/spatial_distribution.png" style="zoom:50%" alt="空间分布散点图" align=center />

空间分布散点图，能够展示小鼠的组织切片在空间范围的转录表达大体情况。

<img src="https://github.com/molindoudou/bio_tools/blob/main/data/violin_distribution.png?raw=true" style="zoom:50%" alt="小提琴图" align=center />

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

<img src="https://github.com/molindoudou/bio_tools/blob/main/data/spatial_cluster.png?raw=true" width = "500" height = "500" alt="聚类空间分布散点图" align=center />


对所有位点进行聚类后，再查看其空间分布情况

### 6.find marker
```python
marker = st.tools.FindMarker(data=andata, cluster='clustering', corr_method='bonferroni', method='t-test', name='marker_test')
marker.fit()
st.plots.plot_heatmap_maker_genes(andata, marker_uns_key='marker_test', cluster_method='clustering')
```

<img src="https://github.com/molindoudou/bio_tools/blob/main/data/heatmap.png?raw=true" style="zoom:50%" alt="热图" align=center />


### 7.annotation
```python
cell_anno = st.tools.CellTypeAnno(adata=andata)
cell_anno.fit()
st.plots.plot_degs(andata, key='marker_test')
```

<img src="https://github.com/molindoudou/bio_tools/blob/main/data/degs.png?raw=true" style="zoom:50%" alt="热图" align=center />

