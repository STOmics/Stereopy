# 绘图数据需求

以 AnnData 为参照获取对应的输入数据

## plot_spatial_distribution

### 所需 AnnData 数据：

- 图像坐标信息： 
    - `spatial_data = np.array(adata.obsm["spatial"])`

- 图像散点数值信息： 
    - `color_data = np.array(adata.obs_vector(key))`
    - 其中 key 为散点所需呈现的数据种类，如 total_counts, n_genes_by_counts，可利用的数据种类都在 `adata.obs.keys()`

### 实际所需数据格式：

- 图像坐标信息demo：
    
    ```Text
    array([[ 256,  -91],[ 185, -157],[ 257, -288],...,[ 151, -246],[ 290, -141],[ 177, -132]])
    ```

- 图像散点数值信息demo：
    
    ```Text
    array([3582., 1773., 2283., ..., 4062., 4371., 1172.], dtype=float32)
    ```

其中散点数值和坐标信息一一对应

## plot_spatial_cluster && plot_cluster_umap

这两个函数整合成了 plot_cluster_result，区别在于所用的 spatial data （pos_key） 不同， plot_spatial_cluster 使用的是 spatial 空间数据，plot_cluster_umap 使用的是 X_umap 的主成分值数据

### 所需 AnnData 数据：

- 图像坐标信息： 
    - `spatial_data = np.array(adata.obsm[pos_key])`

- 图像散点数值信息： 
    - `color_data = np.array(adata.obs_vector(key))`
    - 其中 key 为散点所需呈现的数据种类，可利用的数据种类都在 adata.obs.keys()，此处用到的主要是聚类相关数据，如 ["phenograph", "leiden"]
    - 在 cluster 相关的图像中，散点数据属于 categorical 数据

### 实际所需数据格式：

- 图像坐标信息demo：
    
    spatial:

    ```Text
    array([[ 256,  -91],[ 185, -157],[ 257, -288],...,[ 151, -246],[ 290, -141],[ 177, -132]])
    ```

    umap：
    
    ```Text
    array([[-2.234908  ,  3.7087667 ], [ 3.7431364 ,  2.003201  ], [ 4.849478  , -0.21215418], ..., [ 6.861894  ,  1.7589074 ], [-0.4605783 ,  4.0366364 ], [ 7.3391566 , -0.40746352]], dtype=float32)
    ```

- 图像散点数值信息demo：
    
    ```Text
    ['0', '3', '5', '3', '2', ..., '4', '3', '6', '1', '5']
    Length: 22726
    Categories (10, object): ['0', '1', '2', '3', ..., '6', '7', '8', '9']
    ```

其中散点数值和坐标信息一一对应


## plot_to_select_filter_value

### 所需 AnnData 数据

- `x = adata.obs_vector(var1)`
- `y = adata.obs_vector(var2)`

var1 var2 可以是任意两个 obs_key 中的值

### 实际所需数据格式

```Text
array([3582., 1773., 2283., ..., 4062., 4371., 1172.], dtype=float32)
```

## plot_variable_gene

### 所需 AnnData 数据：

- `adata.var.highly_variable`
- `adata.var.means`
- `adata.var.dispersions`
- `adata.var.dispersions_norm`

### 实际所需数据格式

pandas.core.series.Series

例如： adata.var.means:

```Text
AL355102.2    0.012042
SLC25A2       0.000596
                ...   
AC069214.1    0.009608
AL356056.1    0.001655
Name: means, Length: 33304, dtype: float64
```

## plot_expression_difference

### 所需 AnnData 数据

- 聚类得到的类名： `group_names = adata.uns["rank_genes_groups"]['names'].dtype.names`
- 在各个类名（group_name in group_names）的基础上获取各个类的基因列表('names')和分数('scores')：
    - `gene_names = adata.uns["rank_genes_groups"]['names'][group_name][:number_of_gene_to_show]`
    - `scores = adata.uns["rank_genes_groups"]['scores'][group_name][:number_of_gene_to_show]`

### 实际所需数据格式：

- 聚类得到的类名：

    ```Text
    tuple('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    ```
- 各个类名对应的基因列表和分数列表：
  
    ```Text
    numpy.ndarray
    
    array(['TUBB4B', 'SLC12A2', 'EPHB3', ..., 'TFF3', 'TIMP1', 'OLFM4'],
      dtype=object)
    
    array([ 28.094688,  27.638664,  24.523687, ..., -41.63738 , -42.894638, -48.384773], dtype=float32)
    ```

## plot_violin_distribution

### 所需 AnnData 数据

- `adata.obs['total_counts']`
- `adata.obs['n_genes_by_counts']`
- `adata.obs['pct_counts_mt']`

### 实际所需数据格式

pandas.core.series.Series

例如： total_counts:

```Text
256-91     3582.0
185-157    1773.0
            ...  
290-141    4371.0
177-132    1172.0
Name: total_counts, Length: 22726, dtype: float32
```

## plot_heatmap_maker_genes

### 所需 AnnData 数据：

- 聚类得到的类名： `marker_clusters = adata.uns["rank_genes_groups"]['names'].dtype.names`
- 在各个类名（cluster in marker_clusters）的基础上获取各个类的基因列表('names'):
    - `genes_array = adata.uns["rank_genes_groups"]['names'][group_name][:number_of_gene_to_show]`
- 设定表达量矩阵（热图矩阵）
    - 矩阵index： `adata.obs_name`
    - 整合各个类的基因列表，获取 uniq gene list，然后获取这些uniq gene 的表达量来构建 pandas.DataFrame
        - 表达量： `exp_matrix = adata.X[tuple([slice(None), adata.var.index.get_indexer(uniq_gene_names)])]`
        - pd.DataFrame:  `pd.DataFrame(exp_matrix,  columns=uniq_gene_names, index=adata.obs_names)`
    - 添加 obs 列在最开头： `pd.concat([draw_df, adata.obs[cluster_method]], axis=1)`
    

### 实际所需数据格式：

- 聚类得到的类名：

    ```Text
    tuple('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    ```

- 各个类名对应的基因列表：
  
    ```Text
    numpy.ndarray
    
    array(['TUBB4B', 'SLC12A2', 'EPHB3', ..., 'TFF3', 'TIMP1', 'OLFM4'],
      dtype=object)
    ```

- 表达量矩阵最终效果：
  
    ```Text
                      TUBB4B   SLC12A2     EPHB3    MT-ND1       FTX  NAALADL2  
        phenograph                                                               
        0           3.758313  1.884562  2.705292  4.040148  2.498723  2.238069   
        0           3.118562  1.224227  1.758388  3.311160  1.758388  0.000000   
        0           2.790583  0.000000  0.000000  3.175375  0.000000  0.000000   
        0           0.000000  1.885036  1.885036  3.821939  0.000000  0.000000   
        0           2.596922  0.000000  2.596922  3.644590  2.596922  2.596922   
        ...              ...       ...       ...       ...       ...       ...   
        9           3.172084  1.570598  0.000000  4.186041  1.570598  0.000000   
        9           0.000000  1.847625  0.000000  3.893912  0.000000  2.458688   
        9           2.681648  2.681648  0.000000  3.339970  1.314215  1.314215   
        9           1.955572  1.955572  0.000000  3.229691  1.955572  0.000000   
        9           2.963927  1.721601  1.721601  3.176694  1.721601  0.000000     
    ```


