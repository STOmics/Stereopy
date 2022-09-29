Gaussian smoothing
----------------
This example show how to run the function of gaussian smooth on Stereopy.

Gaussian smoothing can make expression matrix closer reality, the detail of algorithm refer to https://www.biorxiv.org/content/10.1101/2022.05.26.493527v1.abstract.

Generally, you should do some preprocessing such as filtering cells, filtering genes, normalization, pca before running gaussian smooth.

Especially, you must to save raw expression matrix by running raw_checkpoint before all of the operations such as normalization those will change the values of expression matrix.

Once ran the raw_checkpoint, you can not run the operations those will change the 1-dimension of expression matrix before running gaussian smooth.

Also, you need to run pca before running gaussian smooth.

.. code:: python

    import stereo as st

    input_file = "./SS200000141TL_B5_raw.h5ad"
    data = st.io.read_ann_h5ad(input_file, spatial_key='spatial')
    data.tl.cal_qc()
    data.tl.filter_cells(min_gene=300, pct_counts_mt=10)
    data.tl.filter_genes(min_cell=10)
    data.tl.raw_checkpoint()
    data.tl.normalize_total(target_sum=10000)
    data.tl.log1p()
    data.tl.pca(use_highly_genes=False, n_pcs=50, svd_solver='arpack')
    data.tl.gaussian_smooth(n_neighbors=10, smooth_threshold=90)
    data.tl.scale(max_value=10) #only for gaussian_smooth_scatter_by_gene
    data.plt.gaussian_smooth_scatter_by_gene(gene_name='C1ql2')
    data.plt.gaussian_smooth_scatter_by_gene(gene_name='Irx2')
    data.plt.gaussian_smooth_scatter_by_gene(gene_name='Calb1')

+--------------------------------------------+--------------------------------+
|.. image:: ../_static/gaussian_smooth_1.png |.. image:: ../_static/C1ql2.jpg |
+--------------------------------------------+--------------------------------+
|.. image:: ../_static/gaussian_smooth_2.png |.. image:: ../_static/Inx2.jpg  |
+--------------------------------------------+--------------------------------+
|.. image:: ../_static/gaussian_smooth_3.png |.. image:: ../_static/cabl1.jpg |
+--------------------------------------------+--------------------------------+

After, if you want to do other operations such as clustering, you need to do the same preprocessing you did before.

Because of the preprocessing you did before just only for searching the nearest points, the result still base on the raw expression matrix saved by running raw_checkpoint.

.. code:: python

    import os
    import stereo as st

    input_file = "./SS200000141TL_B5_raw.h5ad"
    data = st.io.read_ann_h5ad(input_file, spatial_key='spatial')
    data.tl.cal_qc()
    data.tl.filter_cells(min_gene=300, pct_counts_mt=10)
    data.tl.filter_genes(min_cell=10)
    data.tl.raw_checkpoint()
    data.tl.normalize_total(target_sum=10000)
    data.tl.log1p()
    data.tl.pca(use_highly_genes=False, n_pcs=50, svd_solver='arpack')
    data.tl.gaussian_smooth(n_neighbors=10, smooth_threshold=90)
    data.tl.normalize_total(target_sum=10000)
    data.tl.log1p()
    data.tl.pca(use_highly_genes=False, n_pcs=50, svd_solver='arpack')
    data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors')
    data.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
    data.plt.cluster_scatter(res_key='leiden')

Gaussian smoothing can make clustering result to more subtypes.

+---------------------------------------------------+---------------------------------------------------+
|Before                                             |After                                              |
+===================================================+===================================================+
|.. image:: ../_static/clustering_before_smooth.png |.. image:: ../_static/clustering_after_smooth.png  |
+---------------------------------------------------+---------------------------------------------------+