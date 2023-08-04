# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 13:41
# @Author  : liuxiaobin
# @File    : CCC_visualization_process.py
# @Version：V 0.1
# @desc :

from functools import partial
# import scanpy as sc
import pandas as pd
# import numpy as np
# import anndata
# import sys
# sys.path.append(r'E:\VT3D\test_stereopy_3D_browser-main')
# from stereopy_3D_browser import launch
# import plotly.graph_objs as go


def preprocess_significant_means_result(significant_means: pd.DataFrame):
    # drop_list = ['id_cp_interaction', 'partner_a', 'partner_b', 'gene_a', 'gene_b', 'secreted', 'receptor_a',
    #              'receptor_b', 'annotation_strategy', 'is_integrin', 'rank']
    drop_list = ['id_cp_interaction', 'gene_a', 'gene_b', 'secreted', 'receptor_a', 'receptor_b', 'annotation_strategy', 'is_integrin', 'rank']
    significant_means = significant_means.drop(drop_list, axis=1)
    significant_means = significant_means.dropna(subset=significant_means.columns.difference(['interacting_pair']), how='all')
    significant_means = significant_means.set_index('interacting_pair')
    return significant_means


def preprocess_data(
    significant_means: pd.DataFrame,
    separator_cluster='|',
    separator_interaction='_'
):
    result = []
    for index, row in significant_means.iterrows():
        celltype_pair = row.dropna()
        partner_a, partner_b = row['partner_a'], row['partner_b']
        if partner_a.startswith('complex') or partner_b.startswith('complex'):
            continue
        celltype_pair = celltype_pair[2:]
        for pair in celltype_pair.index:
            current = pair.split(separator_cluster) + index.split(separator_interaction)
            result.append(current)
    return pd.DataFrame(result, columns=['celltype1', 'celltype2', 'ligand', 'receptor'])

def visualization_process(
    significant_means: pd.DataFrame,
    separator_celltype='|',
    separator_lr='_',
    human_genes_to_mouse=None
):
    significant_means = preprocess_significant_means_result(significant_means)
    visualization_data = preprocess_data(significant_means, separator_celltype, separator_lr)
    if human_genes_to_mouse is not None:
        def human2mouse(row, human_genes_to_mouse):
            ligand = row['ligand'] if row['ligand'] not in human_genes_to_mouse else human_genes_to_mouse[row['ligand']]
            receptor = row['receptor'] if row['receptor'] not in human_genes_to_mouse else human_genes_to_mouse[row['receptor']]
            return row['celltype1'], row['celltype2'], ligand, receptor
        apply_func = partial(human2mouse, human_genes_to_mouse=human_genes_to_mouse)
        visualization_data = visualization_data.apply(apply_func, axis=1, result_type='broadcast')
    return visualization_data


# def get_cell_type_1(adata):
#     """
#     Return list of all possible celltype1.
#     """
#     celltype1 = list(set(adata.uns['ccc_data']['celltype1']))
#     celltype1.sort()
#     return celltype1


# def get_cell_type_2(adata, celltype1):
#     """
#     Given a celltype1, return a list of all possible celltype2.
#     """
#     data = adata.uns['ccc_data']
#     data1 = data[data['celltype1'] == celltype1]
#     celltype2 = list(set(data1['celltype2']))
#     celltype2.sort()
#     return celltype2


# def get_ligand(adata, celltype1, celltype2):
#     """
#     Given a celltype1 and a celltype2, return a list of all possible ligands.
#     """
#     data = adata.uns['ccc_data']
#     data12 = data[(data['celltype1'] == celltype1) & (data['celltype2'] == celltype2)]
#     ligand = list(set(data12['ligand']))
#     ligand.sort()
#     return ligand


# def get_receptor(adata, celltype1, celltype2, ligand):
#     """
#     Given a celltype1, a celltype2 and a ligand, return a list of all possible receptors.
#     """
#     data = adata.uns['ccc_data']
#     data12 = data[(data['celltype1'] == celltype1) & (data['celltype2'] == celltype2) & (data['ligand'] == ligand)]
#     receptor = list(set(data12['receptor']))
#     receptor.sort()
#     return receptor


# def get_data_for_CCC_visulization(adata, celltype1, celltype2, ligand, receptor, celltype_col='celltype'):
#     """
#     Given celltype pairs and lr pairs, return expression data for visualization.
#     """
#     adata_ligand = adata[adata.obs[celltype_col] == celltype1, adata.var.index.str.lower() == ligand.lower()]
#     adata_receptor = adata[adata.obs[celltype_col] == celltype2, adata.var.index.str.lower() == receptor.lower()]
#     return adata_ligand, adata_receptor


# def draw_3d_scatter_plot(adata_ligand, adata_receptor):
#     x1 = [x[0] for x in adata_ligand.obsm['spatial_regis']]
#     y1 = [x[1] for x in adata_ligand.obsm['spatial_regis']]
#     z1 = [x[2] for x in adata_ligand.obsm['spatial_regis']]
#     e1 = [x[0] for x in adata_ligand.X.toarray()]
#     mask1 = [True if x > 0 else False for x in e1]
#     x1 = [x for x, y in zip(x1, mask1) if y]
#     y1 = [x for x, y in zip(y1, mask1) if y]
#     z1 = [x for x, y in zip(z1, mask1) if y]
#     trace1 = go.Scatter3d(x=x1, y=y1, z=z1, mode='markers',
#                           marker=dict(size=2, color=e1, colorscale='Reds', cmin=1,
#                                       colorbar=dict(thickness=20, x=1.02, xanchor='right', title=adata_ligand.var.index[0])),
#                           name=adata_ligand.var.index[0])

#     x2 = [x[0] for x in adata_receptor.obsm['spatial_regis']]
#     y2 = [x[1] for x in adata_receptor.obsm['spatial_regis']]
#     z2 = [x[2] for x in adata_receptor.obsm['spatial_regis']]
#     e2 = [x[0] for x in adata_receptor.X.toarray()]
#     mask2 = [True if x > 0 else False for x in e2]
#     x2 = [x for x, y in zip(x2, mask2) if y]
#     y2 = [x for x, y in zip(y2, mask2) if y]
#     z2 = [x for x, y in zip(z2, mask2) if y]
#     trace2 = go.Scatter3d(x=x2, y=y2, z=z2, mode='markers',
#                           marker=dict(size=2, color=e2, colorscale='Blues', cmin=1,
#                                       colorbar=dict(thickness=20, x=1.08, xanchor='right', title=adata_receptor.var.index[0])),
#                           name=adata_receptor.var.index[0])
#     # layout = go.Layout(title='3D Scatter plot')
#     fig = go.Figure(data=[trace1, trace2])
#     fig.update_layout(showlegend=False)
#     fig.show()


# if __name__ == "__main__":
#     significant_mean = preprocess_significant_mean_result(
#         r'E:\Stereopy\论文写作\分析结果\心肌细胞-mesh内部-v1\significant_means_statistical_liana_v1.csv')

#     # mesh = {'heart': r'E:\VT3D\Heart.obj'}
#     adata = anndata.read_h5ad(r'E:\VT3D\inside_mesh_68.anno_v1.h5ad')
#     # sc.pp.normalize_total(adata)
#     adata = preprocess_adata(adata, significant_mean)
#     adata.write(r'E:\VT3D\inside_mesh_68.anno_v1.uns_added.h5ad', compression='gzip')
#     adata_ligand, adata_receptor = get_data_for_CCC_visulization(adata, 'endocardial/endothelial (EC)',
#                                                                  'ventricular-specific CM', 'Igf2', 'Igf2r')
#     draw_3d_scatter_plot(adata_ligand, adata_receptor)

#     slices = list(set(adata.obs['slice']))
#     slices.sort()
#     stats = pd.DataFrame()
#     for s in slices:
#         temp_data = adata[adata.obs['slice'] == s]
#         temp = temp_data.obs.groupby(['slice', 'celltype']).size().reset_index(name='number_cells')
#         temp = temp.pivot(index='slice', columns='celltype', values='number_cells').reset_index()
#         temp.columns.name = None
#         stats = pd.concat([stats, temp], axis=0, ignore_index=True)

