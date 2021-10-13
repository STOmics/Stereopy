#!/usr/bin/env python3
# coding: utf-8
"""
@file: hotspot.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/8/19 create file.
"""
import copy

import numpy as np
import pandas as pd
import hotspot
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import pickle
from ..preprocess.filter import filter_genes


def get_hotspot(data,model='normal',n_neighbors=30,n_jobs=20):
    # Load the counts and positions
    #     counts_file = '/Users/wuyiran55/Documents/jupyter/test/slideseq_data/MappedDGEForR.csv'
    #     pos_file = '/Users/wuyiran55/Documents/jupyter/test/slideseq_data/BeadLocationsForR.csv'
    #
    #     pos = pd.read_csv(pos_file, index_col=0)
    #     counts = pd.read_csv(counts_file, index_col=0) # Takes a while, ~10min
    #
    #     # Align the indices
    #     pos = pos.loc[counts.columns, :]
    #     counts = counts.loc[:, pos.index]
    #     barcodes = pos.index.values
    #
    #     # Swap position axes
    #     # We swap x'=y and y'=-x to match the slides in the paper
    #     pos = pd.DataFrame(
    #         {
    #             'X': pos.ycoord,
    #             'Y': pos.xcoord*-1,
    #         }, index=pos.index
    #     )
    #
    #     num_umi = counts.sum(axis=0)
    #
    hit_data = copy.deepcopy(data)
    num_umi = hit_data.cells.total_counts
    if min_cell:
        hit_data = filter_genes(hit_data, min_cell=min_cell)
    counts = hit_data.to_df()
    pos = hit_data.position

    # Create the Hotspot object and the neighborhood graph
    hs = hotspot.Hotspot(counts, model=model, latent=pos, umi_counts=num_umi)

    hs.create_knn_graph(
        weighted_graph=False, n_neighbors=n_neighbors,
    )

    hs_results = hs.compute_autocorrelations(jobs=n_jobs)

    with open(HS_RESULTS, "wb") as f:
        pickle.dump(hs_results,f)

    # select the genes with significant spatial autocorrelation
    hs_genes = hs_results.index[hs_results.FDR < 0.05]
    # Compute pair-wise local correlations between these genes
    lcz = hs.compute_local_correlations(hs_genes, jobs=20)
    with open(LCZ, "wb") as f:
        pickle.dump(lcz, f)

    modules = hs.create_modules(
        min_gene_threshold=20, core_only=False, fdr_threshold=0.05
    )
    hs.plot_local_correlations()
    with open(MODULES, "wb") as f:
        pickle.dump(modules, f)

    with open(HOTSPOT, "wb") as f:
        pickle.dump(hs, f)


def plot():
    plt.rcParams['figure.figsize'] = (15.0, 12.0)

    hs.plot_local_correlations()

    plt.savefig(''.join([outdir, "/", NAME, "_module_number.png"]), dpi=600)
    plt.close()

    results = hs.results.join(hs.modules)
    results.to_csv(''.join([outdir, "/", NAME, "_Cluster.csv"]))

    module_scores = hs.calculate_module_scores()
    module_scores.to_csv(''.join([outdir, "/", NAME, "_ModuleScore.csv"]))

    if not os.path.exists(f'{outdir}/ModuleFig'):
        os.makedirs(f'{outdir}/ModuleFig')
    for module in range(1, hs.modules.max() + 1):
        scores = hs.module_scores[module]

        vmin = np.percentile(scores, 1)
        vmax = np.percentile(scores, 99)

        plt.scatter(x=hs.latent.iloc[:, 0],
                    y=hs.latent.iloc[:, 1],
                    s=8,
                    c=scores,
                    vmin=vmin,
                    vmax=vmax,
                    edgecolors='none'
                    )
        axes = plt.gca()
        for sp in axes.spines.values():
            sp.set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.title('Module {}'.format(module))
        plt.savefig(f'{outdir}/ModuleFig/Module{module}.png')
        plt.close()

def show(hs):
    module = 1

    results = hs.results.join(hs.modules)
    results = results.loc[results.Module == module]
    results.sort_values('Z', ascending=False).head(10)

# Plot the module scores on top of positions

    module = 1

    results = hs.results.join(hs.modules)
    results = results.loc[results.Module == module]
    genes = results.sort_values('Z', ascending=False).head(6).index

    fig, axs = plt.subplots(2, 3, figsize=(11, 7.5))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'grays', ['#DDDDDD', '#000000'])

    for ax, gene in zip(axs.ravel(), genes):

        expression = np.log2(hs.counts.loc[gene]/hs.umi_counts*45 + 1) # log-counts per 45 (median UMI/barcode)

        vmin = 0
        vmax = np.percentile(expression, 95)
        vmax = 2

        plt.sca(ax)
        plt.scatter(x=hs.latent.iloc[:, 0],
                    y=hs.latent.iloc[:, 1],
                    s=2,
                    c=expression,
                    vmin=vmin,
                    vmax=vmax,
                    edgecolors='none',
                    cmap=cmap
                   )

        for sp in ax.spines.values():
            sp.set_visible(False)

        plt.xticks([])
        plt.yticks([])
        plt.title(gene)

def summary(hs):
    module_scores = hs.calculate_module_scores()
    # Plot the module scores on top of positions

    fig, axs = plt.subplots(2, 3, figsize=(11, 7.5))

    for ax, module in zip(axs.ravel(), range(1, hs.modules.max() + 1)):
        scores = hs.module_scores[module]

        vmin = np.percentile(scores, 1)
        vmax = np.percentile(scores, 99)

        plt.sca(ax)
        plt.scatter(x=hs.latent.iloc[:, 0],
                    y=hs.latent.iloc[:, 1],
                    s=2,
                    c=scores,
                    vmin=vmin,
                    vmax=vmax,
                    edgecolors='none'
                    )

        for sp in ax.spines.values():
            sp.set_visible(False)

        plt.xticks([])
        plt.yticks([])
        plt.title('Module {}'.format(module))