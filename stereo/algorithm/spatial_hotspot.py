#!/usr/bin/env python3
# coding: utf-8
"""
@file: get_hotspot.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/10/14 create file.
"""
import copy
import pandas as pd
import hotspot
from ..log_manager import logger


def spatial_hotspot(data, model='normal', n_neighbors=30, n_jobs=20, fdr_threshold=0.05,
                    min_gene_threshold=10, outdir=None):
    """
    identifying informative genes (and gene modules)

    :param data: StereoExpData
    :param model: Specifies the null model to use for gene expression.
        Valid choices are:
            - 'danb': Depth-Adjusted Negative Binomial
            - 'bernoulli': Models probability of detection
            - 'normal': Depth-Adjusted Normal
            - 'none': Assumes data has been pre-standardized
    :param n_neighbors: Neighborhood size used in create_knn_graph.
    :param n_jobs: Number of parallel jobs to run.
    :param fdr_threshold: Correlation threshold at which to stop assigning genes to modules
    :param min_gene_threshold: Minimum number of genes to create a module. Controls how small modules can be. Increase
        if there are too many modules being formed. Decrease if substructre is not being captured
    :param outdir: directory containing output file(hotspot.pkl). Hotspot object will be totally output here.
    If None, results will not be output to a file.

    :return:Hotspot object.

    """

    hit_data = copy.deepcopy(data)
    counts = hit_data.to_df().T  # gene x cell
    pos = pd.DataFrame(hit_data.position, index=counts.columns)  # cell name as index
    num_umi = counts.sum(axis=0)  # total counts per cell
    # Create the Hotspot object and the neighborhood graph
    logger.info(f'create the Hotspot object with {counts.shape[0]} genes and {counts.shape[1]} cells, model={model}.')
    hs = hotspot.Hotspot(counts, model=model, latent=pos, umi_counts=num_umi)
    logger.info(f'create_knn_graph with n_neighbors={n_neighbors}.')
    hs.create_knn_graph(
        weighted_graph=False, n_neighbors=n_neighbors,
    )
    logger.info('Start compute_autocorrelations.')
    hs_results = hs.compute_autocorrelations(jobs=n_jobs)
    # select the genes with significant spatial autocorrelation
    hs_genes = hs_results.index[hs_results.FDR < fdr_threshold]
    logger.info(f'Remain {len(hs_genes)} genes whose FDR < {fdr_threshold}.')
    # Compute pair-wise local correlations between these genes
    logger.info('Start compute_local_correlations.')
    lcz = hs.compute_local_correlations(hs_genes, jobs=n_jobs)
    logger.info(f'Start create_modules with min_gene_threshold={min_gene_threshold}, fdr_threshold={fdr_threshold}.')
    modules = hs.create_modules(
        min_gene_threshold=min_gene_threshold, core_only=False, fdr_threshold=fdr_threshold,
    )
    logger.info('Start calculate_module_scores in per cell.')
    module_scores = hs.calculate_module_scores()
    if module_scores.shape[1] == 0:
        logger.error(f'No modules be created. Please decrease min_gene_threshold.')
    if outdir is not None:
        from stereo.io.writer import save_pkl
        save_pkl(hs, output=f"{outdir}/hotspot.pkl")
    return hs


