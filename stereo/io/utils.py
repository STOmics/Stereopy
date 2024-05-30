from natsort import natsorted
import numpy as np
from scipy.sparse import csr_matrix
import gzip

def remove_genes_number(gene_names):
    underline = np.char.find(gene_names, '_', start=1, end=-1)
    underline = (underline > 0)
    if not np.any(underline):
        return gene_names
    gene_idx = np.arange(gene_names.size, dtype=np.uint64)
    gene_names_with_underline = gene_names[underline]
    gene_idx_with_underline = gene_idx[underline]
    for gul_idx, gul in zip(gene_idx_with_underline, gene_names_with_underline):
        tmp = gul.split('_')
        if not tmp[-1].isnumeric():
            continue
        gnul = '_'.join(tmp[0:-1])
        gene_names[gul_idx] = gnul
    return gene_names

def integrate_matrix_by_genes(
        gene_names: np.ndarray,
        cell_num: int,
        mtx_data: np.ndarray,
        mtx_indices: np.ndarray,
        mtx_indptr: np.ndarray
    ):
    gene_unique, gene_sorted_ind, gene_count = np.unique(gene_names, return_inverse=True, return_counts=True)
    gene_sorted_ind_unique = np.unique(gene_sorted_ind)

    gene_gt1 = gene_sorted_ind_unique[gene_count > 1]
    gene_idx = np.arange(gene_names.size, dtype=np.uint64)
    gene_extraction_flag = np.ones(gene_names.size, dtype=bool)
    one_row_data = np.zeros(gene_names.size, dtype=mtx_data.dtype)
    data_extraction_flag = np.ones(one_row_data.size, dtype=bool)
    data = np.zeros_like(mtx_data)
    indices = np.zeros_like(mtx_indices)
    indptr = np.zeros(cell_num + 1, dtype=mtx_indptr.dtype)
    ind_start = 0
    for i in range(cell_num):
        start, end = mtx_indptr[i], mtx_indptr[i+1]
        one_row_data[mtx_indices[start:end]] = mtx_data[start:end]
        for g in gene_gt1:
            gi = gene_idx[gene_sorted_ind == g]
            one_row_data[gi[0]] = one_row_data[gi].sum()
            data_extraction_flag[gi[1:]] = False
            gene_extraction_flag[gi[1:]] = False
        one_row_data_extracted = one_row_data[data_extraction_flag]
        nonezero_idx = np.nonzero(one_row_data_extracted)[0]
        ind_end = ind_start + nonezero_idx.size
        data[ind_start:ind_end] = one_row_data_extracted[nonezero_idx]
        indices[ind_start:ind_end] = nonezero_idx
        indptr[i], indptr[i+1] = ind_start, ind_end
        ind_start = ind_end
        one_row_data[:] = 0
        data_extraction_flag[:] = True
    data = data[indptr[0]:indptr[-1]]
    indices = indices[indptr[0]:indptr[-1]]

    gene_names_unique = gene_names[gene_extraction_flag]
    exp_matrix = csr_matrix((data, indices, indptr), shape=(cell_num, gene_names_unique.size))
    return exp_matrix, gene_names_unique

def transform_marker_genes_to_anndata(marker_genes_res: dict):
    marker_genes_result = {}
    method = marker_genes_res['parameters']['method']
    if method == 't_test':
        method = 't-test'
    elif method == 'wilcoxon_test':
        method = 'wilcoxon'
    marker_genes_result['params'] = {
        'groupby': marker_genes_res['parameters']['cluster_res_key'],
        'reference': marker_genes_res['parameters']['control_groups'],
        'method': method,
        'use_raw': marker_genes_res['parameters']['use_raw'],
        'layer': None,
        'corr_method': marker_genes_res['parameters']['corr_method']
    }
    if 'marker_genes_res_key' in marker_genes_res['parameters']:
        marker_genes_result['params']['marker_genes_res_key'] = marker_genes_res['parameters']['marker_genes_res_key']
    marker_genes_result['pts'] = marker_genes_res['pct'].set_index('genes')
    marker_genes_result['pts'].index.name = None
    marker_genes_result['pts_rest'] = marker_genes_res['pct_rest'].set_index('genes')
    marker_genes_result['pts_rest'].index.name = None
    marker_genes_result['mean_count'] = marker_genes_res['mean_count']
    groups_key = natsorted([k for k in  marker_genes_res if '.vs.' in k])
    key_map = {
        'genes': 'names',
        'scores': 'scores', 
        'pvalues': 'pvals',
        'pvalues_adj': 'pvals_adj',
        'log2fc': 'logfoldchanges'
    }
    for k1, k2 in key_map.items():
        dtype = []
        for k in groups_key:
            group = k.split('.vs.')[0]
            dtype.append((group, marker_genes_res[k][k1].dtype))
        recarr = np.recarray(shape=marker_genes_res[k].shape[0], dtype=dtype)
        for k in groups_key:
            group = k.split('.vs.')[0]
            recarr[group] = marker_genes_res[k][k1].to_numpy()
        marker_genes_result[k2] = recarr
    return marker_genes_result

def get_gem_comments(gem_file_path: str):
    if gem_file_path.endswith('.gz'):
        open_func =  gzip.open
    else:
        open_func = open
    
    comments = []
    with open_func(gem_file_path, 'rb') as fp:
        for line in fp:
            if not line.startswith(b'#'):
                break
            comments.append(line)
    return len(comments), comments