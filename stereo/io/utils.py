import numpy as np
from scipy.sparse import csr_matrix

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