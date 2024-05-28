import pandas as pd
import numpy as np


def cal_proportion_slices_(slicesl, tyli, ctype_col='annotation'):
    def cal_proportion(adata, tyli, ctype_col):
        ty2pro = {}
        for ty in tyli:
            len_all = len(adata)
            pro = len(adata[adata.obs[ctype_col] == ty]) / len_all
            ty2pro[ty] = pro
        return ty2pro

    pro_df = pd.DataFrame(columns=tyli)
    for slice_ in slicesl:
        ty2pro = cal_proportion(slice_, tyli, ctype_col)
        pro_df = pd.concat([pro_df, pd.DataFrame(ty2pro, index=[0])], axis=0, ignore_index=True)
    return pro_df


def kl_div_slicesl_(pro_df):
    def kl_divergence(X, Y):
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)

        X = X / np.sum(X, axis=1)
        Y = Y / np.sum(Y, axis=1)

        log_X = np.log(X+1e-8)
        log_Y = np.log(Y+1e-8)
        X_log_X = np.einsum('ij,ij->i', X, log_X)
        X_log_X = np.reshape(X_log_X, (1, X_log_X.shape[0]))
        D = X_log_X.T - np.dot(X, log_Y.T)
        return D.item()
    kl_div_li = []
    for i in range(len(pro_df)-1):
        kl_d_val = kl_divergence(pro_df.iloc[i].to_numpy(), pro_df.iloc[i+1].to_numpy())
        kl_div_li.append(kl_d_val)
    return np.array(kl_div_li)


def get_tyli_(slicesl, ctype_col):
    tyli = []
    for slice in slicesl:
        tyli += list(dict.fromkeys(slice.obs[ctype_col]))
    tyli = list(dict.fromkeys(tyli))
    return tyli


def calculate_max_kl(slicesl, ctype_col):
    """
    To calculate probabilistic distribution of number of spots on different clusters or annotations for each section,
    then measure Kullback-Leibler (KL) divergence of the distribution between closest section pairs .

    :param slicesl: list of AnnData.adata
    :param ctype_col: cluster or annotation type stored in adata.obs[ctype_col]
    :return: maximum kl divergence
    """
    tyli = get_tyli_(slicesl, ctype_col)
    pro_df = cal_proportion_slices_(slicesl, tyli)
    kl_pairs = kl_div_slicesl_(pro_df)
    return kl_pairs.max()