import numpy as np


def fit_glmgp(y, coldata, design="~ log10_umi"):
    import rpy2
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects import Formula
    from rpy2.robjects import IntVector
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    rpy2.robjects.numpy2ri.activate()
    glmgp = importr("glmGamPoi")
    y_ro = np.asmatrix(y)
    # design_matrix_ro = np.asarray(design_matrix)
    fit = glmgp.glm_gp(
        data=y_ro, design=Formula(design), col_data=coldata, size_factors=False
    )
    overdispersions = fit[fit.names.index("overdispersions")]
    mu = fit[fit.names.index("Mu")]
    beta = fit[fit.names.index("Beta")][0]
    return {
        "theta": np.vstack((1 / overdispersions[0], np.mean(mu, axis=1) / 1e-4)).min(
            axis=0
        )[0],
        "Intercept": beta[0],
        "log10_umi": beta[1],
    }


def fit_glmgp_offset(y, coldata, log_umi, design="~ 1"):
    import rpy2
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects import Formula
    from rpy2.robjects import IntVector
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    rpy2.robjects.numpy2ri.activate()
    glmgp = importr("glmGamPoi")
    y_ro = np.asmatrix(y)
    # design_matrix_ro = np.asarray(design_matrix)
    fit = glmgp.glm_gp(
        data=y_ro,
        design=Formula(design),
        col_data=coldata,
        size_factors=False,
        offset=log_umi,
    )
    overdispersions = fit[fit.names.index("overdispersions")]
    mu = fit[fit.names.index("Mu")]
    beta = fit[fit.names.index("Beta")][0]
    return {
        "theta": np.vstack((1 / overdispersions[0], np.mean(mu, axis=1) / 1e-4)).min(
            axis=0
        )[0],
        "Intercept": beta[0],
        "log10_umi": np.log(10),
    }
