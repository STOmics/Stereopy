import numpy as np
import pandas as pd

try:
    import rpy2
except ImportError:
    rpy2 = None
if rpy2:
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import r
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()
    stats = importr("stats")
    base = importr("base")


from scipy import stats as scipystats


def bw_SJr(y, bw_adjust=3):
    if rpy2 is None:
        raise ImportError("bw_SJr requires rpy2 which is not installed.")
    rpy2.robjects.numpy2ri.activate()
    stats = importr("stats")
    base = importr("base")
    return np.asarray(stats.bw_SJ(y)) * bw_adjust


def ksmooth(genes_log_gmean, genes_log_gmean_step1, col_to_smooth, bw):
    if rpy2 is None:
        raise ImportError("bw_SJr requires rpy2 which is not installed.")
    rpy2.robjects.numpy2ri.activate()
    stats = importr("stats")
    base = importr("base")
    x_points = base.pmax(genes_log_gmean, base.min(genes_log_gmean_step1))
    x_points = base.pmin(x_points, base.max(genes_log_gmean_step1))
    o = base.order(x_points)
    dispersion_par = stats.ksmooth(
        x=genes_log_gmean_step1,
        y=col_to_smooth,
        x_points=x_points,
        bandwidth=bw,
        kernel="normal",
    )
    dispersion_par = dispersion_par[dispersion_par.names.index("y")]
    return {"smoothed": dispersion_par, "order": o}


def robust_scale(x):
    if rpy2 is None:
        raise ImportError("bw_SJr requires rpy2 which is not installed.")
    rpy2.robjects.numpy2ri.activate()
    stats = importr("stats")
    base = importr("base")
    return (x - np.median(x)) / (
        scipystats.median_absolute_deviation(x) + np.finfo(float).eps
    )


def robust_scale_binned_r(y, x, breaks):
    if rpy2 is None:
        raise ImportError("bw_SJr requires rpy2 which is not installed.")
    rpy2.robjects.numpy2ri.activate()
    stats = importr("stats")
    base = importr("base")
    bins = base.cut(x=x, breaks=breaks, ordered_result=True)
    df = pd.DataFrame({"x": y, "bins": bins})
    tmp = df.groupby(["bins"]).apply(robust_scale)
    # tmp = base.aggregate(x = y, by = ro.ListVector({"bin": bins}), FUN = robust_scale)
    ##score = np.asarray([0] * len(x))
    ##o = base.order(bins)
    # if (inherits(x = tmp$x, what = 'list')) {
    #        score[o] <- unlist(tmp$x)
    # } else {
    #    score[o] <- as.numeric(t(tmp$x))
    # }
    ##score[o] = tmp["x"]
    order = df["bins"].argsort()
    tmp = tmp.loc[order]  # sort_values(by=["bins"])
    score = np.asarray(tmp["x"].values)
    return score


def is_outlier_r(y, x, th=10):
    if rpy2 is None:
        raise ImportError("bw_SJr requires rpy2 which is not installed.")
    bin_width = (np.max(x) - np.min(x)) * bw_SJr(x, bw_adjust=1 / 2)
    eps = np.finfo(float).eps * 10
    breaks1 = base.seq(base.min(x) - eps, base.max(x) + bin_width, by=bin_width)
    breaks2 = base.seq(
        base.min(x) - eps - bin_width / 2, base.max(x) + bin_width, by=bin_width
    )
    score1 = robust_scale_binned_r(y, x, breaks1)
    score2 = robust_scale_binned_r(y, x, breaks2)
    return base.pmin(base.abs(score1), base.abs(score2)) > th
