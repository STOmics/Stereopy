import numba
import numpy as np


def ksmooth(x, y, xp, skrn, sbw):
    xp.sort()
    krn = skrn
    bw = sbw
    x = x
    y = y
    nx, nxp = len(x), len(xp)
    yp = np.zeros(nxp, dtype=float)
    o = np.array(sorted(range(0, len(x)), key=lambda i: x[i]))
    return bdr_ksmooth(x.iloc[o,].to_numpy(), y.iloc[o,].to_numpy(), nx, xp, yp, nxp, krn, bw)


@numba.jit(nopython=True)
def bdr_ksmooth(x, y, n, xp, yp, nxp, kern, bw):
    imin = 0
    cutoff = 0.0

    if kern == 1:
        bw *= 0.5
        cutoff = bw
    if kern == 2:
        bw *= 0.3706506
        cutoff = 4 * bw

    while x[imin] < xp[0] - cutoff and imin < n:
        imin += 1

    def dokern(x1, kern1):
        if kern1 == 1:
            return 1.0
        if kern1 == 2:
            return np.exp(-0.5 * x1 * x1)
        return 0.0

    for j in range(nxp):
        num = den = 0.0
        x0 = xp[j]
        for i in range(imin, n):
            if x[i] < x0 - cutoff:
                imin = i
            else:
                if x[i] > x0 + cutoff:
                    break
                w = dokern(np.fabs(x[i] - x0) / bw, kern)
                num += w * y[i]
                den += w
            i += 1
        if den > 0:
            yp[j] = num / den
        else:
            yp[j] = 0
    return xp, yp

# @numba.jit(cache=True, forceobj=True, parallel=True, nogil=True)
# def dokern(x, kern):
#     if kern == 1:
#         return 1.0
#     if kern == 2:
#         return np.exp(-0.5 * x * x)
#     return 0.0

# @numba.jit(cache=True, forceobj=True, parallel=True, nogil=True)
# def BDRksmooth_test(x, y, n, xp, yp, nxp, kern, bw):
#     # start = time.time()
#     imin = 0
#     cutoff = 0.0
#
#     if kern == 1:
#         bw *= 0.5
#         cutoff = bw
#     if kern == 2:
#         bw *= 0.3706506
#         cutoff = 4 * bw
#
#     while x[imin] < xp[0] - cutoff and imin < n:
#         imin += 1
#
#     for j in range(nxp):
#         print(j, nxp, f'{time.time() - start}') if j % 1000 == 0 else ""
#         x0 = xp[j]
#         num, den, imin = test(x0, n, imin, x, cutoff, bw, kern, y)
#         if (den > 0):
#             yp[j] = num / den
#         else:
#             yp[j] = 0
#     # print(f'end cost {time.time() - start}')
#     return xp, yp


# @numba.jit(cache=True, forceobj=True)
# def test(x0, n, imin, x, cutoff, bw, kern, y):
#     num = den = 0.0
#     for i in range(imin, n):
#         if x[i] < x0 - cutoff:
#             imin = i
#         else:
#             if x[i] > x0 + cutoff:
#                 break
#             w = dokern(np.fabs(x[i] - x0) / bw, kern)
#             num += w * y[i]
#             den += w
#         i += 1
#     return num, den, imin


# from joblib import Parallel
# from joblib import delayed
# def BDRksmooth_test(x, y, n, xp, yp, nxp, kern, bw):
#     imin = 0
#     cutoff = 0.0
#
#     if kern == 1:
#         bw *= 0.5
#         cutoff = bw
#     if kern == 2:
#         bw *= 0.3706506
#         cutoff = 4 * bw
#
#     while x[imin] < xp[0] - cutoff and imin < n:
#         imin += 1
#     res = Parallel(n_jobs=8, verbose=100)(
#         delayed(test)(xp[j], n, imin, x, cutoff, bw, kern, y)
#         for j in range(nxp)
#     )
#     yp = np.array(res)
#     return xp, yp
#
# import numba
#
# @numba.jit(cache=True, forceobj=True)
# def test(x0, n, imin, x, cutoff, bw, kern, y):
#     num = den = 0.0
#     for i in range(imin, n):
#         if x[i] < x0 - cutoff:
#             imin = i
#         else:
#             if x[i] > x0 + cutoff:
#                 break
#             w = dokern(np.fabs(x[i] - x0) / bw, kern)
#             num += w * y[i]
#             den += w
#         i += 1
#     return (num / den) if (den > 0) else 0
