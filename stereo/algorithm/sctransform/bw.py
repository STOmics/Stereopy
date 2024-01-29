import numpy as np
from scipy import optimize

PI = 3.14159265
DELTA_MAX = 1000


def bwSJ(x, nb=1000):
    n = len(x)
    Z = bw_pair_cnts(x, nb, n > nb / 2)

    d = Z[0]
    cnt = Z[1]

    def SDh(h):
        return bw_phi4(n, d, cnt, h)

    def TDh(h):
        return bw_phi6(n, d, cnt, h)

    q75, q25 = np.percentile(x, [75, 25])
    x_iqr = q75 - q25
    scale = min(np.std(x, ddof=1), x_iqr / 1.349)

    n = len(x)
    a = 1.24 * scale * np.power(n, -1 / 7)
    b = 1.23 * scale * np.power(n, -1 / 9)
    c1 = 1 / (2 * np.sqrt(PI) * n)

    TD = -TDh(b)
    if np.isinf(TD) or TD <= 0:
        raise Exception

    hmax = 1.144 * scale * np.power(n, -1 / 5)
    lower = 0.1 * hmax
    upper = hmax

    alph2 = 1.357 * np.power(SDh(a) / TD, 1 / 7)

    def fSD(h):
        return np.power(c1 / SDh(alph2 * np.power(h, 5 / 7)), 1 / 5) - h

    itry = 1
    while fSD(lower) * fSD(upper) > 0:
        if itry >= 99:
            raise Exception
        if itry % 2:
            upper = upper * 1.2
        else:
            lower = lower / 1.2
        itry += 1

    return uniroot(fSD, (lower, upper), 0.1 * lower)


def uniroot(fun, limit, tol):
    return optimize.brentq(fun, limit[0], limit[1], xtol=tol)


def bw_pair_cnts(x, nb, binned: bool):
    if binned:
        r = (np.min(x), np.max(x))
        d = np.diff(r) * 1.01 / nb
        xx = np.trunc(np.abs(x) / d) * np.sign(x)
        xx = xx - np.min(xx) + 1
        xx = xx.astype(dtype=np.int64)
        xxx = np.bincount(xx, minlength=nb + 1)[1:]
        return d, bw_den_binned(xxx)
    else:
        return bw_den(nb, x)


def bw_den(nb, sx):
    n = len(sx)
    sc = np.array([0] * nb)
    x = sx
    cnt = sc
    for i in range(nb):
        cnt[i] = 0
    xmin = xmax = x[0]
    for i in range(n):
        xmin = min(xmin, x[i])
        xmax = max(xmax, x[i])
    rang = (xmax - xmin) * 1.01
    dd = rang / nb
    for i in range(n):
        ii = int(x[i] / dd)
        for j in range(0, i):
            jj = int(x[j] / dd)
            cnt[abs(ii - jj)] += 1
    return dd, sc


def bw_den_binned(sx):
    nb = len(sx)
    cnt = np.array([0] * nb, dtype=float)
    ii = 0
    while ii < nb:
        w = sx[ii]
        cnt[0] += w * (w - 1.)
        jj = 0
        while jj < ii:
            cnt[ii - jj] += w * sx[jj]
            jj += 1
        ii += 1
    cnt[0] *= 0.5
    return cnt


def bw_phi4(sn, sd, cnt, sh) -> float:
    h = sh
    d = sd
    _sum = 0.0
    n = sn
    nbin = len(cnt)
    x = cnt
    for i in range(nbin):
        delta = i * d / h
        delta *= delta
        if delta >= DELTA_MAX:
            break
        term = np.exp(-delta / 2) * (delta * delta - 6 * delta + 3)
        _sum += term * x[i]
    _sum = 2 * _sum + n * 3
    return _sum / (n * (n - 1) * np.power(h, 5.0) * np.sqrt(2 * PI))


def bw_phi6(sn, sd, cnt, sh) -> float:
    h = sh
    d = sd
    _sum = 0.0
    n = sn
    nbin = len(cnt)
    x = cnt
    for i in range(nbin):
        delta = i * d / h
        delta *= delta
        if delta >= DELTA_MAX:
            break
        term = np.exp(-delta / 2) * (delta * delta * delta - 15 * delta * delta + 45 * delta - 15)
        _sum += term * x[i]
    _sum = 2 * _sum - 15 * n
    return _sum / (n * (n - 1) * np.power(h, 7.0) * np.sqrt(2 * PI))


if __name__ == "__main__":
    '''
    function `bw_pair_cnts`'s result from R using example-1
    [[1]]
    [1] 0.0004203446

    [[2]]
       [1] 385   0   0   0   0   0   0   0   0   0   0   0   0
      [14]   0   0   0   0   0   0   0   0   0   0   0   0   0
      [27]   0   0   0   0   0   0   0   0   0   0   0   0   0
      [40]   0   0   0   0   0   0   0   0   0   0   0   0   0
      [53]   0   0   1   3   0   0   0   0   0   0   0   9   0
      [66]   0   0   0   0   0   0   0   0   0  17   0   0   0
      [79]   0   0   0   0   0   9   0   0   0   0   0   0   0
      [92]   0   0   0   0   0   0  17   0   5   0   0   0   0
     [105]   0   0   0   0   0  25   0   0   0   0   0  20   0
     [118]   0   0   0   1   0  15   0   0   0   0   0   0   0
     [131]   0   0   0   0   0   0   0   0  27   0   0   0   0
     [144]   0   0   0   1   0   0   0   0   0   0   5   0   0
     [157]   0   0   0   0 153   0   0   0   0   0   0   0   0
     [170]   0   0   1   0   0   1   0   0   5   0   0   0   0
     [183]   0   0   0   0   0   0   0 340   0   0   0   0   0
     [196]   0   0   0   0   0   0   3   0   0   0   0   0   0
     [209]   5   0   0   0   0   0   0   0   0   0   0   0   0
     [222]   0   0   0   0   0   0   0   0   0   0  15   0   0
     [235]   9   0   0   0   0   0   0   0   0  17   0   0   0
     [248]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [261]  45   0   5   0   0   0   0   0   0   0   0   0   0
     [274]   5   0   0   0   0   0   0   0   0   0   0   0   0
     [287]  25   0   0   0   0   0   0   0   0   0   0   0  51
     [300]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [313]   0   0   0   0   0   1   0   0   0   0   0   5   0
     [326]   0   0   0   0   0   3   0   0   0   0   0   0   0
     [339]   0   0   0   0   0   0   0   0   0   0   0 180   0
     [352]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [365]   0   0   0   0   0  45   0   0   3   0   0   0   0
     [378]   0   0   0   0   0   5   0   3   1   0   0   0   0
     [391]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [404]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [417]   0   0   0   0  85   0   0   0   0   0   0   0   0
     [430]   0   0   0  25   0   0   0   0   0   0   1   0   0
     [443]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [456]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [469]   9   0   0   0   0   0   0   0   0   0   0   0   0
     [482]   0   0   0   0   0   0  60   0   0   0   0   0   0
     [495]   5   0   0   0   0   0   0   0   0   0   3   0   0
     [508]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [521]   0   0   9   0   0   0   0   0   0  85   0   1   0
     [534]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [547]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [560]   1   0   0   0   0   0   0   0   0   0   0   0   0
     [573]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [586]   1   0   0   0   0   0   0   0   0   0   0   0   0
     [599]   0   0   0   0   0   5   0   0   0   0   0 100   0
     [612]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [625]   0   0   0   0  17   0   0   0   0   0   0   0   0
     [638]   0   0   0   0   0   9   0   0   0   0   0   0   0
     [651]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [664]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [677]   0   0   0   0   0   0  17   0   0   0   0   0   0
     [690]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [703]   1   0   0   1   0   0   0   0   0   0   0   0   0
     [716]   0   0   0 100   0   0   0   0   0   0   0   0   0
     [729]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [742]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [755]   0   0   1   0   0   0   0   0   0   0   0   0   0
     [768]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [781]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [794]   0   0   0   0   0   0   0   0   0  17   0   0   0
     [807]   0   0   0   0   0   0   0   0   0   0   0  20   0
     [820]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [833]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [846]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [859]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [872]  20   0   0   0   0   1   0   0   0   0   0   0   0
     [885]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [898]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [911]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [924]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [937]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [950]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [963]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [976]   0   0   0   0   0   0   0   0   0   0   0   0   0
     [989]   0   0   0  20   0   0   0   0   0   0   0   0
    '''
    # example-1
    print(bwSJ(np.array([-2.3801186, -2.3801186, -2.1576715, -2.3801186, -2.3801186,
                         -2.3801186, -2.3801186, -2.4594538, -2.3130193, -2.2780812,
                         -2.3801186, -2.4594538, -2.203572, -2.4594538, -2.4594538,
                         -2.4594538, -2.3801186, -2.3130193, -2.4594538, -2.4594538,
                         -2.3801186, -2.3801186, -2.1576715, -2.3801186, -2.4594538,
                         -2.3396304, -2.203572, -2.4594538, -2.203572, -2.3130193,
                         -2.4594538, -2.2548757, -2.3801186, -2.1576715, -2.1161213,
                         -2.3130193, -2.3801186, -2.1576715, -2.4594538, -2.4594538,
                         -2.4594538, -2.0935314, -2.3801186, -2.4594538, -2.3130193,
                         -2.4594538, -2.203572, -2.043271, -2.1576715, -2.3801186,
                         -2.3801186, -2.4594538, -2.3130193, -2.3801186, -2.4594538,
                         -2.4594538, -2.3130193, -2.4594538, -2.203572, -2.3130193,
                         -2.4113126, -2.4594538, -2.3130193, -2.2548757, -2.2548757])))
    print(bwSJ(np.array([1.0, 2.0, 3.0])))
