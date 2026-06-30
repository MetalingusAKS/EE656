

import numpy as np
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import pdist



def normalize(X):
    x_min, x_max = X.min(0), X.max(0)
    return (X - x_min) / np.where(x_max - x_min == 0, 1, x_max - x_min)



def run_imc_pass(X_norm, nk, beta):
    n, D = X_norm.shape
    u = X_norm.copy().astype(float)
    U = u.copy()
    m = np.zeros(nk, dtype=int)
    t = np.zeros(nk + 1, dtype=int); t[0] = n
    sl = np.zeros((n, nk + 1), dtype=int); sl[:, 0] = np.arange(n)
    cc_norm = np.zeros((nk, D))
    d1 = np.zeros(nk)

    for v in range(nk):
        if t[v] == 0:
            break
        d = sum(
            X_norm[sl[j, v]].min() / max(X_norm[sl[j, v]].sum(), 1e-10)
            for j in range(t[v])
        )
        d1[v] = max((1.0 / (2 * t[v])) * d * beta[v], 1e-10)
        ur = u[sl[:t[v], v]]
        diff = ur[:, None, :] - ur[None, :, :]
        P = np.exp(-(diff ** 2).sum(-1) / d1[v] ** 2).sum(1)
        cc_norm[v] = ur[P.argmax()]
        nxt_sl, nxt_u = [], []
        for r in range(t[v]):
            if ((ur[r] - cc_norm[v]) ** 2).sum() <= d1[v]:
                m[v] += 1
            else:
                nxt_sl.append(sl[r, v])
                nxt_u.append(ur[r])
        t[v + 1] = len(nxt_sl)
        if nxt_sl:
            sl[:t[v + 1], v + 1] = nxt_sl
            u[:t[v + 1]] = np.array(nxt_u)

    # assign any remaining unassigned points to nearest centre
    if t[nk] > 0:
        ur = u[:t[nk]]
        asgn = ((ur[:, None, :] - cc_norm[None, :, :]) ** 2).sum(-1).argmin(1)
        for r, v in enumerate(asgn):
            m[v] += 1

    dd = ((U[:, None, :] - cc_norm[None, :, :]) ** 2).sum(-1)
    idx = dd.argmin(1)
    return idx, cc_norm, d1, m



def gsi_fast(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.zeros(max(1, len(uniq))), 0.0
    try:
        samp = silhouette_samples(X, labels)
        Sm = np.array([
            samp[labels == m].mean() if (labels == m).sum() > 0 else 0.
            for m in uniq
        ])
        return Sm, float(Sm.mean())
    except Exception:
        return np.zeros(len(uniq)), 0.0


def compute_pi(X, labels, centroids):
    nk = len(centroids)
    pi = 0.0
    for m in range(nk):
        Xm = X[labels == m]; Nm = len(Xm)
        if Nm == 0:
            continue
        ssw = ((Xm - centroids[m]) ** 2).sum()
        sep = sum(
            ((centroids[m] - centroids[k]) ** 2).sum()
            for k in range(nk) if k != m
        )
        if sep > 0:
            pi += ssw / (Nm * sep)
    return pi


def compute_si(X, labels, centroids):
    nk = len(centroids)
    inertia = sum(
        ((X[labels == m] - centroids[m]) ** 2).sum()
        for m in range(nk) if (labels == m).sum() > 0
    )
    dists = [
        ((centroids[m] - centroids[k]) ** 2).sum()
        for m in range(nk) for k in range(nk) if m != k
    ]
    min_d = min(dists) if dists else 1.0
    return inertia / (len(X) * min_d) if min_d > 0 else np.nan


def compute_di(X, labels, centroids):
    nk = len(centroids)
    max_diam = max(
        (pdist(X[labels == m]).max() if len(X[labels == m]) >= 2 else 0.)
        for m in range(nk)
    )
    if max_diam == 0:
        max_diam = 1e-10
    min_inter = np.inf
    for m in range(nk):
        for k in range(m + 1, nk):
            Xm, Xk = X[labels == m], X[labels == k]
            if len(Xm) == 0 or len(Xk) == 0:
                continue
            d_mk = (
                np.linalg.norm(Xm - centroids[k], axis=1).sum() +
                np.linalg.norm(Xk - centroids[m], axis=1).sum()
            ) / (len(Xm) + len(Xk))
            min_inter = min(min_inter, d_mk)
    return min_inter / max_diam if min_inter < np.inf else np.nan


def all_indices(X, labels, centroids):
    _, GSI = gsi_fast(X, labels)
    return (
        GSI,
        compute_pi(X, labels, centroids),
        compute_si(X, labels, centroids),
        compute_di(X, labels, centroids),
    )



def lagrange_find_eta(delta_list, Sm_list):

    min_len = min(len(delta_list), len(Sm_list))
    delta_list = delta_list[:min_len]
    Sm_list = Sm_list[:min_len]
    deltas = np.array(delta_list, dtype=float)
    Ss = np.array(Sm_list, dtype=float)
    M = len(deltas)
    if M < 2:
        return deltas[0] if M > 0 else 0.01

    def leval(dt):
        r = 0.
        for m in range(M):
            lm = 1.
            for k in range(M):
                if k != m and abs(deltas[m] - deltas[k]) > 1e-15:
                    lm *= (dt - deltas[k]) / (deltas[m] - deltas[k])
            r += Ss[m] * lm
        return r

    grid = np.linspace(1e-6, 0.1666, 3000)
    vals = np.array([leval(d) for d in grid])
    diff = vals - 1.
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return grid[np.argmin(np.abs(diff))]

    best_eta, best_err = grid[sc[0]], np.inf
    for i in sc:
        d1, d2 = grid[i], grid[i + 1]
        f1, f2 = diff[i], diff[i + 1]
        if abs(f2 - f1) > 1e-15:
            eta = d1 - f1 * (d2 - d1) / (f2 - f1)
            err = abs(leval(eta) - 1.)
            if err < best_err:
                best_err, best_eta = err, eta
    return best_eta


def factorcal(X_norm, nk, max_iter=10):
    beta = np.ones(nk)
    best_beta, best_gsi = beta.copy(), -np.inf
    history = []

    for it in range(1, max_iter + 1):
        idx, cc, d1, m = run_imc_pass(X_norm, nk, beta)
        if len(np.unique(idx)) < 2:
            break
        Sm, GSI = gsi_fast(X_norm, idx)
        history.append({'iter': it, 'GSI': GSI, 'delta': d1.copy(),
                        'Sm': list(Sm), 'eta': None})
        if GSI > best_gsi:
            best_gsi, best_beta = GSI, beta.copy()
        if len(set(d1.round(8))) < nk:
            break
        eta = lagrange_find_eta(list(d1), list(Sm))
        history[-1]['eta'] = eta
        beta = np.array([eta / max(d, 1e-10) for d in d1])

    return best_beta, history



def soc_cluster(X, nk):
    X_norm = normalize(X)
    best_beta, history = factorcal(X_norm, nk)
    idx, cc, d1, m = run_imc_pass(X_norm, nk, best_beta)
    cc_raw = cc * (X.max(0) - X.min(0) + 1e-10) + X.min(0)
    return idx, cc_raw, history
