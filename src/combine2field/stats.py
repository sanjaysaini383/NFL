"""Small, dependency-light statistics helpers.

With roughly 100 players per position and many candidate metrics, the analysis
leans on rank statistics, bootstrap intervals, false-discovery control and
out-of-fold prediction rather than in-sample fits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _clean(*arrays):
    a = np.column_stack([np.asarray(x, float) for x in arrays])
    a = a[np.isfinite(a).all(axis=1)]
    return [a[:, i] for i in range(a.shape[1])]


def spearman_ci(x, y, n_boot: int = 1000, seed: int = 0) -> dict:
    x, y = _clean(x, y)
    n = x.size
    if n < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": n, "rho": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    boots = []
    for i in idx:
        a, b = rx[i], ry[i]
        if a.std() > 0 and b.std() > 0:
            boots.append(np.corrcoef(a, b)[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"n": n, "rho": rho, "lo": lo, "hi": hi, "p": p}


def partial_spearman(x, y, z) -> dict:
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    x, y, z = _clean(x, y, z)
    n = x.size
    if n < 10:
        return {"prho": np.nan, "pp": np.nan}
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones(n), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return {"prho": np.nan, "pp": np.nan}
    r = np.corrcoef(ex, ey)[0, 1]
    df = n - 3
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    return {"prho": r, "pp": 2 * stats.t.sf(abs(t), df)}


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs preserved)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    m = pv.size
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def icc_3_1(t1, t2) -> float:
    """ICC(3,1): two-way mixed, consistency, single measurement (Shrout & Fleiss)."""
    t1, t2 = _clean(t1, t2)
    n = t1.size
    if n < 5:
        return np.nan
    Y = np.column_stack([t1, t2])
    k = 2
    grand = Y.mean()
    ss_rows = k * ((Y.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((Y.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((Y - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err
    return float((ms_rows - ms_err) / denom) if denom > 0 else np.nan


def cv_r2(df: pd.DataFrame, y: str, features: list[str], n_repeats: int = 20, seed: int = 0) -> np.ndarray:
    """Out-of-fold R^2 on the rank-transformed outcome, one value per CV repeat."""
    d = df[df[y].notna()]
    feats = [f for f in features if f in d and d[f].notna().any()]
    if len(d) < 25 or not feats:
        return np.array([np.nan])
    X = d[feats].to_numpy(float)
    target = stats.rankdata(d[y].to_numpy(float)) / len(d)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 3, 20)))
    scores = []
    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    pred = np.zeros_like(target)
    for i, (tr, te) in enumerate(rkf.split(X)):
        model.fit(X[tr], target[tr])
        pred[te] = model.predict(X[te])
        if (i + 1) % 5 == 0:
            ss_res = ((target - pred) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            scores.append(1 - ss_res / ss_tot)
    return np.array(scores)
