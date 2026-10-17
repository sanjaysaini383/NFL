import numpy as np
import pandas as pd

from combine2field import stats


def test_icc_perfect_and_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert stats.icc_3_1(x, x) > 0.999
    assert abs(stats.icc_3_1(x, rng.normal(size=200))) < 0.2


def test_bh_is_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.03, 0.5, np.nan])
    q = stats.bh_qvalues(p)
    assert np.isnan(q[-1])
    assert (q[:4] >= p[:4]).all() and (q[:4] <= 1).all()
    order = np.argsort(p[:4])
    assert (np.diff(q[:4][order]) >= -1e-12).all()


def test_partial_spearman_removes_confounder():
    rng = np.random.default_rng(1)
    z = rng.normal(size=400)
    x = z + rng.normal(0, 0.3, 400)
    y = z + rng.normal(0, 0.3, 400)
    assert stats.spearman_ci(x, y)["rho"] > 0.7
    assert abs(stats.partial_spearman(x, y, z)["prho"]) < 0.15


def test_cv_r2_detects_signal():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=150), "noise": rng.normal(size=150)})
    df["y"] = df["a"] + rng.normal(0, 0.5, 150)
    assert stats.cv_r2(df, "y", ["a"], n_repeats=3).mean() > 0.5
    assert stats.cv_r2(df, "y", ["noise"], n_repeats=3).mean() < 0.1
