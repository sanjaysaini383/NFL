import numpy as np
import pandas as pd
import pytest

from combine2field import combine_features as cf
from combine2field import config, game_profile as gp, io, synthetic
from combine2field.kinematics import grouped_tangential_acc


@pytest.mark.parametrize("s0,tau", [(10.3, 1.0), (8.4, 1.3), (9.2, 1.15)])
def test_sprint_fit_recovers_parameters(s0, tau):
    x, _ = synthetic._sprint(s0, tau, rng=np.random.default_rng(1))
    t = np.arange(x.size) * config.DT
    out = cf.fit_sprint(t, x, np.zeros_like(x))
    assert out["S0"] == pytest.approx(s0, rel=0.03)
    assert out["tau"] == pytest.approx(tau, rel=0.08)
    assert out["A0"] == pytest.approx(s0 / tau, rel=0.08)
    assert out["fit_r2"] > 0.97
    # 40-yard split from tracking should be close to the analytic time
    tt = np.linspace(0, 8, 8001)
    t40 = tt[np.searchsorted(s0 * (tt - tau * (1 - np.exp(-tt / tau))), 40)]
    assert out["split40"] == pytest.approx(t40, abs=0.15)


def test_sprint_fit_rejects_standing_trace():
    t = np.arange(50) * config.DT
    out = cf.fit_sprint(t, np.zeros(50), np.zeros(50))
    assert np.isnan(out["S0"])


def test_cod_features_detect_turns():
    rng = np.random.default_rng(0)
    xy = synthetic._waypoint_run([(0, 0), (0, 5), (0, -5), (0, 0)], 8.5, 9.0, 7.0, rng)
    t = np.arange(len(xy)) * config.DT
    out = cf.cod_features(t, xy[:, 0], xy[:, 1])
    assert out["n_turns"] >= 2
    assert 3 < out["brake_peak"] < 15


def test_as_profile_recovers_line():
    rng = np.random.default_rng(3)
    a0, s0 = 7.0, 9.5
    v = np.repeat(np.arange(3.1, 9.0, 0.2), 2)
    a = a0 * (1 - v / s0) + rng.normal(0, 0.05, v.size)
    out = gp.fit_as_profile(v, a)
    assert out["A0"] == pytest.approx(a0, rel=0.05)
    assert out["S0"] == pytest.approx(s0, rel=0.05)


def test_grouped_acc_is_nan_at_group_edges():
    df = pd.DataFrame({
        "nfl_id": [1, 1, 1, 2, 2, 2],
        "t": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
        "s": [0.0, 1.0, 2.0, 5.0, 5.0, 5.0],
    })
    acc = grouped_tangential_acc(df, ["nfl_id"]).to_numpy()
    assert np.isnan(acc[[0, 2, 3, 5]]).all()
    assert acc[1] == pytest.approx(10.0)
    assert acc[4] == pytest.approx(0.0)


def test_time_parsing_handles_mixed_separators():
    s = pd.Series(["2025-03-01T10:00:00.000000", "2025-03-01 10:00:00.100000"])
    sec = io.time_to_seconds(s)
    assert sec[1] - sec[0] == pytest.approx(0.1, abs=1e-6)


def test_engine_from_splits_round_trip():
    s0, tau = 10.0, 1.1
    tt = np.linspace(0, 8, 80001)
    x = s0 * (tt - tau * (1 - np.exp(-tt / tau)))
    t10, t40 = tt[np.searchsorted(x, 10)], tt[np.searchsorted(x, 40)]
    out = cf.engine_from_splits(t10, t40)
    assert out["S0"] == pytest.approx(s0, rel=0.01)
    assert out["tau"] == pytest.approx(tau, rel=0.02)
    # a realistic WR: 1.50 / 4.40
    real = cf.engine_from_splits(1.50, 4.40)
    assert 9 < real["S0"] < 13 and real["A0"] > 5


def test_sprint_fit_ignores_walk_up_before_stance():
    x, _ = synthetic._sprint(10.0, 1.05, rng=np.random.default_rng(4))
    walk = np.linspace(-6, 0, 40)  # 4 s walking to the line at 1.5 yd/s
    xs = np.concatenate([walk, x])
    t = np.arange(xs.size) * config.DT
    out = cf.fit_sprint(t, xs, np.zeros_like(xs))
    assert out["S0"] == pytest.approx(10.0, rel=0.03)
    assert out["tau"] == pytest.approx(1.05, rel=0.1)


def test_sample_plays_gives_equal_exposure():
    keys = pd.DataFrame({
        "nfl_id": [1] * 50 + [2] * 10,
        "game_id": 1,
        "play_id": list(range(50)) + list(range(10)),
    })
    s = gp.sample_plays(keys, k=20)
    assert s.groupby("nfl_id").size().to_dict() == {1: 20}  # player 2 has too few plays
    assert s.equals(gp.sample_plays(keys, k=20))  # deterministic
