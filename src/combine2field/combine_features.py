"""The Combine "engine": sprint mechanics and change-of-direction braking.

Sprint: the 40 is fitted with the mono-exponential model used in sprint science
(Furusawa 1927; Samozino et al. 2016),

    v(t) = S0 * (1 - exp(-(t - t0) / tau)),

which implies a linear acceleration-speed relation a(v) = A0 * (1 - v / S0) with
A0 = S0 / tau. The same linear A-S relation is fitted to game tracking in
`game_profile.py`, so the Combine engine and its in-game expression sit on the
same axes.

Change of direction: the 5-10-5 shuttle and three-cone are split into turns at
local speed minima. For each turn we measure braking (peak deceleration on
entry), speed kept through the turn and re-acceleration on exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from . import config
from .kinematics import path_length, speed_from_xy, tangential_acc

ATTEMPT_KEYS = ["nfl_id", "drill_name", "attempt", "event_id"]


def _mono_exp(t, s0, tau, t0):
    return np.where(t > t0, s0 * (1 - np.exp(-(t - t0) / tau)), 0.0)


def fit_sprint(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Fit the sprint model to one 40-yard attempt. Returns NaNs when the trace is unusable."""
    out = {k: np.nan for k in ("S0", "tau", "A0", "Pmax_rel", "split10", "split20", "split40",
                               "vmax_obs", "fit_rmse", "fit_r2", "t0")}
    if t.size < 20:
        return out
    t = t - t[0]
    s = speed_from_xy(t, x, y)
    moving = np.flatnonzero(s > 0.8)
    if moving.size == 0:
        return out
    i0 = max(moving[0] - 2, 0)
    d = path_length(x, y)
    d = d - d[i0]
    # fit window: from the start of motion until 42 yd or peak speed + 0.5 s
    i_peak = i0 + int(np.argmax(s[i0:]))
    end = min(np.searchsorted(d, 42.0), i_peak + int(0.5 * config.HZ)) + 1
    tw, sw = t[i0:end], s[i0:end]
    if tw.size < 15 or s.max() < 4:
        return out
    p0 = [float(sw.max()), 1.1, float(tw[0] + 0.2)]
    try:
        (s0, tau, t0), _ = curve_fit(
            _mono_exp, tw, sw, p0=p0,
            bounds=([3.0, 0.3, tw[0] - 0.5], [13.0, 3.0, tw[0] + 1.5]), maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return out
    pred = _mono_exp(tw, s0, tau, t0)
    rss = float(np.sum((sw - pred) ** 2))
    tss = float(np.sum((sw - sw.mean()) ** 2)) or np.nan
    a0 = s0 / tau
    t_start = t0

    def split(dist):
        j = np.searchsorted(d, dist)
        return float(t[j] - t_start) if j < t.size else np.nan

    out.update({
        "S0": s0, "tau": tau, "A0": a0, "t0": t0,
        "Pmax_rel": a0 * s0 / 4.0,  # peak of a*v on the linear A-S line (yd^2/s^3, ∝ W/kg)
        "split10": split(10), "split20": split(20), "split40": split(40),
        "vmax_obs": float(s[i0:end].max()),
        "fit_rmse": np.sqrt(rss / tw.size), "fit_r2": 1 - rss / tss,
    })
    return out


def engine_from_splits(t10: float, t40: float) -> dict:
    """Recover (S0, tau, A0) from official 10- and 40-yard times.

    Solves x(t) = S0 * (t - tau * (1 - exp(-t / tau))) through both splits, so a
    prospect with only stopwatch data can be placed on the same A-S axes.
    """
    from scipy.optimize import brentq

    def dist(t, s0, tau):
        return s0 * (t - tau * (1 - np.exp(-t / tau)))

    def s0_for(tau):  # S0 that makes the 40 exact for a given tau
        return 40.0 / (t40 - tau * (1 - np.exp(-t40 / tau)))

    def resid(tau):
        return dist(t10, s0_for(tau), tau) - 10.0

    try:
        tau = brentq(resid, 0.2, min(3.0, t40 * 0.6))
    except ValueError:
        return {"S0": np.nan, "tau": np.nan, "A0": np.nan}
    s0 = s0_for(tau)
    return {"S0": s0, "tau": tau, "A0": s0 / tau}


def cod_features(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Braking and turn features for one change-of-direction attempt."""
    out = {k: np.nan for k in ("brake_peak", "brake_mean", "turn_speed_ratio", "reaccel_peak",
                               "drill_time", "cod_vmax", "n_turns")}
    if t.size < 15:
        return out
    t = t - t[0]
    s = speed_from_xy(t, x, y)
    a = tangential_acc(t, s)
    moving = np.flatnonzero(s > 0.8)
    if moving.size < 10:
        return out
    i0, i1 = moving[0], moving[-1]
    peak = s[i0:i1 + 1].max()
    # turns = local speed minima well below the drill's peak speed
    minima, _ = find_peaks(-s[i0:i1 + 1], prominence=0.25 * peak)
    minima = minima + i0
    brakes, ratios, reacc = [], [], []
    for m in minima:
        lo = max(i0, m - int(1.0 * config.HZ))
        hi = min(i1, m + int(1.0 * config.HZ))
        if m - lo < 2 or hi - m < 2:
            continue
        entry = s[lo:m].max()
        brakes.append(-a[lo:m + 1].min())
        reacc.append(a[m:hi + 1].max())
        if entry > 0:
            ratios.append(s[m] / entry)
    if not brakes:
        return out
    out.update({
        "brake_peak": float(np.max(brakes)),
        "brake_mean": float(np.mean(brakes)),
        "turn_speed_ratio": float(np.mean(ratios)) if ratios else np.nan,
        "reaccel_peak": float(np.mean(reacc)),
        "drill_time": float(t[i1] - t[i0]),
        "cod_vmax": float(peak),
        "n_turns": len(brakes),
    })
    return out


def attempt_features(ct: pd.DataFrame) -> pd.DataFrame:
    """One row per Combine attempt (40s and COD drills) with engine features."""
    keys = [k for k in ATTEMPT_KEYS if k in ct.columns]
    drills = [config.DRILL_FORTY, *config.COD_DRILLS]
    ct = ct[ct["drill_name"].isin(drills)].sort_values(keys + ["t"])
    rows = []
    for key, g in ct.groupby(keys, sort=False):
        rec = dict(zip(keys, key))
        t = g["t"].to_numpy(float)
        x = g["x"].to_numpy(float)
        y = g["y"].to_numpy(float)
        if rec["drill_name"] == config.DRILL_FORTY:
            rec.update(fit_sprint(t, x, y))
        else:
            rec.update(cod_features(t, x, y))
        rows.append(rec)
    return pd.DataFrame(rows)


SPRINT_COLS = ["S0", "tau", "A0", "Pmax_rel", "split10", "split20", "split40", "vmax_obs"]
COD_COLS = ["brake_peak", "brake_mean", "turn_speed_ratio", "reaccel_peak", "drill_time"]


def player_engine(attempts: pd.DataFrame, min_r2: float = 0.9) -> pd.DataFrame:
    """Collapse attempts to one row per player.

    Sprint: best attempt (fastest observed split40 among well-fitted attempts),
    mirroring how the official Combine keeps the best time.
    COD: braking/turn features are averaged per drill, then over drills.
    """
    out = []
    forty = attempts[attempts["drill_name"] == config.DRILL_FORTY]
    if not forty.empty:
        good = forty[forty["fit_r2"] >= min_r2].copy()
        good["_rank"] = good["split40"].fillna(99)
        best = good.sort_values("_rank").groupby("nfl_id").head(1)
        out.append(best.set_index("nfl_id")[SPRINT_COLS + ["fit_r2"]].add_prefix("eng_"))
        out.append(good.groupby("nfl_id").size().rename("eng_n_forty"))
    cod = attempts[attempts["drill_name"].isin(config.COD_DRILLS)]
    if not cod.empty:
        per_drill = cod.groupby(["nfl_id", "drill_name"])[COD_COLS].mean()
        wide = per_drill.drop(columns="drill_time").groupby("nfl_id").mean().add_prefix("eng_")
        times = per_drill["drill_time"].unstack("drill_name").add_prefix("eng_time_")
        out += [wide, times]
    if not out:
        return pd.DataFrame(index=pd.Index([], name="nfl_id"))
    return pd.concat(out, axis=1).rename_axis("nfl_id").reset_index()


def retest_pairs(attempts: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Attempt 1 vs attempt 2 for each player/drill, in long form, for reliability."""
    cols = cols or SPRINT_COLS + COD_COLS
    rows = []
    for (pid, drill), g in attempts.groupby(["nfl_id", "drill_name"]):
        g = g.sort_values("attempt")
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        for c in cols:
            if c in g and pd.notna(a.get(c)) and pd.notna(b.get(c)):
                rows.append({"nfl_id": pid, "drill_name": drill, "metric": c, "trial1": a[c], "trial2": b[c]})
    return pd.DataFrame(rows)
