"""In-game "expression": in-situ acceleration-speed profiles from game tracking.

Method (Morin et al., 2021, "Individual acceleration-speed profile in-situ"):
pool every frame a player produced, bin speed in 0.2 yd/s steps above 3 yd/s,
keep the two highest tangential accelerations per bin, and fit a straight line
a = A0 - (A0/S0) * v. Points outside the 95% prediction band are dropped and the
line is refitted. The intercept A0 and the speed intercept S0 are directly
comparable to the Combine sprint fit.

The game files have ~22.6M rows, so the reduction runs per season and keeps only
the per-bin top accelerations and per-play peaks for each player.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config, io
from .kinematics import grouped_tangential_acc

PLAY_KEYS = ["nfl_id", "game_id", "play_id"]
MAX_SPEED = 12.5  # yd/s; anything faster is a tracking glitch
MAX_ACC = 12.0  # yd/s^2


def frame_reduce(gt: pd.DataFrame, nfl_ids: set | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce one season of game tracking.

    Returns
    -------
    bins : nfl_id, bin, a   (top-N tangential accelerations per 0.2 yd/s bin)
    plays: nfl_id, game_id, play_id, s_max, a_min, a_max  (per-play peaks)
    """
    if nfl_ids is not None:
        gt = gt[gt["nfl_id"].isin(nfl_ids)]
    gt = gt[PLAY_KEYS + ["t", "s"]].dropna().sort_values(PLAY_KEYS + ["t"], kind="stable",
                                                         ignore_index=True)
    gt["a_tan"] = grouped_tangential_acc(gt, PLAY_KEYS).to_numpy(dtype="float32")
    gt = gt[(gt["s"] <= MAX_SPEED) & (gt["a_tan"].abs() <= MAX_ACC)]

    plays = gt.groupby(PLAY_KEYS, sort=False).agg(
        s_max=("s", "max"), a_min=("a_tan", "min"), a_max=("a_tan", "max"),
    ).reset_index()

    hs = gt[gt["s"] >= config.AS_MIN_SPEED]
    b = np.floor((hs["s"].to_numpy() - config.AS_MIN_SPEED) / config.AS_BIN).astype(int)
    bins = pd.DataFrame({"nfl_id": hs["nfl_id"].to_numpy(), "bin": b, "a": hs["a_tan"].to_numpy()})
    bins = top_n_per_bin(bins)
    return bins, plays


def top_n_per_bin(bins: pd.DataFrame, n: int = config.AS_TOP_N) -> pd.DataFrame:
    bins = bins.sort_values(["nfl_id", "bin", "a"], ascending=[True, True, False])
    return bins.groupby(["nfl_id", "bin"], sort=False).head(n).reset_index(drop=True)


def fit_as_profile(speed: np.ndarray, acc: np.ndarray) -> dict:
    """Robust linear A-S fit. Returns A0 (yd/s^2), S0 (yd/s), r2, n points."""
    out = {"A0": np.nan, "S0": np.nan, "as_r2": np.nan, "as_n": int(speed.size)}
    ok = (acc > 0) & np.isfinite(acc)
    v, a = speed[ok], acc[ok]
    if v.size < 2 * config.AS_MIN_BINS:
        return out
    for _ in range(2):
        slope, inter, r, _, _ = stats.linregress(v, a)
        resid = a - (inter + slope * v)
        sd = resid.std(ddof=2) if v.size > 3 else 0
        keep = np.abs(resid) <= 1.96 * sd if sd > 0 else np.ones_like(v, bool)
        if keep.all():
            break
        v, a = v[keep], a[keep]
        if v.size < 2 * config.AS_MIN_BINS:
            return out
    slope, inter, r, _, _ = stats.linregress(v, a)
    if slope >= 0 or inter <= 0:
        return out
    out.update({"A0": inter, "S0": -inter / slope, "as_r2": r ** 2, "as_n": int(v.size)})
    return out


def player_expression(bins: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """One row per player: in-game A-S profile and peak-intensity summaries."""
    bins = top_n_per_bin(bins)
    rows = []
    for pid, g in bins.groupby("nfl_id"):
        v = config.AS_MIN_SPEED + (g["bin"].to_numpy() + 0.5) * config.AS_BIN
        rows.append({"nfl_id": pid, **fit_as_profile(v, g["a"].to_numpy())})
    prof = pd.DataFrame(rows).set_index("nfl_id").add_prefix("game_") if rows else pd.DataFrame()
    peaks = plays.groupby("nfl_id").agg(
        game_n_plays=("s_max", "size"),
        game_vmax=("s_max", lambda x: np.nanpercentile(x, 99.5)),
        game_v95=("s_max", lambda x: np.nanpercentile(x, 95)),
        game_brake=("a_min", lambda x: -np.nanpercentile(x, 1)),
        game_accel=("a_max", lambda x: np.nanpercentile(x, 99)),
    )
    return peaks.join(prof, how="left").rename_axis("nfl_id").reset_index()


def sample_plays(play_keys: pd.DataFrame, k: int, seed: int = 2027) -> pd.DataFrame:
    """Pick the same number of plays (k) for every player, at random.

    Peak-based metrics rise with the number of chances a player gets, and
    players who earn more snaps get more chances. Measuring everyone on k
    random plays removes that exposure effect. Players with fewer than k
    tracked plays are dropped rather than measured on a smaller sample.
    """
    rng = np.random.default_rng(seed)
    pk = play_keys.drop_duplicates().assign(_u=lambda d: rng.random(len(d)))
    pk["_rank"] = pk.groupby("nfl_id")["_u"].rank(method="first")
    n = pk.groupby("nfl_id")["_u"].transform("size")
    return pk[(pk["_rank"] <= k) & (n >= k)][PLAY_KEYS].reset_index(drop=True)


def build(nfl_ids: set | None = None, data_dir: Path | None = None, work: Path | None = None,
          seasons=config.SEASONS, plays_per_player: int | None = config.EXPOSURE_PLAYS,
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the per-season reduction and return (expression, bins, plays).

    With `plays_per_player` set, every player is measured on the same number of
    randomly sampled plays (equal exposure); `game_n_plays_total` keeps the full count.
    """
    def load(season):
        try:
            gt = io.game_tracking(season, data_dir, work)
        except FileNotFoundError:
            return None
        return gt[gt["nfl_id"].isin(nfl_ids)] if nfl_ids is not None else gt

    # pass 1: which plays exist for each player
    keys = [g[PLAY_KEYS].drop_duplicates() for s in seasons if (g := load(s)) is not None]
    if not keys:
        raise FileNotFoundError("No game_tracking_<season>.csv files found")
    keys = pd.concat(keys, ignore_index=True)
    totals = keys.groupby("nfl_id").size().rename("game_n_plays_total")
    chosen = sample_plays(keys, plays_per_player) if plays_per_player else keys
    chosen_idx = pd.MultiIndex.from_frame(chosen)

    # pass 2: reduce only the chosen plays
    all_bins, all_plays = [], []
    for season in seasons:
        gt = load(season)
        if gt is None:
            continue
        gt = gt[pd.MultiIndex.from_frame(gt[PLAY_KEYS]).isin(chosen_idx)]
        b, p = frame_reduce(gt)
        all_bins.append(b)
        all_plays.append(p)
        del gt
    bins = pd.concat(all_bins, ignore_index=True)
    plays = pd.concat(all_plays, ignore_index=True)
    expr = player_expression(bins, plays).merge(totals.reset_index(), on="nfl_id", how="right")
    return expr, top_n_per_bin(bins), plays
