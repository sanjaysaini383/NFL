"""Engine vs. Expression: put Combine and game physics side by side and test them.

Questions answered, per position group:
1. Reliability: which tracking metrics repeat across Combine attempts (ICC)?
2. Translation: does the Combine engine (A0, S0, braking) show up in games?
3. Value: which metrics predict NFL success *beyond draft slot*?
4. Increment: do tracking metrics add out-of-fold predictive value on top of
   draft slot and the traditional stopwatch Combine?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats
from .outcomes import LOWER_IS_BETTER, PRIMARY_PLAY_OUTCOME

ENGINE_FEATURES = {
    "eng_S0": "Combine top speed S0 (yd/s)",
    "eng_A0": "Combine max accel A0 (yd/s²)",
    "eng_tau": "Shorter accel time constant τ",
    "eng_Pmax_rel": "Combine relative power",
    "eng_brake_peak": "Combine peak braking (yd/s²)",
    "eng_turn_speed_ratio": "Combine turn speed retention",
    "eng_reaccel_peak": "Combine re-acceleration (yd/s²)",
}
EXPRESSION_FEATURES = {
    "game_S0": "Game A-S top speed S0 (yd/s)",
    "game_A0": "Game A-S max accel A0 (yd/s²)",
    "game_vmax": "Game peak speed (yd/s)",
    "game_brake": "Game peak braking (yd/s²)",
}
RATIO_FEATURES = {
    "expr_speed": "Speed expression (game peak / Combine S0)",
    "expr_accel": "Accel expression (game A0 / Combine A0)",
    "expr_brake": "Braking expression (game / Combine)",
}
# Finer roles inside a position group (from players.nfl_position).
ROLES = {"DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL"}
PASS_RUSH_FEATURES = ["eng_Pmax_rel", "ten_yd_split", "eng_A0", "eng_S0", "forty",
                      "game_A0", "game_brake", "game_vmax", "combine_weight"]

TRADITIONAL = ["forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle",
               "bench_reps", "combine_weight", "combine_height", "arm_length"]
FEATURE_LABELS = {**ENGINE_FEATURES, **EXPRESSION_FEATURES, **RATIO_FEATURES,
                  "forty": "Faster 40 (official)", "ten_yd_split": "Faster 10-yd split (official)",
                  "combine_weight": "Lighter body weight", "ngs_athleticism_score": "NGS athleticism score"}
# metrics where a smaller value means a better athlete
SMALLER_IS_BETTER = {"eng_tau", "forty", "ten_yd_split", "three_cone", "short_shuttle", "combine_weight"}


def assemble(base: pd.DataFrame, engine: pd.DataFrame, expression: pd.DataFrame,
             career: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    df = (base.merge(engine, on="nfl_id", how="left")
              .merge(expression, on="nfl_id", how="left")
              .merge(career, on="nfl_id", how="left")
              .merge(plays, on="nfl_id", how="left"))
    df["expr_speed"] = df["game_vmax"] / df["eng_S0"]
    df["expr_accel"] = df["game_A0"] / df["eng_A0"]
    df["expr_brake"] = df["game_brake"] / df["eng_brake_peak"]
    df["log_pick"] = np.log(df["pick"])
    df["role"] = df["nfl_position"].astype(str).str.upper().map(ROLES).fillna(df["pos_group"])
    # position percentiles make cards and comps readable
    for c in list(ENGINE_FEATURES) + list(EXPRESSION_FEATURES) + list(RATIO_FEATURES):
        if c in df:
            df[f"pct_{c}"] = df.groupby("pos_group")[c].rank(
                pct=True, ascending=c not in SMALLER_IS_BETTER)
    return df


def reliability(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (drill, metric), g in pairs.groupby(["drill_name", "metric"]):
        rows.append({"drill_name": drill, "metric": metric, "n": len(g),
                     "icc": stats.icc_3_1(g["trial1"], g["trial2"]),
                     "r": g[["trial1", "trial2"]].corr().iloc[0, 1] if len(g) > 2 else np.nan})
    return pd.DataFrame(rows).sort_values("icc", ascending=False)


def translation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Combine engine vs in-game expression of the same quantity."""
    pairs = [("eng_S0", "game_vmax"), ("eng_S0", "game_S0"), ("eng_A0", "game_A0"),
             ("eng_brake_peak", "game_brake"), ("forty", "game_vmax")]
    rows = []
    for pos, g in df.groupby("pos_group"):
        for a, b in pairs:
            if a in g and b in g:
                r = stats.spearman_ci(g[a], g[b])
                rows.append({"pos_group": pos, "combine": a, "game": b, **r})
    return pd.DataFrame(rows)


def _outcomes_for(pos: str) -> list[str]:
    return ["snaps_per_season", "starts_per_season", PRIMARY_PLAY_OUTCOME.get(pos)]


def association_table(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Spearman and draft-adjusted partial Spearman for every feature x outcome x position.

    Signs are oriented so that a positive value always means "better athlete ->
    better outcome". q-values use Benjamini-Hochberg within each position.
    """
    features = features or [f for f in FEATURE_LABELS if f in df]
    rows = []
    for pos, g in df.groupby("pos_group"):
        for out in _outcomes_for(pos):
            if out is None or out not in g or g[out].notna().sum() < 10:
                continue
            y = g[out] * (-1 if out in LOWER_IS_BETTER else 1)
            for f in features:
                if f not in g:
                    continue
                x = g[f] * (-1 if f in SMALLER_IS_BETTER else 1)
                r = stats.spearman_ci(x, y)
                pr = stats.partial_spearman(x, y, g["log_pick"])
                rows.append({"pos_group": pos, "outcome": out, "feature": f, **r, **pr})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["q"] = res.groupby("pos_group")["p"].transform(stats.bh_qvalues)
    res["pq"] = res.groupby("pos_group")["pp"].transform(stats.bh_qvalues)
    return res


def pass_rush_table(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """DL pressure rate vs each trait, draft-adjusted, split into EDGE and interior (IDL).

    Edge rushers are lighter, faster and pressure more often, so a pooled DL
    correlation partly measures role. Splitting by role removes that.
    """
    dl = df[df["pos_group"] == "DL"]
    rows = []
    for role, g in [("All DL", dl), *dl.groupby("role")]:
        for f in PASS_RUSH_FEATURES:
            if f not in g:
                continue
            x = g[f] * (-1 if f in SMALLER_IS_BETTER else 1)
            r = stats.partial_spearman_ci(x, g["pressure_rate"], g["log_pick"], n_boot=n_boot)
            rows.append({"role": role, "feature": f, **r})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pq"] = out.groupby("role")["pp"].transform(stats.bh_qvalues)
    return out


def incremental_value(df: pd.DataFrame, n_repeats: int = 20) -> pd.DataFrame:
    """Out-of-fold R^2 for nested models: draft slot -> + stopwatch Combine -> + tracking."""
    tracking = ["eng_S0", "eng_A0", "eng_brake_peak", "eng_turn_speed_ratio",
                "game_S0", "game_A0", "game_vmax", "expr_speed"]
    models = {
        "Draft slot": ["log_pick"],
        "+ Stopwatch Combine": ["log_pick", *TRADITIONAL],
        "+ Tracking (engine & expression)": ["log_pick", *TRADITIONAL, *tracking],
    }
    rows = []
    for pos, g in df.groupby("pos_group"):
        for out in _outcomes_for(pos):
            if out is None or out not in g:
                continue
            for name, feats in models.items():
                sc = stats.cv_r2(g, out, feats, n_repeats=n_repeats)
                rows.append({"pos_group": pos, "outcome": out, "model": name,
                             "r2": np.nanmean(sc), "r2_sd": np.nanstd(sc), "n": int(g[out].notna().sum())})
    return pd.DataFrame(rows)


def comps(df: pd.DataFrame, nfl_id: int, k: int = 5,
          features=("pct_eng_S0", "pct_eng_A0", "pct_eng_brake_peak", "pct_game_vmax", "pct_game_A0")) -> pd.DataFrame:
    """Nearest neighbours within a position group on percentile profile."""
    me = df.loc[df["nfl_id"] == nfl_id].iloc[0]
    pool = df[(df["pos_group"] == me["pos_group"]) & (df["nfl_id"] != nfl_id)]
    feats = [f for f in features if f in df and pd.notna(me.get(f))]
    if not feats:
        return pool.head(0)
    d = np.sqrt(((pool[feats] - me[feats].astype(float)) ** 2).sum(axis=1, min_count=1))
    return pool.assign(distance=d).nsmallest(k, "distance")
