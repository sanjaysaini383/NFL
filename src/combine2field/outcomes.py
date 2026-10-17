"""Position-specific definitions of "NFL success".

Career outcomes are normalised by the seasons a player has been in the league
(the 2023 class has three seasons, the 2025 class one). Play-level outcomes come
from player_play.csv and are only computed with enough plays to be stable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PLAY_COLUMNS = [
    "game_id", "play_id", "nfl_id", "route_ran", "separation_at_pass_forward", "target",
    "pressure_allowed", "sack_allowed", "time_to_pressure", "player_get_off",
    "pass_rushers_encountered", "tackle", "sack",
]
MIN_PLAYS = 30

OUTCOME_LABELS = {
    "snaps_per_season": "Snaps per season",
    "starts_per_season": "Starts per season",
    "starter": "Became a starter (≥8 starts/season)",
    "sep_mean": "Avg separation at throw (yd)",
    "target_rate": "Targets per route",
    "pressure_allowed_rate": "Pressure allowed rate (lower = better)",
    "pressure_rate": "Pressure rate per pass rush",
    "get_off": "Get-off time (s, lower = better)",
    "tackle_rate": "Tackles per play",
}
# outcomes where lower is better; their sign is flipped in analysis
LOWER_IS_BETTER = {"pressure_allowed_rate", "get_off"}

PRIMARY_PLAY_OUTCOME = {
    "WR": "sep_mean", "TE": "sep_mean", "OL": "pressure_allowed_rate",
    "DL": "pressure_rate", "DB": "tackle_rate",
}


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    m = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    return s.astype(str).str.lower().map(m)


def career_outcomes(base: pd.DataFrame) -> pd.DataFrame:
    seasons = base["seasons_available"].astype(float)
    snaps = np.select(
        [base["pos_group"].map(config.SNAP_COLUMN) == c for c in
         ("career_offensive_snaps", "career_defensive_snaps")],
        [base["career_offensive_snaps"], base["career_defensive_snaps"]],
        default=np.nan,
    )
    out = pd.DataFrame({"nfl_id": base["nfl_id"]})
    out["snaps_per_season"] = snaps / seasons
    out["starts_per_season"] = base["career_games_started"] / seasons
    out["starter"] = (out["starts_per_season"] >= 8).astype(float)
    return out


def play_outcomes(pp: pd.DataFrame) -> pd.DataFrame:
    pp = pp.copy()
    for c in ("target", "pressure_allowed", "sack_allowed"):
        if c in pp:
            pp[c] = _bool(pp[c])
    g = pp.groupby("nfl_id")
    parts = []
    if "route_ran" in pp:
        routes = pp[pp["route_ran"].notna()]
        r = routes.groupby("nfl_id").agg(
            n_routes=("play_id", "size"),
            sep_mean=("separation_at_pass_forward", "mean"),
            target_rate=("target", "mean"),
        )
        r.loc[r["n_routes"] < MIN_PLAYS, ["sep_mean", "target_rate"]] = np.nan
        parts.append(r)
    if "pressure_allowed" in pp:
        pb = pp[pp["pressure_allowed"].notna()]
        b = pb.groupby("nfl_id").agg(n_pass_block=("play_id", "size"),
                                     pressure_allowed_rate=("pressure_allowed", "mean"))
        b.loc[b["n_pass_block"] < MIN_PLAYS, "pressure_allowed_rate"] = np.nan
        parts.append(b)
    if "player_get_off" in pp:
        rush = pp[pp["player_get_off"].notna()].assign(
            pressured=lambda d: d["time_to_pressure"].notna().astype(float))
        k = rush.groupby("nfl_id").agg(n_pass_rush=("play_id", "size"),
                                       pressure_rate=("pressured", "mean"),
                                       get_off=("player_get_off", "mean"))
        k.loc[k["n_pass_rush"] < MIN_PLAYS, ["pressure_rate", "get_off"]] = np.nan
        parts.append(k)
    if "tackle" in pp:
        t = g.agg(n_plays=("play_id", "size"), tackle_rate=("tackle", "mean"))
        t.loc[t["n_plays"] < MIN_PLAYS, "tackle_rate"] = np.nan
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["nfl_id"])
    return pd.concat(parts, axis=1).rename_axis("nfl_id").reset_index()
