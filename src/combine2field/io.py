"""Loaders for the Big Data Bowl 2027 files.

Large tracking tables are converted once to Parquet in the work directory so
later runs are fast. Only the columns the pipeline needs are read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config

NA = ["NA", ""]

COMBINE_TRACKING_COLS = [
    "nfl_id", "drill_name", "drill_type", "entity_type", "attempt", "event_id",
    "time", "x", "y", "s", "a",
]
GAME_TRACKING_COLS = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "event"]


def _path(name: str, data_dir: Path | None = None) -> Path:
    return (data_dir or config.data_dir()) / name


def _read_csv(name: str, data_dir: Path | None = None, **kw) -> pd.DataFrame:
    return pd.read_csv(_path(name, data_dir), na_values=NA, low_memory=False, **kw)


def players(data_dir: Path | None = None) -> pd.DataFrame:
    return _read_csv("players.csv", data_dir)


def combine_results(data_dir: Path | None = None) -> pd.DataFrame:
    df = _read_csv("combine_results.csv", data_dir)
    df["combine_position"] = df["combine_position"].str.upper()
    return df


def career(data_dir: Path | None = None) -> pd.DataFrame:
    return _read_csv("player_career_successes.csv", data_dir)


def games(data_dir: Path | None = None) -> pd.DataFrame:
    return _read_csv("games.csv", data_dir)


def player_play(data_dir: Path | None = None, columns: Iterable[str] | None = None) -> pd.DataFrame:
    usecols = None
    if columns is not None:
        header = pd.read_csv(_path("player_play.csv", data_dir), nrows=0).columns
        usecols = [c for c in columns if c in header]
    return _read_csv("player_play.csv", data_dir, usecols=usecols)


def time_to_seconds(t: pd.Series) -> np.ndarray:
    """Timestamps -> float seconds since epoch.

    The vectorised ISO8601 parser is ~100x faster than format="mixed", so it is
    tried first; numeric seconds are passed through.
    """
    if pd.api.types.is_numeric_dtype(t):
        return t.to_numpy(dtype=float)
    t = t.astype(str).str.replace(" ", "T", n=1, regex=False)
    try:
        ts = pd.to_datetime(t, format="ISO8601", utc=True)
    except (ValueError, TypeError):
        ts = pd.to_datetime(t, format="mixed", utc=True)
    return (ts - pd.Timestamp("2020-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def _cached(name: str, usecols: list[str], data_dir: Path | None, work: Path | None,
            dtypes: dict) -> pd.DataFrame:
    work = work or config.work_dir()
    cache = work / "cache" / name.replace(".csv", ".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    src = _path(name, data_dir)
    header = pd.read_csv(src, nrows=0).columns
    cols = [c for c in usecols if c in header]
    df = pd.read_csv(src, usecols=cols, na_values=NA, low_memory=False,
                     dtype={k: v for k, v in dtypes.items() if k in cols})
    df["t"] = time_to_seconds(df.pop("time"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def combine_tracking(data_dir: Path | None = None, work: Path | None = None) -> pd.DataFrame:
    df = _cached(
        "combine_tracking.csv", COMBINE_TRACKING_COLS, data_dir, work,
        {"x": "float32", "y": "float32", "s": "float32", "a": "float32"},
    )
    if "entity_type" in df:
        df = df[df["entity_type"].fillna("player") == "player"]
    return df


def game_tracking(season: int, data_dir: Path | None = None, work: Path | None = None) -> pd.DataFrame:
    return _cached(
        f"game_tracking_{season}.csv", GAME_TRACKING_COLS, data_dir, work,
        {"x": "float32", "y": "float32", "s": "float32", "a": "float32"},
    )


def player_base(data_dir: Path | None = None) -> pd.DataFrame:
    """One row per player: bio, draft slot, combine results, career outcomes."""
    p = players(data_dir)
    c = combine_results(data_dir).drop(columns=["draft_year"], errors="ignore")
    k = career(data_dir)
    df = p.merge(c, on="nfl_id", how="left").merge(k, on="nfl_id", how="left")
    df = df.rename(columns={"combine_position": "pos_group"})
    df["drafted"] = df["draft_overall_pick"].notna()
    df["pick"] = df["draft_overall_pick"].fillna(config.UNDRAFTED_PICK)
    df["seasons_available"] = (config.LAST_SEASON - df["draft_year"] + 1).clip(lower=1)
    return df
