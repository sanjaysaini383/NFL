"""Speed and tangential acceleration from 10 Hz x/y tracking.

The Combine `dis` field is unreliable, and the provided `a` is the magnitude of
acceleration, which includes centripetal acceleration in turns. For
acceleration-speed profiles we need *tangential* acceleration (dv/dt), so both
are rebuilt from the position trace.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth(values: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    if values.size < window:
        return values.astype(float)
    return savgol_filter(values.astype(float), window, poly)


def speed_from_xy(t: np.ndarray, x: np.ndarray, y: np.ndarray, window: int = 5) -> np.ndarray:
    """Speed (yd/s) from smoothed positions."""
    if t.size < 3:
        return np.zeros(t.size)
    xs, ys = smooth(x, window), smooth(y, window)
    vx = np.gradient(xs, t)
    vy = np.gradient(ys, t)
    return np.hypot(vx, vy)


def tangential_acc(t: np.ndarray, s: np.ndarray, window: int = 5) -> np.ndarray:
    if t.size < 3:
        return np.zeros(t.size)
    return np.gradient(smooth(s, window), t)


def path_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative distance travelled along the trace (yd)."""
    step = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(step)])


def grouped_tangential_acc(df: pd.DataFrame, keys: list[str], s_col: str = "s") -> pd.Series:
    """Centred-difference dv/dt for each tracking group, fully vectorised.

    `df` must already be sorted by keys + "t". Rows at a group boundary get NaN.
    """
    same_prev = np.ones(len(df), dtype=bool)
    same_next = np.ones(len(df), dtype=bool)
    for k in keys:
        v = df[k].to_numpy()
        same_prev[1:] &= v[1:] == v[:-1]
        same_next[:-1] &= v[:-1] == v[1:]
    same_prev[0] = False
    same_next[-1] = False
    s = df[s_col].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    s_next = np.roll(s, -1)
    s_prev = np.roll(s, 1)
    t_next = np.roll(t, -1)
    t_prev = np.roll(t, 1)
    ok = same_prev & same_next
    dt = np.where(ok, t_next - t_prev, np.nan)
    acc = np.where(ok & (dt > 0) & (dt < 0.35), (s_next - s_prev) / dt, np.nan)
    return pd.Series(acc, index=df.index)
