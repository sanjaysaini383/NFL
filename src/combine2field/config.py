"""Paths, constants and position mappings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Kaggle mounts competition data here; locally use data/raw (or BDB_DATA_DIR).
_KAGGLE_DIRS = [
    Path("/kaggle/input/nfl-big-data-bowl-2027"),
    Path("/kaggle/input/competitions/nfl-big-data-bowl-2027"),
]


def on_kaggle() -> bool:
    """True only inside a Kaggle kernel (a stray local /kaggle folder must not count)."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir() and os.name != "nt"


def find_kaggle_data() -> Path | None:
    """Locate players.csv anywhere under /kaggle/input (mount paths vary between kernels)."""
    for d in _KAGGLE_DIRS:
        if (d / "players.csv").exists():
            return d
    base = Path("/kaggle/input")
    if base.is_dir():
        for hit in sorted(base.rglob("players.csv")):
            if (hit.parent / "combine_tracking.csv").exists():
                return hit.parent
    return None


def data_dir() -> Path:
    env = os.environ.get("BDB_DATA_DIR")
    if env:
        return Path(env)
    if on_kaggle():
        found = find_kaggle_data()
        if found is not None:
            return found
    return ROOT / "data" / "raw"


def work_dir() -> Path:
    """Where caches and derived tables go. Kept out of git (see .gitignore)."""
    env = os.environ.get("BDB_WORK_DIR")
    if env:
        return Path(env)
    if on_kaggle():
        return Path("/kaggle/working/c2f")
    return ROOT / "outputs"


SEASONS = (2023, 2024, 2025)
LAST_SEASON = 2025
POSITION_GROUPS = ("WR", "TE", "DB", "OL", "DL")

# Snap column that defines "playing time" for each group.
SNAP_COLUMN = {
    "WR": "career_offensive_snaps",
    "TE": "career_offensive_snaps",
    "OL": "career_offensive_snaps",
    "DL": "career_defensive_snaps",
    "DB": "career_defensive_snaps",
}

# Undrafted players get a pick just past the last compensatory pick.
UNDRAFTED_PICK = 263

# Drill names in combine_tracking.csv
DRILL_FORTY = "forty_yard_dash"
DRILL_THREE_CONE = "three_cone_drill"
DRILL_SHUTTLE = "short_shuttle"
COD_DRILLS = (DRILL_THREE_CONE, DRILL_SHUTTLE)

# Tracking sample rate (Hz)
HZ = 10.0
DT = 1.0 / HZ

# In-situ A-S profile settings (Morin et al., 2021, adapted to yards).
AS_MIN_SPEED = 3.0  # yd/s, ignore the noisy low-speed region
AS_BIN = 0.2  # yd/s
AS_TOP_N = 2  # top accelerations kept per bin
AS_MIN_BINS = 8  # minimum bins required for a valid fit

# Equal exposure: every player's in-game metrics use this many random plays.
EXPOSURE_PLAYS = 80
