"""Writeup cover image (1280x640): writeup/figures/thumbnail.png

    python scripts/make_thumbnail.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG = "#0f1b2d"
INK = "#f0efec"
INK2 = "#c3c2b7"
BLUE = "#3987e5"
ORANGE = "#eb6834"
GRID = "#24324a"

fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)

# --- left: text
fig.text(0.05, 0.80, "NFL BIG DATA BOWL 2027", color=ORANGE, fontsize=15, fontweight="bold")
fig.text(0.05, 0.62, "Engine vs.\nExpression", color=INK, fontsize=50, fontweight="bold",
         va="center", linespacing=1.0)
fig.text(0.05, 0.40, "Which Combine athleticism\nshows up on Sundays?", color=INK2, fontsize=20,
         va="center", linespacing=1.25)
fig.text(0.05, 0.17, "Edge rushers win with the first step,\nnot top speed.", color=INK, fontsize=16,
         fontweight="bold", va="center", linespacing=1.3)

# --- right: the two A-S lines
ax = fig.add_axes([0.58, 0.16, 0.38, 0.66], facecolor=BG)
v = np.linspace(0, 12.5, 100)
ax.plot(v, np.clip(9.0 * (1 - v / 12.0), 0, None), color=BLUE, lw=5, ls=(0, (4, 2)))
ax.plot(v, np.clip(8.6 * (1 - v / 9.8), 0, None), color=ORANGE, lw=5)
rng = np.random.default_rng(3)
vs = np.linspace(3, 9.4, 26)
ax.scatter(vs, np.clip(8.6 * (1 - vs / 9.8) + rng.normal(0, 0.35, vs.size), 0.1, None),
           s=60, color=ORANGE, edgecolor=BG, linewidth=1.5, zorder=3)
ax.text(6.5, 5.0, "Combine engine", color=BLUE, fontsize=16, fontweight="bold")
ax.text(0.4, 1.2, "In-game expression", color=ORANGE, fontsize=16, fontweight="bold")
ax.set_xlim(0, 12.8)
ax.set_ylim(0, 10)
ax.set_xlabel("Speed", color=INK2, fontsize=14)
ax.set_ylabel("Acceleration", color=INK2, fontsize=14)
ax.tick_params(colors=BG, length=0, labelsize=0)
for side, s in ax.spines.items():
    s.set_visible(side in ("left", "bottom"))
    s.set_color(INK2)
    s.set_linewidth(1.5)
ax.grid(color=GRID, lw=1)

out = ROOT / "writeup" / "figures" / "thumbnail.png"
fig.savefig(out, facecolor=BG)
print(f"wrote {out}")
