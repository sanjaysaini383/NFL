"""Static figures for the write-up (matplotlib, light theme for Kaggle)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config  # noqa: E402
from .combine_features import _mono_exp, fit_sprint  # noqa: E402
from .kinematics import speed_from_xy  # noqa: E402
from .outcomes import PRIMARY_PLAY_OUTCOME  # noqa: E402
from .translation import FEATURE_LABELS  # noqa: E402

METRIC_LABELS = {
    "vmax_obs": "Peak speed", "S0": "Top speed S0", "split40": "40-yd split", "split20": "20-yd split",
    "split10": "10-yd split", "Pmax_rel": "Relative power", "A0": "Max accel A0", "tau": "Time constant τ",
    "drill_time": "Drill time", "turn_speed_ratio": "Turn speed retention", "brake_mean": "Mean braking",
    "brake_peak": "Peak braking", "reaccel_peak": "Re-acceleration",
}
DRILL_LABELS = {"forty_yard_dash": "40", "three_cone_drill": "3-cone", "short_shuttle": "shuttle"}

# Reference palette (dataviz skill): surfaces, ink, categorical slots, diverging.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots validate all-pairs
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "c2f_div", ["#b7302f", "#e34948", "#f4a3a2", NEUTRAL, "#9ec5f4", "#2a78d6", "#184f95"])
POS_ORDER = list(config.POSITION_GROUPS)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "ytick.labelcolor": INK2, "xtick.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2, "legend.frameon": False,
    })


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_method(ct: pd.DataFrame, bins: pd.DataFrame, player: pd.Series, out: Path) -> Path:
    """One player, one model: Combine 40 speed curve and in-game A-S cloud on shared axes."""
    _style()
    pid = player["nfl_id"]
    forty = ct[(ct["nfl_id"] == pid) & (ct["drill_name"] == config.DRILL_FORTY)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    if not forty.empty:
        att = forty.groupby("attempt")
        g = att.get_group(sorted(att.groups)[0]).sort_values("t")
        t = g["t"].to_numpy() - g["t"].iloc[0]
        x, y = g["x"].to_numpy(float), g["y"].to_numpy(float)
        s = speed_from_xy(t, x, y)
        fit = fit_sprint(t, x, y)
        a1.plot(t, s, color=MUTED, lw=1.2, marker="o", ms=2.5, label="Tracked speed (10 Hz)")
        if np.isfinite(fit["S0"]):
            tt = np.linspace(0, t.max(), 300)
            a1.plot(tt, _mono_exp(tt, fit["S0"], fit["tau"], fit["t0"]), color=SERIES[0],
                    label=f"Fit: S0={fit['S0']:.1f} yd/s, τ={fit['tau']:.2f} s")
    a1.set(title="Combine 40: the engine", xlabel="Time (s)", ylabel="Speed (yd/s)")
    a1.legend(loc="lower right", fontsize=8)

    b = bins[bins["nfl_id"] == pid]
    v = config.AS_MIN_SPEED + (b["bin"] + 0.5) * config.AS_BIN
    a2.scatter(v, b["a"], s=14, color=SERIES[1], alpha=0.75, label="Game frames (top-2 per bin)",
               edgecolor=SURFACE, linewidth=0.6)
    vv = np.linspace(0, max(player.get("eng_S0", 10) or 10, 1) * 1.02, 50)
    if pd.notna(player.get("game_A0")):
        a2.plot(vv, np.clip(player["game_A0"] * (1 - vv / player["game_S0"]), 0, None),
                color=SERIES[1], label=f"Game A-S: A0={player['game_A0']:.1f}, S0={player['game_S0']:.1f}")
    if pd.notna(player.get("eng_A0")):
        a2.plot(vv, np.clip(player["eng_A0"] * (1 - vv / player["eng_S0"]), 0, None),
                color=SERIES[0], ls="--", label=f"Combine A-S: A0={player['eng_A0']:.1f}, S0={player['eng_S0']:.1f}")
    a2.set(title="Games: the expression", xlabel="Speed (yd/s)", ylabel="Acceleration (yd/s²)")
    a2.set_ylim(bottom=0)
    a2.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{player.get('display_name', pid)} ({player.get('pos_group', '')}): "
                 "one acceleration-speed model, two environments", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "fig1_method")


def fig_reliability(rel: pd.DataFrame, out: Path, top: int = 14) -> Path:
    _style()
    r = rel.dropna(subset=["icc"]).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(r) + 1.2))
    labels = [f"{METRIC_LABELS.get(m, m)}  ·  {DRILL_LABELS.get(d, d)}"
              for d, m in zip(r["drill_name"], r["metric"])]
    ax.barh(labels, r["icc"], color=SERIES[0], height=0.62)
    for i, v in enumerate(r["icc"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0.75, color=MUTED, lw=1, ls=":")
    ax.text(0.75, len(r) - 0.3, " good (0.75)", color=MUTED, fontsize=8)
    ax.set(xlim=(0, 1.08), xlabel="Test-retest ICC(3,1), attempt 1 vs 2",
           title="Which tracking metrics repeat across Combine attempts?")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "fig2_reliability")


def fig_translation(df: pd.DataFrame, out: Path, x="eng_S0", y="game_vmax") -> Path:
    _style()
    pos = [p for p in POS_ORDER if p in set(df["pos_group"])]
    fig, axes = plt.subplots(1, len(pos), figsize=(2.6 * len(pos), 3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    d = df.dropna(subset=[x, y])
    lo, hi = (d[[x, y]].min().min() * 0.95, d[[x, y]].max().max() * 1.03) if len(d) else (0, 1)
    for ax, p in zip(axes, pos):
        g = d[d["pos_group"] == p]
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, ls="--")
        ax.scatter(g[x], g[y], s=16, color=SERIES[0], alpha=0.8, edgecolor=SURFACE, linewidth=0.6)
        med = (g[y] / g[x]).median() if len(g) else np.nan
        ax.set_title(f"{p}  ·  {med:.0%} expressed" if np.isfinite(med) else p)
        ax.set_xlabel("Combine S0 (yd/s)")
    axes[0].set_ylabel("Game peak speed (yd/s)")
    fig.suptitle("How much of the Combine top speed shows up in games? (dashed = 100%)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "fig3_translation")


def fig_heatmap(assoc: pd.DataFrame, out: Path, outcome="snaps_per_season", value="prho") -> Path:
    _style()
    if outcome == "primary":  # each position's own play-level outcome
        a = assoc[assoc["outcome"] == assoc["pos_group"].map(PRIMARY_PLAY_OUTCOME)]
    else:
        a = assoc[assoc["outcome"] == outcome]
    m = a.pivot_table(index="feature", columns="pos_group", values=value)
    q = a.pivot_table(index="feature", columns="pos_group", values="pq" if value == "prho" else "q")
    m = m.reindex(columns=[p for p in POS_ORDER if p in m.columns])
    q = q.reindex_like(m)
    order = m.abs().mean(axis=1).sort_values(ascending=False).index
    m, q = m.loc[order], q.loc[order]
    fig, ax = plt.subplots(figsize=(1.0 * m.shape[1] + 4.2, 0.36 * m.shape[0] + 1.4))
    ax.imshow(m.to_numpy(float), cmap=DIVERGING, vmin=-0.5, vmax=0.5, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), [FEATURE_LABELS.get(f, f) for f in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if np.isfinite(v):
                sig = np.isfinite(q.iat[i, j]) and q.iat[i, j] < 0.10
                ax.text(j, i, f"{v:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=8,
                        color=INK if abs(v) < 0.35 else SURFACE, fontweight="bold" if sig else "normal")
    what = "each position's play-level outcome" if outcome == "primary" else outcome.replace("_", " ")
    ax.set_title(f"Draft-adjusted partial ρ with {what}  (* BH q < 0.10)")
    if outcome == "primary":
        ax.set_xticks(range(m.shape[1]),
                      [f"{p}\n{PRIMARY_PLAY_OUTCOME[p].replace('_', ' ')}" for p in m.columns])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, out, "fig4_value_beyond_draft" + ("_play" if outcome == "primary" else ""))


def fig_pass_rush(pr: pd.DataFrame, out: Path) -> Path:
    """Dot-and-whisker: draft-adjusted correlation with pressure rate, EDGE vs interior."""
    _style()
    roles = [r for r in ("EDGE", "IDL") if r in set(pr["role"])]
    feats = list(dict.fromkeys(pr["feature"]))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(feats) + 1.4))
    names = {"EDGE": "Edge rushers (DE/OLB)", "IDL": "Interior (DT/NT)"}
    for k, role in enumerate(roles):
        g = pr[pr["role"] == role].set_index("feature").reindex(feats)
        ys = np.arange(len(feats)) + (k - (len(roles) - 1) / 2) * 0.28
        ax.errorbar(g["prho"], ys, xerr=[g["prho"] - g["plo"], g["phi"] - g["prho"]], fmt="o",
                    color=SERIES[k], ms=6, lw=1.6, capsize=0,
                    label=f"{names[role]}, n≤{int(g['n'].max())}")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(feats)), [FEATURE_LABELS.get(f, f) for f in feats])
    ax.invert_yaxis()
    ax.set(xlabel="Partial ρ with pressure rate, controlling for draft slot (95% bootstrap CI)",
           title="Pass rush: which traits predict pressure beyond draft slot?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "fig6_pass_rush")


def fig_increment(inc: pd.DataFrame, out: Path, outcome="snaps_per_season") -> Path:
    _style()
    d = inc[inc["outcome"] == outcome]
    models = list(dict.fromkeys(d["model"]))
    pos = [p for p in POS_ORDER if p in set(d["pos_group"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.8 / max(len(models), 1)
    for k, mname in enumerate(models[:3]):
        g = d[d["model"] == mname].set_index("pos_group").reindex(pos)
        xs = np.arange(len(pos)) + (k - (len(models) - 1) / 2) * w
        ax.bar(xs, g["r2"], width=w * 0.92, color=SERIES[k], label=mname)
        ax.errorbar(xs, g["r2"], yerr=g["r2_sd"], fmt="none", ecolor=INK2, lw=1, capsize=2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(range(len(pos)), pos)
    ax.set(ylabel="Out-of-fold R² (rank outcome)",
           title=f"Does tracking add predictive value for {outcome.replace('_', ' ')}?")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "fig5_incremental_value")
