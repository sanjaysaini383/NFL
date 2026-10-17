"""Prospect Translator — Engine vs. Expression scouting app.

    streamlit run app/streamlit_app.py

Reads the tables written by `combine2field` (outputs/ by default). With no
real outputs it builds and labels a SYNTHETIC demo so the app always opens.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combine2field import config, pipeline  # noqa: E402
from combine2field.combine_features import engine_from_splits  # noqa: E402
from combine2field.outcomes import OUTCOME_LABELS  # noqa: E402
from combine2field.translation import FEATURE_LABELS, comps  # noqa: E402

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
DIVERGING = [[0, "#b7302f"], [0.25, "#f4a3a2"], [0.5, "#f0efec"], [0.75, "#9ec5f4"], [1, "#184f95"]]

st.set_page_config(page_title="Prospect Translator", page_icon="🏈", layout="wide")


@st.cache_data(show_spinner="Loading results…")
def load(work: str):
    w = Path(work)
    t = w / "tables"
    return (pd.read_parquet(t / "players.parquet"), pd.read_parquet(t / "game_as_bins.parquet"),
            pd.read_csv(t / "associations.csv"), pd.read_csv(t / "incremental_value.csv"),
            pd.read_csv(t / "reliability.csv"))


def resolve_work() -> tuple[Path, bool]:
    real = Path(os.environ.get("BDB_WORK_DIR", config.work_dir()))
    if (real / "tables" / "players.parquet").exists():
        return real, False
    demo = ROOT / "outputs" / "synthetic"
    if not (demo / "tables" / "players.parquet").exists():
        with st.spinner("No results found — building a SYNTHETIC demo (about 30 s)…"):
            pipeline.main(["--synthetic", "--repeats", "3", "--no-figures"])
    return demo, True


work, is_demo = resolve_work()
df, bins, assoc, inc, rel = load(str(work))

st.title("Prospect Translator")
st.caption("Engine vs. Expression: how much of a prospect's Combine physics shows up on Sundays, "
           "and which parts predict NFL success beyond draft slot.  ·  NFL Big Data Bowl 2027")
if is_demo:
    st.warning("**Demo mode: SYNTHETIC data.** No real player or measurement is shown. Run "
               "`combine2field` on the competition data to see real results.", icon="⚠️")

tab_card, tab_calc, tab_value, tab_rel = st.tabs(
    ["Prospect card", "Stopwatch → engine calculator", "What matters, by position", "Reliability"])

# ------------------------------------------------------------------ prospect card
with tab_card:
    c1, c2 = st.columns([1, 3])
    pos = c1.selectbox("Position group", [p for p in config.POSITION_GROUPS if p in set(df["pos_group"])])
    pool = df[df["pos_group"] == pos].sort_values("display_name")
    name = c2.selectbox("Player", pool["display_name"])
    p = pool[pool["display_name"] == name].iloc[0]

    m = st.columns(5)
    m[0].metric("Draft pick", "UDFA" if not p["drafted"] else int(p["pick"]))
    m[1].metric("Combine S0", f"{p['eng_S0']:.2f} yd/s" if pd.notna(p["eng_S0"]) else "—")
    m[2].metric("Game peak speed", f"{p['game_vmax']:.2f} yd/s" if pd.notna(p["game_vmax"]) else "—")
    m[3].metric("Speed expressed", f"{p['expr_speed']:.0%}" if pd.notna(p["expr_speed"]) else "—")
    m[4].metric("Snaps / season", f"{p['snaps_per_season']:.0f}" if pd.notna(p["snaps_per_season"]) else "—")

    left, right = st.columns(2)
    with left:
        st.subheader("Percentile within position")
        rows = [("Engine", "eng_S0"), ("Engine", "eng_A0"), ("Engine", "eng_brake_peak"),
                ("Engine", "eng_turn_speed_ratio"), ("Expression", "game_vmax"),
                ("Expression", "game_A0"), ("Expression", "game_brake"), ("Expression", "expr_speed")]
        lab, val, col = [], [], []
        for grp, f in rows:
            v = p.get(f"pct_{f}")
            if pd.notna(v):
                lab.append(FEATURE_LABELS.get(f, f))
                val.append(100 * v)
                col.append(BLUE if grp == "Engine" else ORANGE)
        fig = go.Figure(go.Bar(x=val, y=lab, orientation="h", marker_color=col,
                               text=[f"{v:.0f}" for v in val], textposition="outside",
                               hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>"))
        fig.update_layout(height=360, margin=dict(l=0, r=20, t=10, b=0), xaxis=dict(range=[0, 110],
                          title="Percentile (blue = Combine engine, orange = in-game expression)"),
                          yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Acceleration-speed profile")
        b = bins[bins["nfl_id"] == p["nfl_id"]]
        v = config.AS_MIN_SPEED + (b["bin"] + 0.5) * config.AS_BIN
        fig = go.Figure()
        fig.add_scatter(x=v, y=b["a"], mode="markers", name="Game frames (top-2 per bin)",
                        marker=dict(color=ORANGE, size=7, line=dict(color="white", width=1)))
        vmax = np.nanmax([p.get("eng_S0", np.nan), p.get("game_S0", np.nan), 10])
        vv = np.linspace(0, vmax, 60)
        for pre, color, dash, label in (("game", ORANGE, "solid", "Game A-S"),
                                        ("eng", BLUE, "dash", "Combine A-S")):
            if pd.notna(p.get(f"{pre}_A0")):
                fig.add_scatter(x=vv, y=np.clip(p[f"{pre}_A0"] * (1 - vv / p[f"{pre}_S0"]), 0, None),
                                mode="lines", line=dict(color=color, dash=dash, width=2.5),
                                name=f"{label}: A0 {p[f'{pre}_A0']:.1f}, S0 {p[f'{pre}_S0']:.1f}")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Speed (yd/s)",
                          yaxis_title="Acceleration (yd/s²)", legend=dict(orientation="h", y=-0.25),
                          plot_bgcolor="rgba(0,0,0,0)", hovermode="closest")
        st.plotly_chart(fig, width="stretch")
        st.caption("The gap between the dashed (Combine) and solid (game) lines is engine left unused on Sundays.")

    st.subheader("Closest athletic comps")
    cp = comps(df, p["nfl_id"])
    show = ["display_name", "draft_year", "pick", "eng_S0", "game_vmax", "expr_speed",
            "snaps_per_season", "starts_per_season"]
    st.dataframe(cp[[c for c in show if c in cp]].round(2), hide_index=True, width="stretch")

# ------------------------------------------------------------------ calculator
with tab_calc:
    st.markdown("Enter a prospect's **official 10-yard split and 40 time**. The calculator fits the "
                "same sprint model to them, ranks the result against this position's pool (scored the "
                "same way from their official splits), and projects in-game peak speed from how much "
                "of their engine this position usually shows in games.")
    c = st.columns(3)
    cpos = c[0].selectbox("Position", config.POSITION_GROUPS, key="calc_pos")
    t10 = c[1].number_input("10-yard split (s)", 1.35, 2.10, 1.55, 0.01)
    t40 = c[2].number_input("40-yard dash (s)", 4.20, 5.60, 4.50, 0.01)
    e = engine_from_splits(t10, t40)
    ref = df[(df["pos_group"] == cpos)].dropna(subset=["ten_yd_split", "forty"]).copy()
    if ref.empty or not np.isfinite(e["S0"]):
        st.info("Not enough reference data for this position, or the splits are inconsistent.")
    else:
        fits = ref.apply(lambda r: pd.Series(engine_from_splits(r["ten_yd_split"], r["forty"])), axis=1)
        ref[["sp_S0", "sp_A0"]] = fits[["S0", "A0"]].to_numpy()
        ratio = (ref["game_vmax"] / ref["sp_S0"]).median()
        k = st.columns(4)
        k[0].metric("Top speed S0", f"{e['S0']:.2f} yd/s",
                    f"{(ref['sp_S0'] < e['S0']).mean():.0%} percentile", delta_color="off")
        k[1].metric("Max acceleration A0", f"{e['A0']:.1f} yd/s²",
                    f"{(ref['sp_A0'] < e['A0']).mean():.0%} percentile", delta_color="off")
        k[2].metric("Time constant τ", f"{e['tau']:.2f} s")
        k[3].metric("Projected game peak speed", f"{e['S0'] * ratio:.2f} yd/s" if np.isfinite(ratio) else "—",
                    f"{cpos} median expression {ratio:.0%}" if np.isfinite(ratio) else None, delta_color="off")
        fig = go.Figure()
        fig.add_scatter(x=ref["sp_S0"], y=ref["sp_A0"], mode="markers", name=f"{cpos} pool",
                        marker=dict(color=MUTED, size=7, opacity=0.6), text=ref["display_name"],
                        hovertemplate="%{text}<br>S0 %{x:.2f} · A0 %{y:.1f}<extra></extra>")
        fig.add_scatter(x=[e["S0"]], y=[e["A0"]], mode="markers", name="Your prospect",
                        marker=dict(color=BLUE, size=16, symbol="star", line=dict(color="white", width=1.5)))
        fig.update_layout(height=380, xaxis_title="Top speed S0 (yd/s)", yaxis_title="Max acceleration A0 (yd/s²)",
                          margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ value by position
with tab_value:
    outs = sorted(set(assoc["outcome"]), key=lambda o: list(OUTCOME_LABELS).index(o) if o in OUTCOME_LABELS else 99)
    o = st.selectbox("Outcome", outs, format_func=lambda x: OUTCOME_LABELS.get(x, x))
    adj = st.toggle("Adjust for draft slot (partial ρ)", value=True)
    a = assoc[assoc["outcome"] == o]
    val, qcol = ("prho", "pq") if adj else ("rho", "q")
    mat = a.pivot_table(index="feature", columns="pos_group", values=val)
    qm = a.pivot_table(index="feature", columns="pos_group", values=qcol).reindex_like(mat)
    mat = mat.reindex(columns=[p for p in config.POSITION_GROUPS if p in mat.columns])
    qm = qm.reindex_like(mat)
    text = mat.map(lambda v: "" if pd.isna(v) else f"{v:+.2f}") + qm.map(lambda q: "*" if q < 0.1 else "")
    fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=[FEATURE_LABELS.get(f, f) for f in mat.index],
                               colorscale=DIVERGING, zmin=-0.5, zmax=0.5, text=text.values,
                               texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=34 * len(mat) + 80, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Spearman ρ oriented so positive = better athlete → better outcome. * = Benjamini-Hochberg q < 0.10.")
    st.subheader("Out-of-fold predictive value")
    d = inc[inc["outcome"] == o]
    fig = go.Figure()
    for mname, color in zip(dict.fromkeys(d["model"]), (BLUE, ORANGE, AQUA)):
        g = d[d["model"] == mname]
        fig.add_bar(x=g["pos_group"], y=g["r2"], name=mname, marker_color=color,
                    error_y=dict(type="data", array=g["r2_sd"], color="#52514e"))
    fig.update_layout(barmode="group", height=340, yaxis_title="Out-of-fold R²",
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ reliability
with tab_rel:
    st.markdown("Test-retest ICC(3,1) between a player's first and second attempt of the same drill. "
                "A metric that does not repeat cannot be a trait.")
    st.dataframe(rel.round(3), hide_index=True, width="stretch")
