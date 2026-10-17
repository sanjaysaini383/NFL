"""End-to-end pipeline: raw competition CSVs -> features, tables, figures.

    combine2field                         # uses data/raw or the Kaggle mount
    combine2field --synthetic             # demo on generated data
    combine2field --data-dir D --work-dir W --repeats 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import combine_features as cf
from . import config, game_profile, io, outcomes, report, synthetic, translation


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def run(data_dir: Path, work: Path, n_repeats: int = 20, make_figures: bool = True,
        exposure: int | None = config.EXPOSURE_PLAYS) -> dict:
    t0 = time.time()
    tables = work / "tables"
    figs = work / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    base = io.player_base(data_dir)
    base = base[base["pos_group"].isin(config.POSITION_GROUPS)]
    ids = set(base["nfl_id"])
    log(f"players: {len(base)}", t0)

    ct = io.combine_tracking(data_dir, work)
    attempts = cf.attempt_features(ct)
    engine = cf.player_engine(attempts)
    rel = translation.reliability(cf.retest_pairs(attempts))
    log(f"combine attempts: {len(attempts)}, players with engine: {len(engine)}", t0)

    expression, bins, plays = game_profile.build(ids, data_dir, work, plays_per_player=exposure)
    log(f"game expression for {expression['game_A0'].notna().sum()} players "
        f"({exposure or 'all'} plays each)", t0)

    car = outcomes.career_outcomes(base)
    pp = io.player_play(data_dir, outcomes.PLAY_COLUMNS)
    play_out = outcomes.play_outcomes(pp)
    df = translation.assemble(base, engine, expression, car, play_out)
    log("assembled player table", t0)

    trans = translation.translation_table(df)
    assoc = translation.association_table(df)
    inc = translation.incremental_value(df, n_repeats=n_repeats)
    pass_rush = translation.pass_rush_table(df)
    log("statistics done", t0)

    df.to_parquet(tables / "players.parquet", index=False)
    bins.to_parquet(tables / "game_as_bins.parquet", index=False)
    attempts.to_parquet(tables / "combine_attempts.parquet", index=False)
    for name, t in {"reliability": rel, "translation": trans, "associations": assoc,
                    "incremental_value": inc, "pass_rush": pass_rush}.items():
        t.to_csv(tables / f"{name}.csv", index=False)

    if make_figures:
        ex = df.dropna(subset=["game_A0", "eng_A0"])
        if len(ex):
            pick = ex[ex["pos_group"] == "WR"] if (ex["pos_group"] == "WR").any() else ex
            report.fig_method(ct, bins, pick.sort_values("game_n_plays").iloc[-1], figs)
        report.fig_reliability(rel, figs)
        report.fig_translation(df, figs)
        if not assoc.empty:
            report.fig_heatmap(assoc, figs)
            report.fig_heatmap(assoc, figs, outcome="primary")
        if not pass_rush.empty and pass_rush["prho"].notna().any():
            report.fig_pass_rush(pass_rush, figs)
        if not inc.empty:
            report.fig_increment(inc, figs)
        log(f"figures -> {figs}", t0)

    summary = summarise(df, rel, trans, assoc, inc)
    if not pass_rush.empty:
        summary["pass_rush"] = pass_rush[["role", "feature", "n", "prho", "plo", "phi", "pp", "pq"]] \
            .round(3).to_dict("records")
    (work / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done -> {work}", t0)
    return summary


def summarise(df, rel, trans, assoc, inc) -> dict:
    s = {"n_players": int(len(df)),
         "n_by_position": df["pos_group"].value_counts().to_dict(),
         "n_with_engine": int(df["eng_S0"].notna().sum()),
         "n_with_expression": int(df["game_A0"].notna().sum())}
    s["exposure_check_rho_nplays"] = {
        c: round(float(df[[c, "game_n_plays_total"]].corr(method="spearman").iloc[0, 1]), 3)
        for c in ("game_vmax", "game_S0", "game_A0") if c in df and "game_n_plays_total" in df}
    s["median_speed_expression"] = df.groupby("pos_group")["expr_speed"].median().round(3).to_dict()
    s["top_reliability"] = rel.head(5)[["drill_name", "metric", "icc"]].to_dict("records")
    if not assoc.empty:
        sig = assoc[(assoc["pq"] < 0.10)].sort_values("pq")
        s["significant_beyond_draft"] = sig[["pos_group", "outcome", "feature", "prho", "pq"]] \
            .head(15).to_dict("records")
    if not inc.empty:
        piv = inc.pivot_table(index=["pos_group", "outcome"], columns="model", values="r2")
        s["incremental_r2"] = {f"{p}|{o}": {k: (None if np.isnan(v) else round(v, 3)) for k, v in row.items()}
                               for (p, o), row in piv.iterrows()}
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true", help="generate and analyse SYNTHETIC data")
    ap.add_argument("--repeats", type=int, default=20, help="CV repeats for incremental value")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--exposure", type=int, default=config.EXPOSURE_PLAYS,
                    help="random plays per player for in-game metrics (0 = use all plays)")
    a = ap.parse_args(argv)
    work = a.work_dir or config.work_dir()
    data_dir = a.data_dir or config.data_dir()
    if a.synthetic:
        work = a.work_dir or (config.ROOT / "outputs" / "synthetic")
        data_dir = synthetic.write(work / "raw", n_per_pos=30, plays_per_player=60)
    pd.set_option("display.width", 160)
    exposure = a.exposure or None
    if a.synthetic:
        exposure = min(exposure or 40, 40)
    summary = run(data_dir, work, a.repeats, not a.no_figures, exposure)
    print(json.dumps({k: summary[k] for k in ("n_players", "n_with_engine", "n_with_expression",
                                              "median_speed_expression")}, indent=2, default=float))


if __name__ == "__main__":
    main()
