# Engine vs. Expression — NFL Big Data Bowl 2027

> **Does the Combine engine show up on Sundays?**

An entry for the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027). It links 10 Hz NFL Combine tracking to in-game tracking and NFL success for WR, TE, DB, OL and DL.

We fit **one physics model**, the acceleration–speed profile `a(v) = A0·(1 − v/S0)`, in two places:

| | Where | What it measures |
|---|---|---|
| **Engine** | Combine 40-yd tracking, fitted with `v(t)=S0(1−e^{−(t−t0)/τ})`; braking and turns from the shuttle and three-cone | What the athlete *can* do |
| **Expression** | In-situ A–S profile from every game frame (Morin et al., 2021) | What he *does* on Sundays |

Then we ask, per position: Is the metric reliable (test-retest ICC)? How much of the engine is expressed? Which pieces predict snaps, starts and play-level success **beyond draft slot** (partial Spearman + FDR)? And does tracking add **out-of-fold** predictive value over the stopwatch Combine?

## What's in the repo

```
src/combine2field/      Python package
  combine_features.py   sprint fit, splits→engine solver, COD braking/turn features
  game_profile.py       in-situ A-S profile from 22.6M game rows (per-season reduction)
  outcomes.py           position-specific success definitions
  translation.py        reliability, translation, value-beyond-draft, incremental CV
  report.py             write-up figures
  pipeline.py           one-command CLI
  synthetic.py          SYNTHETIC schema-identical data for tests and demo
app/streamlit_app.py    "Prospect Translator" scouting app
notebooks/              self-contained Kaggle notebook (built by scripts/make_notebook.py)
writeup/WRITEUP.md      write-up draft (placeholders filled from real results)
tests/                  pytest suite (physics, stats, pipeline, app)
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[app,dev]"

# 1. Get the data (Kaggle account + accepted competition rules required)
kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw
unzip data/raw/nfl-big-data-bowl-2027.zip -d data/raw

# 2. Run the analysis -> outputs/tables, outputs/figures, outputs/results.json
combine2field --repeats 50

# 3. Open the scouting app
streamlit run app/streamlit_app.py
```

No data yet? `combine2field --synthetic` and the app's demo mode use **synthetic** data with the same schema. It contains no real players and must never be quoted as a result.

### On Kaggle

Run `python scripts/make_notebook.py`, upload `notebooks/engine-vs-expression.ipynb` to Kaggle, attach the competition data and run it. The package source is inlined, so it works with internet off. Derived tables stay in a temp directory; only figures are written to `/kaggle/working`.

## Prospect Translator (app)

* **Prospect card**: engine vs. expression percentiles, A–S overlay, nearest athletic comps
* **Stopwatch → engine calculator**: official 10-yd + 40 → S0, A0, τ, percentile and projected game speed for *any* prospect
* **What matters, by position**: draft-adjusted correlations and out-of-fold model comparison
* **Reliability**: which tracking metrics repeat across attempts

## Tests

```bash
pytest -q
```

The physics tests recover known sprint and A–S parameters. The stats tests check ICC, BH and partial correlation. The pipeline and app tests run end to end on synthetic data.

## Data & license

* Code: MIT (see `LICENSE`).
* Competition data: **CC BY-NC 4.0**, provided by the NFL via Kaggle. It is **not** in this repository and `.gitignore` blocks CSV/Parquet files. Use it only under the competition rules.
