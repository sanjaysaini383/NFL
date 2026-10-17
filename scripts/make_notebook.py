"""Build the self-contained Kaggle notebook: notebooks/engine-vs-expression.ipynb

The package source is inlined with %%writefile so the notebook runs on Kaggle
with internet off. Derived tables go to /tmp; only figures land in the public
/kaggle/working output, so no copy of the competition data is republished.

    python scripts/make_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "combine2field"
MODULES = ["__init__", "config", "io", "kinematics", "combine_features", "game_profile",
           "outcomes", "stats", "translation", "report", "synthetic", "pipeline"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Engine vs. Expression\n"
       "### Does the Combine engine show up on Sundays?\n\n"
       "One physics model, the **acceleration-speed (A-S) profile**, is fitted twice for every prospect: "
       "once to his 10 Hz Combine 40-yard dash (the *engine*) and once to every frame he ran in NFL games "
       "(the *expression*). Because both live on the same axes (max acceleration **A0**, top speed **S0**), "
       "we can measure how much of a player's engine he uses, and test which part predicts NFL success "
       "**beyond what draft slot already priced in**.\n\n"
       "Steps: (1) build the engine from Combine tracking, (2) build the expression from game tracking, "
       "(3) check test-retest reliability, (4) measure translation, (5) estimate value beyond draft slot, "
       "(6) out-of-fold incremental value.\n\n"
       "Source code, tests and the *Prospect Translator* app: https://github.com/sanjaysaini383/NFL"),
    md("## 0 · Package source (inlined so the notebook runs offline)"),
    code("import os, sys, tempfile\n"
         "TMP = tempfile.gettempdir().replace(os.sep, '/')\n"
         "SRC = f'{TMP}/c2f_src'\n"
         "os.makedirs(f'{SRC}/combine2field', exist_ok=True)\n"
         "sys.path.insert(0, SRC)"),
]
for m in MODULES:
    src = (PKG / f"{m}.py").read_text(encoding="utf8")
    c = code(f"%%writefile {{SRC}}/combine2field/{m}.py\n{src}")
    c.metadata["jupyter"] = {"source_hidden": True}
    cells.append(c)

cells += [
    md("## 1 · Run the pipeline\n"
       "Reads the competition files from `/kaggle/input`. Derived tables are written to a temp directory "
       "(not published); figures are copied to `/kaggle/working/figures`."),
    code("import os, shutil, json\nfrom pathlib import Path\n"
         "from combine2field import config, pipeline\n"
         "DATA = config.data_dir()\n"
         "SYNTH = not (DATA / 'players.csv').exists()\n"
         "if SYNTH:\n"
         "    print('Competition data not found -> running on SYNTHETIC data (demo only)')\n"
         "    from combine2field import synthetic\n"
         "    DATA = synthetic.write(Path(TMP) / 'c2f_raw', n_per_pos=30, plays_per_player=60)\n"
         "WORK = Path(TMP) / 'c2f'\n"
         "summary = pipeline.run(DATA, WORK, n_repeats=int(os.environ.get('C2F_REPEATS', 50)))\n"
         "PUBLIC = Path('/kaggle/working') if config.on_kaggle() else Path(TMP) / 'c2f_public'\n"
         "shutil.copytree(WORK / 'figures', PUBLIC / 'figures', dirs_exist_ok=True)\n"
         "print(json.dumps({k: summary[k] for k in ('n_players','n_with_engine','n_with_expression')}, indent=2))"),
    code("import pandas as pd\nfrom IPython.display import Image, display\n"
         "pd.set_option('display.precision', 3)\n"
         "T = WORK / 'tables'; F = WORK / 'figures'\n"
         "players = pd.read_parquet(T / 'players.parquet')\n"
         "rel = pd.read_csv(T / 'reliability.csv'); trans = pd.read_csv(T / 'translation.csv')\n"
         "assoc = pd.read_csv(T / 'associations.csv'); inc = pd.read_csv(T / 'incremental_value.csv')"),
    md("## 2 · One model, two environments\n"
       "Left: the 40 is fitted with $v(t)=S_0\\,(1-e^{-(t-t_0)/\\tau})$, which implies the linear profile "
       "$a(v)=A_0(1-v/S_0)$ with $A_0=S_0/\\tau$. Right: every game frame at ≥3 yd/s is binned by speed; "
       "the two highest tangential accelerations per 0.2 yd/s bin are kept and a robust line gives the "
       "in-game $A_0$ and $S_0$ (Morin et al., 2021)."),
    code("display(Image(F / 'fig1_method.png'))"),
    md("## 3 · Reliability: which tracking metrics are traits?\n"
       "A metric that does not repeat between attempt 1 and attempt 2 cannot be a trait. "
       "ICC(3,1) ≥ 0.75 is conventionally *good*."),
    code("display(Image(F / 'fig2_reliability.png'))\nrel.head(15)"),
    md("## 4 · Translation: how much of the engine is expressed?"),
    code("display(Image(F / 'fig3_translation.png'))\n"
         "players.groupby('pos_group')[['expr_speed','expr_accel','expr_brake']].median()"),
    code("trans.pivot_table(index=['combine','game'], columns='pos_group', values='rho')"),
    md("## 5 · Value beyond draft slot\n"
       "Spearman ρ between each metric and snaps per season, after removing the rank-linear effect of "
       "log(draft pick). Signs are oriented so positive = better athlete → more playing time. "
       "Benjamini-Hochberg q-values control false discoveries within each position."),
    code("display(Image(F / 'fig4_value_beyond_draft.png'))\n"
         "assoc[assoc['pq'] < 0.10].sort_values('pq')[['pos_group','outcome','feature','n','rho','prho','pq']]"),
    md("## 6 · Do tracking metrics add out-of-fold predictive value?\n"
       "Nested ridge models, 5-fold CV repeated 50 times, rank-transformed outcome: "
       "draft slot → + stopwatch Combine → + tracking engine & expression."),
    code("display(Image(F / 'fig5_incremental_value.png'))\n"
         "inc.pivot_table(index=['pos_group','outcome'], columns='model', values='r2')"),
    md("## 7 · Limitations\n"
       "* Sample sizes are about 100 players per group, and the 2025 class has one season of outcomes.\n"
       "* Game expression depends on role: a slot WR rarely reaches top speed, so expression mixes ability "
       "with usage. We report it separately from the engine rather than as a single verdict.\n"
       "* Snaps and starts measure opportunity as well as quality; play-level outcomes are added where "
       "the data supports them.\n"
       "* All associations are observational."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
out = ROOT / "notebooks" / "engine-vs-expression.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
