# Engine vs. Expression: Does the Combine Engine Show Up on Sundays?

*NFL Big Data Bowl 2027 · Open Track (adjust if entering the student track)*

> **Draft status.** Methods and structure are final. Every `{{…}}` placeholder is filled from `results.json` and the tables produced when the notebook runs on the competition data. **Do not submit with placeholders, and do not quote numbers from the synthetic demo.** Keep within the official word and figure limits (check the competition Rules tab).

---

## 1. The idea in one picture

Scouts already know a prospect's 40 time. What they don't know is **how the speed is built**, and **whether the player will use it on Sundays**.

We fit **one physics model**, the linear **acceleration–speed (A–S) profile** `a(v) = A0 · (1 − v / S0)`, in two environments:

* **The engine (Combine).** The 10 Hz trace of the 40-yard dash is fitted with the sprint model `v(t) = S0 (1 − e^{−(t−t0)/τ})`. This gives top speed **S0**, the acceleration time constant **τ**, and max acceleration **A0 = S0/τ**.
* **The expression (games).** Every frame a player ran at ≥3 yd/s is binned by speed. The two highest *tangential* accelerations per 0.2 yd/s bin are kept, and a robust line yields the in-game **A0** and **S0** (the in-situ method of Morin et al., 2021).

Both profiles sit on the same axes, so the gap between the lines is the **engine a player leaves unused on Sundays** (Figure 1).

![Figure 1](../outputs/figures/fig1_method.png)
*Figure 1. One player, one model. Left: Combine 40 speed trace and sprint fit. Right: in-game A–S cloud and fit (orange) against the Combine-implied A–S line (dashed blue).*

We add **change-of-direction braking** from the 5-10-5 shuttle and three-cone. Each drill is split into turns at local speed minima, and we measure peak deceleration on entry, speed kept through the turn and re-acceleration on exit. The in-game counterpart is a player's 1st-percentile tangential deceleration.

## 2. Data and outcomes

| Source | Use |
|---|---|
| `combine_tracking.csv` (10 Hz, every attempt) | Engine: sprint fit (40), braking/turn features (shuttle, 3-cone) |
| `game_tracking_{2023-2025}.csv` | Expression: in-situ A–S profile, peak speed, peak braking |
| `combine_results.csv` | Stopwatch baseline (40, 10-yd split, jumps, agility, size) |
| `players.csv`, `player_career_successes.csv` | Draft slot; snaps and starts per available season |
| `player_play.csv` | Play-level outcomes: WR/TE separation at throw, OL pressure allowed, DL pressure rate, DB tackle rate |

Speed and tangential acceleration are rebuilt from x/y because the Combine `dis` field is unreliable and the supplied `a` mixes in centripetal acceleration. Career outcomes are divided by seasons available, so the 2025 class is not penalised for having one season.

## 3. Is it a trait? Reliability first

A metric that doesn't repeat between attempts can't separate athletes. Figure 2 ranks every tracking metric by test-retest ICC(3,1) between attempt 1 and attempt 2.

![Figure 2](../outputs/figures/fig2_reliability.png)

**Finding:** `{{top reliable metrics and ICCs}}`. `{{metrics below 0.5}}` are too noisy to carry scouting weight on their own, and we flag them in the app rather than hiding them.

## 4. Translation: how much of the engine is expressed?

Figure 3 compares each player's Combine S0 with his in-game peak speed (99.5th percentile of per-play maxima).

![Figure 3](../outputs/figures/fig3_translation.png)

**Finding:** median speed expression is `{{WR}}` for WR, `{{DB}}` for DB, `{{TE}}` for TE, `{{DL}}` for DL and `{{OL}}` for OL. Within a position, the Combine engine explains `{{ρ range}}` of the rank order of game speed. Engine sets the ceiling; role and usage decide how close a player gets to it.

## 5. What predicts success beyond draft slot?

Teams already priced the stopwatch into the draft. The scouting question is which metrics still predict success **after controlling for draft slot**. We report draft-adjusted partial Spearman correlations, oriented so that positive means *better athlete → better outcome*, with Benjamini–Hochberg control within each position.

![Figure 4](../outputs/figures/fig4_value_beyond_draft.png)

**Findings:**
1. `{{position-specific result 1}}`
2. `{{position-specific result 2}}`
3. `{{where nothing survives FDR, say so plainly}}`

## 6. Does tracking add out-of-fold predictive value?

Nested ridge models, 5-fold CV × 50 repeats, rank outcome: draft slot → + stopwatch Combine → + tracking engine & expression.

![Figure 5](../outputs/figures/fig5_incremental_value.png)

**Finding:** `{{ΔR² by position, with SD}}`.

## 7. The product: *Prospect Translator*

The analysis ships as a scouting tool (open-source Streamlit app plus Python package):

* **Prospect card**: engine vs. expression percentiles within position, the A–S overlay, and nearest athletic comps.
* **Stopwatch → engine calculator**: official 10-yd and 40 times are inverted to S0, A0 and τ for *any* prospect, including those without tracking. It ranks him against the position pool and projects in-game peak speed from that position's typical expression.
* **What matters by position**: the evidence behind Figures 4 and 5, filterable by outcome.

**How a club would use it:**
1. **Draft.** Prefer the engine components that survive the draft-adjusted test for the position; discount the ones that don't repeat.
2. **Development.** A player with a high engine and low expression is a usage or technique question, not an athletic one. That is a specific, testable coaching target.
3. **Pro scouting.** In-game A–S profiles update every week and can be compared directly with the Combine profile.

## 8. Limitations

About 100 players per group; the 2025 class has one season of outcomes. Expression mixes ability with role (slot receivers rarely reach top speed). Snaps measure opportunity as well as quality. All results are observational.

## References

* Morin, J.-B. et al. (2021). Individual acceleration-speed profile in-situ: a proof of concept in professional football players. *Journal of Biomechanics*.
* Samozino, P. et al. (2016). A simple method for measuring power, force, velocity properties, and mechanical effectiveness in sprint running. *Scand. J. Med. Sci. Sports*.
* Shrout, P. & Fleiss, J. (1979). Intraclass correlations: uses in assessing rater reliability. *Psychological Bulletin*.
* Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*.

*Code: https://github.com/sanjaysaini383/NFL · Notebook: `{{Kaggle notebook link}}`*
