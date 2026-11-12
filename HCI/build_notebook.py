import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


# ============================================================ 0. Title / map
md(r"""
# Results and Discussion Pipeline
**Explicit and Implicit Navigation Guidance Effects on Player Wayfinding in Third-Person Open-World Games**
Narvasa, Encallado, Ng - De La Salle University

> **DISCLAIMER - DUMMY DATA.** Data collection is not finished. Every number, table and figure in this notebook is computed from
> **generated dummy data** (16 invented participants, with deliberate data-entry faults injected so the cleaning can be shown working).
> Nothing here is a study result. Figures carry a "DUMMY DATA" watermark; switch it off with `IS_DUMMY = False` when the real data arrive.

**What this notebook does.** It implements the analysis plan of the paper (Section *Analysis plan*) end to end:

| Stage | Section | Paper reference |
|---|---|---|
| Load raw data and audit it | 1 | Coding and data handling |
| Clean it with a logged, rule-based process | 2 | Coding and data handling, Limitations |
| Prepare analysis-ready tables (one row per trial, one row per participant) | 3 | Measures 1-6; "one row per participant, both trials side by side" |
| Describe participants and the counterbalanced design | 4 | Participants; Study design |
| Results by research question (RQ2, RQ3, RQ4), order/route checks, individual cases, interviews, integration, coder reliability | 5 | Analysis plan, Steps 1-3 |
| Discussion scaffold generated from the numbers | 6 | Limitations |
| Software tests and reproducibility | 7-8 | - |

**Design principle (from the paper).** The study is a contextual inquiry, so quantitative analysis is **descriptive**: medians and ranges, paired plots,
direction counts and estimation of within-participant differences with bootstrap intervals. **No significance tests are run** and no causal language is used.
An optional, clearly labelled sensitivity analysis is included but switched off (Section 5.4).

All logic lives in `pipeline.py` and `plots.py` and is covered by `test_pipeline.py` (Section 7); this notebook only calls it.
""")

md(r"""
### Assumptions to confirm before the real data go in
These are the only places where the code had to assume something the paper leaves open. Each is a one-line change in `pipeline.py`.

1. **Decision points per route** `ROUTE_DECISION_POINTS = {Route_A: 4, Route_B: 4}` is a **placeholder**. The paper only requires D >= 3; the pilot fixes the real value. Used for the rule 0 <= errors <= D and for the error rate.
2. **Route naming:** `Route_A` = R1 and `Route_B` = R2 in the paper's sequence figure.
3. **UEQ-S item direction:** all eight items are assumed to run negative to positive (`UEQS_REVERSED_ITEMS = ()`). Add item numbers if your form reverses any.
4. **Compass use** is coded from the interview notes into 0 none, 1 brief, 2 frequent, 3 relied on (the paper says compass use cannot be coded from video).
5. **Interview codebook and candidate themes** (`CODEBOOK`, `THEMES`) follow the two information-source groups and GQ1-GQ3 in the paper; the researchers' own themes replace the candidates after coding.
6. **Second-coder file** `reliability_*.csv` has one row per coded unit (per decision point for the error code, per trial for counts), with both coders' values.
""")

# ============================================================ Setup
md("## 0. Setup")
code(r"""
import hashlib, platform, subprocess, sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import pipeline as pl
import plots as pp

# ---- configuration -----------------------------------------------------------
DATA_DIR = Path(".")          # folder with the CSV files
SUFFIX = "_dummy"             # "" for the real files: participants.csv, trials.csv, ...
IS_DUMMY = True               # True -> watermark every figure as dummy data
RUN_EXPLORATORY_TESTS = False # the paper commits to NO significance tests; keep False unless amended
SEED = 2026                   # bootstrap seed
OUT = {k: Path(k) for k in ("figures", "tables", "processed")}
for p in OUT.values():
    p.mkdir(exist_ok=True)

pp.DUMMY_WATERMARK = IS_DUMMY
pp.OUTDIR = OUT["figures"]
pd.set_option("display.max_columns", 60, "display.width", 200, "display.max_colwidth", 80)
np.random.seed(SEED)
print(f"pandas {pd.__version__} | numpy {np.__version__} | matplotlib {matplotlib.__version__} | python {platform.python_version()}")
""")

# ============================================================ 1 Load
md(r"""
## 1. Load and audit the raw data
Files are read **as text** (`dtype=str`) so that nothing is silently converted before the cleaning rules see it: `"n/a"`, `"105"` or `" explicit "` stay visible.
""")
code(r"""
raw = pl.load_raw(DATA_DIR, SUFFIX)
audit = pd.DataFrame({
    "table": list(raw),
    "rows": [len(v) for v in raw.values()],
    "columns": [v.shape[1] for v in raw.values()],
    "blank cells": [int((v.apply(lambda c: c.str.strip() == "")).sum().sum()) for v in raw.values()],
    "NA-token cells (n/a, -, null ...)": [int(v.apply(lambda c: c.str.strip().str.lower().isin(pl.NA_TOKENS - {""})).sum().sum()) for v in raw.values()],
    "exact duplicate rows": [int(v.duplicated().sum()) for v in raw.values()],
})
display(audit)
pp.export_table(audit, "table_raw_audit", "Raw data audit", image=False)
display(raw["trials"].head(4))
""")

# ============================================================ 2 Cleaning
md(r"""
## 2. Data cleaning
**Policy.** Every problem is handled by an explicit, logged rule. Nothing is changed silently and nothing untrustworthy enters an analysis:

| Action | When | Examples |
|---|---|---|
| **repaired** | the fix is deterministic and safe | `p9` -> `P09`; `" explicit "` -> `Explicit`; `TRUE`/`Yes` -> `True`; exact duplicate row dropped; `n/a` or `-` read as missing; completed trial whose active time exceeds the 480 s cap -> counted as not completed (paper, measure 1) |
| **set_missing** | a value cannot be trusted | non-numeric, out of range (TLX outside 0-100, UEQ-S outside 1-7), non-integer count, errors > decision points, pause >= elapsed time, map time with 0 openings, map time > trial |
| **quarantined** | a row cannot be used or the correct version cannot be decided | unknown or ineligible participant (age < 18, no third-person 3D play), invalid condition/route, conflicting duplicates, broken trial design (not one trial per condition on different routes, not one of the four sequences) |
| **flagged** | plausible but worth a look; kept | outliers (modified z > 3.5), straight-lined UEQ-S, stopped before the cap, weekly hours > 80, coder 1 differing from the trials table |

Outliers are **flagged, not removed**: with n = 16 an extreme value may be a real player (paper: Limitations).
""")
code(r"""
clean = pl.clean_all(raw)                       # cleaning in dependency order, nothing mutated in `raw`
display(clean.row_counts)
summary = clean.log.summary()
display(summary)
""")
code(r"""
fig = pp.fig_cleaning(clean)
pp.save_fig(fig, "fig_cleaning_summary"); plt.show()
""")
md("**Every repaired, set-missing and quarantined case** (the audit trail; the full log with flags is exported to `processed/cleaning_log.csv`):")
code(r"""
log = clean.log.to_frame()
display(log[log["action"].isin(["repaired", "set_missing", "quarantined"])]
        .drop(columns=["detail"]).reset_index(drop=True))
pp.export_table(summary, "table_cleaning_summary", "Cleaning actions by rule", image=False)
""")
md("**Quarantined rows** are kept (never deleted) so that each can be checked against the recordings:")
code(r"""
for t, q in clean.quarantine.items():
    print(f"{t}: {len(q)} row(s)")
    display(q.drop(columns=["_row"], errors="ignore"))
if not clean.quarantine:
    print("Nothing quarantined.")
if clean.excluded_participants:
    print("Participants excluded for a broken trial design:", clean.excluded_participants)
""")
md(r"""
### 2.1 Design integrity
The paper's design crosses condition order with route pairing into four sequences (4 participants each). The check below verifies that every retained participant played **one trial per condition on different routes**,
that the pair forms one of the four sequences, and that the sequences are balanced.
""")
code(r"""
long, wide = pl.prepare(clean)                  # preparation is Section 3; needed here for the sequence label
seq = pl.sequence_table(wide)
display(seq)
print("Design violations found:", clean.excluded_participants or "none")
print("Balanced sequences:", bool((seq['Participants'] == len(wide) / 4).all()))
pp.export_table(seq, "table_sequences", "Counterbalanced sequences", col_widths=[1.4, 3.4, 1.4, 1.0])
""")
md("### 2.2 Outlier flags and remaining missing values")
code(r"""
pl.flag_outliers(long, ["completion_time_sec", "wayfinding_errors", "map_openings", "nasa_tlx_raw", "ueqs_overall"], clean.log)
log = clean.log.to_frame()
display(log[(log["action"] == "flagged")].drop(columns=["csv_row"]).reset_index(drop=True))
miss = (long.groupby("condition", observed=True)[list(pl.MEASURES)].apply(lambda g: g.isna().sum()).T
        .rename(columns=lambda c: f"missing, {c}").reset_index().rename(columns={"index": "measure"}))
print("Missing values left in the analysis table (each traced to a logged cleaning action):")
display(miss)
""")

# ============================================================ 3 Preparation
md(r"""
## 3. Data preparation
Derived variables follow the paper's definitions exactly (Section *Conditions and measures*).

$$T_{p,m} = \min\Big(t_{\text{end}} - t_{\text{start}} - \sum \text{pauses},\; 480\Big),\qquad T_{p,m}=480 \text{ and not completed if the cap is reached}$$
$$W_{p,m} = \tfrac{1}{6}\sum_{k=1}^{6} s_k \quad (\text{raw NASA-TLX}),\qquad U_{p,m} = \tfrac{1}{8}\sum_{i=1}^{8} (r_i - 4)\ \ (\text{UEQ-S}, -3\ldots{+3}),\ \ \text{pragmatic: items 1-4, hedonic: 5-8}$$
$$\text{error rate} = E_{p,m} / D_{\text{route}},\qquad \text{map time share} = \frac{\text{map open duration}}{T_{p,m}}$$

Two analysis tables are produced:
* **`analysis_long`** - one row per trial (32 rows for 16 participants), for plots and by-condition summaries.
* **`analysis_wide`** - one row per participant with `<measure>_X`, `<measure>_I` and `<measure>_diff` (= I - X) side by side, as stated in the paper.

Unnecessary fields are dropped (raw UEQ-S items after scoring, free-text compass notes after coding, `trial_number` duplicate of `trial_position`, the screening answer that is constant after eligibility filtering). Types are set for analysis: ordered categoricals, nullable integers for counts, booleans.
""")
code(r"""
display(long.head(6))
display(long.dtypes.rename("dtype").to_frame().T)
print("analysis_long :", long.shape, "| analysis_wide :", wide.shape)
""")
code(r"""
DICT = [
 ("participant_id", "Anonymous ID (P01...)", "-", "text"),
 ("condition", "Guidance condition: Explicit = Guided Exploration on (X), Implicit = off (I)", "-", "ordered category"),
 ("route_id", "Route played (Route_A = R1, Route_B = R2)", "-", "ordered category"),
 ("trial_position", "Order of the trial in the session (1 or 2)", "-", "integer"),
 ("active_time_sec", "Elapsed time minus combat and cutscene pauses", "s", "float"),
 ("completion_time_sec", "T: active time capped at 480 s; not-completed trials = 480", "s", "float"),
 ("completed", "Objective reached before the cap", "-", "boolean"),
 ("timed_out", "Not completed (cap reached)", "-", "boolean"),
 ("wayfinding_errors", "E: wrong branches entered and stayed on > 10 s, 0 <= E <= D", "count", "integer"),
 ("error_rate", "E divided by the route's decision points D", "proportion", "float"),
 ("lost_episodes", "L: periods of not knowing where to go (interview + recording)", "count", "integer"),
 ("map_openings", "Number of map openings coded from recording", "count", "integer"),
 ("map_open_duration_sec", "Total time the map was open", "s", "float"),
 ("map_time_share", "Map open duration divided by completion time", "proportion", "float"),
 ("objective_board_openings", "Objective-board / quest-log openings", "count", "integer"),
 ("compass_use", "Interview-coded compass use: 0 none, 1 brief, 2 frequent, 3 relied on", "ordinal 0-3", "integer"),
 ("mental_demand", "NASA-TLX subscale", "0-100", "float"),
 ("physical_demand", "NASA-TLX subscale", "0-100", "float"),
 ("temporal_demand", "NASA-TLX subscale", "0-100", "float"),
 ("performance", "NASA-TLX subscale (0 = perfect, 100 = failure)", "0-100", "float"),
 ("effort", "NASA-TLX subscale", "0-100", "float"),
 ("frustration", "NASA-TLX subscale", "0-100", "float"),
 ("nasa_tlx_raw", "W: unweighted mean of the six subscales", "0-100", "float"),
 ("ueqs_pragmatic", "UEQ-S items 1-4, scored -3...+3", "-3 to +3", "float"),
 ("ueqs_hedonic", "UEQ-S items 5-8, scored -3...+3", "-3 to +3", "float"),
 ("ueqs_overall", "UEQ-S mean of all eight items", "-3 to +3", "float"),
 ("age", "Participant age", "years", "integer"),
 ("gender", "Self-reported gender", "-", "category"),
 ("weekly_gaming_hours", "Gaming hours per week", "hours", "float"),
 ("years_of_gaming", "Years of playing games", "years", "float"),
 ("open_world_game_experience", "Has open-world game experience", "-", "boolean"),
 ("previous_assassins_creed_play", "Has played another Assassin's Creed game", "-", "boolean"),
 ("input_device", "Keyboard and mouse or gamepad (fixed per participant)", "-", "category"),
 ("sequence", "Counterbalanced sequence 1-4 (from trial 1's condition and route)", "-", "integer"),
]
data_dictionary = pd.DataFrame(DICT, columns=["field", "definition", "unit", "type"])
undocumented = set(long.columns) - set(data_dictionary["field"])
assert not undocumented, f"undocumented fields: {undocumented}"
data_dictionary["dtype in table"] = data_dictionary["field"].map(lambda f: str(long[f].dtype))
display(data_dictionary)
pp.export_table(data_dictionary, "table_data_dictionary", "Data dictionary: analysis_long", image=False)
""")

# ============================================================ 4 Participants
md("## 4. Participants and design")
code(r"""
parts_tbl = pl.participant_table(clean.participants)
display(parts_tbl)
pp.export_table(parts_tbl, "table_participants", "Table 1. Participant characteristics", col_widths=[2.2, 6.4], wrap=70)
""")

# ============================================================ 5 Results
md(r"""
## 5. Results
Per the analysis plan, **Step 1** reports for each measure the median and range per condition ($n=16$), mean and SD, and the number of participants whose value is higher under X, higher under I, or equal.
Paired plots show every participant. RQ1 is answered from the literature (paper, Sections *Forms of guidance* and *Measures*) and needs no data.
Conditions: **X** = more explicit (Guided Exploration on), **I** = more implicit (off).
""")

# ---- 5.1 RQ2
md(r"""
### 5.1 RQ2 - Which information sources do players use, and how?
Measures: map openings, map open duration, objective-board openings (coded from recordings); compass use (interview-coded); interview codes for information sources (GQ1).
""")
code(r"""
rq2 = pl.describe_by_condition(long, wide, pl.RQ_MEASURES["RQ2"])
t_rq2 = pl.format_descriptives(rq2)
display(t_rq2)
pp.export_table(t_rq2, "table_rq2_information_use", "Table 2. Information use by condition (RQ2)",
                col_widths=[2.4, 1.8, 1.4, 1.8, 1.4, 2.0, 0.9], note="X = more explicit, I = more implicit.")
fig = pp.fig_paired(wide, pl.RQ_MEASURES["RQ2"], "Information use by condition (RQ2)")
pp.save_fig(fig, "fig_rq2_information_use"); plt.show()
""")
code(r"""
share = pl.describe_by_condition(long, wide, ["map_time_share"])
print("Share of the trial spent with the map open (proportion of completion time):")
display(share[["label", "median_X", "min_X", "max_X", "median_I", "min_I", "max_I"]].round(3))

comp = (long.assign(compass=long["compass_use"].map(pl.COMPASS_LABELS))
        .pivot_table(index="compass", columns="condition", values="participant_id", aggfunc="count", observed=False, fill_value=0)
        .reindex(list(pl.COMPASS_LABELS.values())).fillna(0).astype(int).reset_index())
comp.columns = ["Compass use (interview-coded)", "X (trials)", "I (trials)"]
display(comp)
pp.export_table(comp, "table_rq2_compass", "Compass use, coded from interview notes", col_widths=[3, 1.4, 1.4], font=10)
""")
code(r"""
interviews = clean.tables["interviews"]
counts = pl.code_participant_counts(interviews, len(clean.participants))
src_codes = [c for c, v in pl.CODEBOOK.items() if v[0] == "source"]
fig = pp.fig_code_counts(counts, src_codes, "Information sources mentioned in interviews (GQ1)", len(clean.participants))
pp.save_fig(fig, "fig_rq2_sources_interviews"); plt.show()
""")

# ---- 5.2 RQ3
md(r"""
### 5.2 RQ3 - How does wayfinding behaviour look, and what explains it?
Measures: completion time $T$ (capped at 480 s), wayfinding errors $E$, lost episodes $L$, completion count and timeouts.
""")
code(r"""
rq3 = pl.describe_by_condition(long, wide, pl.RQ_MEASURES["RQ3"])
t_rq3 = pl.format_descriptives(rq3)
display(t_rq3)
pp.export_table(t_rq3, "table_rq3_wayfinding", "Table 3. Wayfinding behaviour by condition (RQ3)",
                col_widths=[2.4, 1.8, 1.4, 1.8, 1.4, 2.0, 0.9], note="X = more explicit, I = more implicit. Not-completed trials count as 480 s.")
ct = pl.completion_table(long)
display(ct)
pp.export_table(ct, "table_rq3_completion", "Completion and timeouts per condition", col_widths=[1.6, 1.0, 1.2, 1.2, 1.8], font=10)
""")
code(r"""
fig = pp.fig_paired(wide, pl.RQ_MEASURES["RQ3"], "Wayfinding behaviour by condition (RQ3)", long=long, cap_line=True)
pp.save_fig(fig, "fig_rq3_wayfinding"); plt.show()
""")
code(r"""
er = pl.describe_by_condition(long, wide, ["error_rate"])
print("Errors per decision point (route-normalised):")
display(er[["label", "median_X", "min_X", "max_X", "median_I", "min_I", "max_I", "n_higher_X", "n_higher_I", "n_equal"]].round(2))
""")

# ---- 5.3 RQ4
md(r"""
### 5.3 RQ4 - How do players describe workload, difficulty and experience?
Measures: raw NASA-TLX (overall and six subscales) and UEQ-S (overall, pragmatic, hedonic).
""")
code(r"""
rq4 = pl.describe_by_condition(long, wide, pl.RQ_MEASURES["RQ4"])
t_rq4 = pl.format_descriptives(rq4)
display(t_rq4)
pp.export_table(t_rq4, "table_rq4_workload_experience", "Table 4. Workload and user experience by condition (RQ4)",
                col_widths=[2.4, 1.8, 1.4, 1.8, 1.4, 2.0, 0.9], note="X = more explicit, I = more implicit. UEQ-S scored -3 to +3.")
fig = pp.fig_paired(wide, pl.RQ_MEASURES["RQ4"], "Workload and user experience by condition (RQ4)", ncols=4)
pp.save_fig(fig, "fig_rq4_workload_experience"); plt.show()
""")
code(r"""
fig = pp.fig_tlx_subscales(long)
pp.save_fig(fig, "fig_rq4_tlx_subscales"); plt.show()
sub = pl.describe_by_condition(long, wide, pl.TLX_COLS)
sub["label"] = sub["label"].str.replace("Performance", "Performance*")
t_sub = pl.format_descriptives(sub.assign(unit="0-100"))
display(t_sub)
pp.export_table(t_sub, "table_rq4_tlx_subscales", "NASA-TLX subscales by condition", col_widths=[2.4, 1.8, 1.4, 1.8, 1.4, 2.0, 0.9],
                note="*Performance: 0 = perfect, 100 = failure.")
""")
code(r"""
inter = [c for c, v in pl.CODEBOOK.items() if v[0] in ("difficulty", "experience")]
fig = pp.fig_code_counts(counts, inter, "Difficulty and experience codes from interviews (GQ2, GQ3)", len(clean.participants))
pp.save_fig(fig, "fig_rq4_interview_codes"); plt.show()
""")

# ---- 5.4 differences
md(r"""
### 5.4 Size and consistency of within-participant differences
Because each participant is their own comparison, the difference $D_p = \text{value}(I) - \text{value}(X)$ is the natural unit. We report it as **estimation**: median and mean difference,
a percentile **bootstrap 95% interval** for the mean (10,000 resamples, fixed seed) and the paired standardised effect $d_z$. These describe the sample; they are not significance tests and carry no causal claim.
""")
code(r"""
all_measures = list(pl.MEASURES)
diffs = pl.paired_differences(wide, all_measures, n_boot=10_000, seed=SEED)
t_diff = diffs.assign(**{c: diffs[c].round(2) for c in ["median_diff", "mean_diff", "ci95_low", "ci95_high", "dz"]})
t_diff = t_diff.rename(columns={"label": "Measure", "n_pairs": "Pairs", "median_diff": "Median I-X", "mean_diff": "Mean I-X",
                                "ci95_low": "95% CI low", "ci95_high": "95% CI high", "dz": "d_z"}).drop(columns="measure")
display(t_diff)
pp.export_table(t_diff, "table_paired_differences", "Table 5. Within-participant differences (I minus X)", col_widths=[2.6, 0.8, 1.2, 1.2, 1.2, 1.2, 0.9],
                note="Bootstrap interval for the mean difference; descriptive estimation, not a significance test.")
fig = pp.fig_differences(wide, all_measures, diffs)
pp.save_fig(fig, "fig_paired_differences"); plt.show()
""")
code(r"""
if RUN_EXPLORATORY_TESTS:
    # NOT part of the paper's analysis plan. Reported only if the plan is amended.
    wx = pl.exploratory_wilcoxon(wide, all_measures)
    display(wx.round(4))
else:
    print("Exploratory Wilcoxon tests are OFF (RUN_EXPLORATORY_TESTS = False): the paper commits to descriptive analysis only.\n"
          "To use them, amend the Analysis plan first, then switch the flag on; Holm correction is applied (pl.exploratory_wilcoxon).")
""")

# ---- 5.5 order and route
md(r"""
### 5.5 Trial position and route (learning and route checks)
The paper reports results "by trial position and route, descriptively". Counterbalancing puts each condition in each position and on each route equally often, so any learning or route effect is spread across conditions; the table and figure show how large it looks.
""")
code(r"""
cols = ["completion_time_sec", "wayfinding_errors", "lost_episodes"]
by_pos = pl.describe_by_group(long, "trial_position", cols)
by_route = pl.describe_by_group(long, "route_id", cols)
for name, t in (("By trial position (median [min, max])", by_pos), ("By route (median [min, max])", by_route)):
    print(name); display(t)
nice = {"completion_time_sec": "Completion time (s)", "wayfinding_errors": "Errors", "lost_episodes": "Lost episodes"}
pp.export_table(by_pos.rename(columns=nice), "table_by_trial_position", "Median [min, max] by trial position and condition", col_widths=[1.2, 1.3, 0.6, 2.0, 1.6, 1.8], font=9)
pp.export_table(by_route.rename(columns=nice), "table_by_route", "Median [min, max] by route and condition", col_widths=[1.2, 1.3, 0.6, 2.0, 1.6, 1.8], font=9)
fig = pp.fig_order_route(long)
pp.save_fig(fig, "fig_order_route"); plt.show()
""")

# ---- 5.6 individual cases
md(r"""
### 5.6 Individual cases and covariates
Covariates (experience, device) are used **to describe participants and read individual cases, not to adjust the measures** (paper, *Recorded covariates*).
We list participants whose pattern differs from the majority, and show the I - X difference next to experience. Spearman $\rho$ is given without a p-value.
""")
code(r"""
cc = pl.counter_cases(wide, ["completion_time_sec", "wayfinding_errors", "nasa_tlx_raw", "ueqs_overall"])
display(cc)
pp.export_table(cc.round(2), "table_counter_cases", "Participants not following the majority direction", image=False)
fig = pp.fig_covariates(wide)
pp.save_fig(fig, "fig_covariates"); plt.show()
cov = pl.covariate_table(wide, ["completion_time_sec", "wayfinding_errors", "nasa_tlx_raw"], ["weekly_gaming_hours", "years_of_gaming"])
display(cov.round(2))
""")

# ---- 5.7 thematic analysis
md(r"""
### 5.7 Interviews: thematic analysis (Analysis plan, Step 2)
Two researchers independently code the transcripts following Braun and Clarke's phases; codes start from the two groups of information sources in the paper (external aids; environment and narrative) and are extended with emergent codes. The table below is the **codebook** with how many participants mentioned each code under each condition. Excerpts in the dummy file are placeholders.
""")
code(r"""
cb = counts.copy()
cb["Group"] = cb["group"]; cb["Code"] = cb["label"]
cb["Participants X / I"] = cb["participants_X"].astype(str) + " / " + cb["participants_I"].astype(str)
cb["Segments X / I"] = cb["segments_X"].astype(str) + " / " + cb["segments_I"].astype(str)
cb = cb.sort_values(["family", "Group", "participants_I"], ascending=[True, True, False])[["Group", "Code", "Participants X / I", "Segments X / I"]]
display(cb)
pp.export_table(cb, "table_codebook_counts", f"Interview codes (n = {len(clean.participants)} participants)", col_widths=[2.6, 3.0, 1.8, 1.6], font=9)
themes = pl.theme_participants(interviews)
t_themes = themes.assign(**{"Participants X / I": themes["participants_X"].astype(str) + " / " + themes["participants_I"].astype(str),
                            "Segments X / I": themes["segments_X"].astype(str) + " / " + themes["segments_I"].astype(str)})[["Theme", "Codes", "Participants X / I", "Segments X / I"]]
display(t_themes)
pp.export_table(t_themes, "table_themes", "Candidate themes (placeholders until coding is finished)", col_widths=[3.0, 3.6, 1.6, 1.5], wrap=46, font=9)
""")

# ---- 5.8 integration
md(r"""
### 5.8 Integration (Analysis plan, Step 3)
For each of RQ2 to RQ4: the descriptive pattern, the supporting or contradicting interview themes, and the related work. A pattern with no coded theme is reported as **unexplained**; an interview theme with no counterpart in the numbers would be reported the same way.
""")
code(r"""
desc_all = pl.describe_by_condition(long, wide, all_measures)
jd = pl.joint_display(desc_all, themes)
for _, r in jd.iterrows():
    print(f"=== {r['RQ']} ===\nPattern:\n  " + r["Descriptive pattern"].replace("\n", "\n  ") +
          "\nThemes:\n  " + r["Interview themes"].replace("\n", "\n  ") + "\nRelated work:\n  " + r["Related work"] + "\n")
pp.export_table(jd, "table_joint_display", "Table 6. Integration of numbers, interview themes and related work",
                col_widths=[0.6, 4.4, 3.6, 3.6], wrap=52, font=8)
""")

# ---- 5.9 reliability
md(r"""
### 5.9 Coder reliability
One researcher codes all recordings; a second independently codes a random quarter of the participants. Agreement on the per-decision-point **error code** is reported with **Cohen's kappa**; the other counts with **percent agreement** (paper, *Coding and data handling*).
Coder 1's values are also checked against the trials table so the two files cannot drift apart.
""")
code(r"""
rel = clean.tables["reliability"]
n_double = rel["participant_id"].nunique()
print(f"Double-coded participants: {n_double} of {len(clean.participants)} (target: a quarter = {len(clean.participants)//4}) -> {sorted(rel['participant_id'].unique())}")
mismatch = pl.reliability_consistency(rel, clean.trials, clean.log)
print("Coder 1 differs from trials table:", "none" if mismatch.empty else mismatch.to_dict("records"))
rs = pl.reliability_summary(rel)
show = rs.assign(value=rs["value"].round(2), percent_agreement=rs["percent_agreement"].round(1),
                 within_1_pct=rs["within_1_pct"].round(1), mean_abs_diff=rs["mean_abs_diff"].round(2))
display(show)
pp.export_table(show.fillna(""), "table_reliability", "Table 7. Inter-coder agreement", col_widths=[3.6, 1.6, 0.7, 0.8, 1.2, 1.0, 1.0, 1.4], wrap=40, font=8.5)
fig = pp.fig_reliability(rs)
pp.save_fig(fig, "fig_reliability"); plt.show()
""")

# ============================================================ 6 Discussion scaffold
md(r"""
## 6. Discussion scaffold
Sentences below are **rebuilt from the data**, so wording cannot drift from the numbers, and use descriptive language only. With real data, the Discussion should follow this structure:

1. **Summary by research question** (RQ2, RQ3, RQ4): the descriptive pattern, in the paper's terms (*more explicit* / *more implicit*).
2. **Convergence and divergence:** where recordings, questionnaires, interviews and related work agree, and where a number has no explanation (or the reverse).
3. **Reading time with care:** longer completion time may mean exploration, not difficulty (Lankes et al., 2026), so time is read together with errors, lost episodes, workload and interviews.
4. **Relation to prior work:** Xiao (2020) on explicit and implicit cues; Iacovides et al. (2015) on expertise; Gardony et al. (2013) on attention - as a basis for expecting a trade-off, not as direct evidence for commercial games.
5. **Individual cases:** participants who differ from the majority and what their covariates and interviews say.
6. **Limitations** (as in the paper): descriptive, no causal claim; conditions differ in more than one element; the implicit condition keeps some aids; non-random student sample; one game; learning across trials only spread, not removed; one main coder; retrospective accounts.
""")
code(r"""
headline = pl.headline_table(desc_all, diffs)
for rq, g in headline.groupby("RQ"):
    print(f"--- {rq} ---")
    for _, r in g.iterrows():
        print(f"* {r['measure']}: {r['sentence']}")
headline.to_csv(OUT["tables"] / "headline_sentences.csv", index=False)
""")

# ============================================================ 7 Tests
md(r"""
## 7. Software testing and validation
**Four layers.** (1) Unit tests of each normaliser and cleaning rule on tiny hand-built tables with a known answer, including boundary values (age 18 vs 17, active time 480 vs 481 s, pause equal to elapsed);
(2) formula tests against values computed by hand (active time, cap, NASA-TLX, UEQ-S scoring incl. reversed items, differences, direction counts); (3) statistical checks (Cohen's kappa against scikit-learn, Holm correction against known values, bootstrap reproducibility);
(4) end-to-end tests on the dummy files: **every injected fault must be detected**, row counts, ranges and types hold, cleaning is **idempotent** (cleaning clean data changes nothing), the pipeline and generator are deterministic.

`mutation_check.py` then breaks `pipeline.py` on purpose in 12 places (changed cap, wrong TLX divisor, wrong UEQ-S offset, flipped difference sign, off-by-one boundaries, broken Holm step ...) and confirms the suite fails every time (12 of 12 mutants killed).
""")
code(r"""
res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_pipeline.py"],
                     capture_output=True, text=True, cwd=".")
print("\n".join(res.stdout.strip().splitlines()[-6:]))
assert res.returncode == 0, "unit tests failed"
""")
code(r"""
checks = [
 ("16 participants retained, 2 trials each", len(wide) == 16 and len(long) == 32),
 ("no duplicate participant x condition rows", not long.duplicated(["participant_id", "condition"]).any()),
 ("every participant has exactly one trial per condition and route", (long.groupby("participant_id")["condition"].nunique() == 2).all()
                                                                    and (long.groupby("participant_id")["route_id"].nunique() == 2).all()),
 ("sequences balanced (4 each)", wide["sequence"].value_counts().eq(4).all()),
 ("completion time within [0, 480] s", long["completion_time_sec"].dropna().between(0, pl.TIME_CAP_SEC).all()),
 ("not-completed trials have T = 480 s", (long.loc[long["timed_out"], "completion_time_sec"] == pl.TIME_CAP_SEC).all()),
 ("errors never exceed decision points", (long["wayfinding_errors"].dropna().astype(float) <= long["decision_points"].dropna().astype(float).max()).all()
                                          if "decision_points" in long else (long["error_rate"].dropna() <= 1).all()),
 ("NASA-TLX within [0, 100]", long["nasa_tlx_raw"].dropna().between(0, 100).all()),
 ("UEQ-S within [-3, +3]", long[["ueqs_overall", "ueqs_pragmatic", "ueqs_hedonic"]].stack().dropna().between(-3, 3).all()),
 ("UEQ-S overall = mean of pragmatic and hedonic (where all 8 items present)", np.allclose((lambda f: f[["ueqs_pragmatic", "ueqs_hedonic"]].mean(axis=1)[f["ueqs_overall"].notna()])(long), long["ueqs_overall"].dropna())),
 ("wide differences equal I minus X", np.allclose(wide["nasa_tlx_raw_diff"], wide["nasa_tlx_raw_I"] - wide["nasa_tlx_raw_X"], equal_nan=True)),
 ("wide and long agree on every completion time",
  np.allclose(wide.set_index("participant_id")["completion_time_sec_X"].sort_index(),
              long[long.condition == "Explicit"].set_index("participant_id")["completion_time_sec"].astype(float).sort_index(), equal_nan=True)),
 ("every direction count sums to the number of pairs", (desc_all[["n_higher_X", "n_higher_I", "n_equal"]].sum(axis=1) == desc_all["n_pairs"]).all()),
 ("no raw UEQ-S items or free text left in the analysis table", not any(c.startswith("ueqs_item") or c.endswith("_notes") for c in long.columns)),
 ("every missing value is traceable to a logged action", long[list(pl.MEASURES)].isna().sum().sum() <= len(log[log["action"].isin(["set_missing", "repaired"])])),
 ("second coder covers a quarter of participants", n_double == len(wide) // 4),
]
vt = pd.DataFrame(checks, columns=["validation check", "passed"])
display(vt)
assert vt["passed"].all(), "a validation check failed"
""")

# ============================================================ 8 Export / reproducibility
md("## 8. Export and reproducibility")
code(r"""
long.to_csv(OUT["processed"] / "analysis_long.csv", index=False)
wide.to_csv(OUT["processed"] / "analysis_wide.csv", index=False)
clean.log.to_frame().to_csv(OUT["processed"] / "cleaning_log.csv", index=False)
for t, q in clean.quarantine.items():
    q.to_csv(OUT["processed"] / f"quarantine_{t}.csv", index=False)
for t, df in clean.tables.items():
    df.drop(columns=["_row"], errors="ignore").to_csv(OUT["processed"] / f"clean_{t}.csv", index=False)

sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(DATA_DIR.glob(f"*{SUFFIX}.csv"))}
display(pd.Series(sha, name="SHA-256 (first 16)").to_frame())
files = sorted(str(p) for d in OUT.values() for p in d.iterdir() if p.suffix in (".png", ".csv"))
print(f"{len(files)} output files written to figures/, tables/, processed/")
""")
md(r"""
**To re-run on real data:** put `participants.csv`, `trials.csv`, `questionnaires.csv`, `reliability.csv`, `interviews_coded.csv` next to this notebook, set `SUFFIX = ""` and `IS_DUMMY = False`, update `ROUTE_DECISION_POINTS` in `pipeline.py`, then *Restart and run all*. The cleaning log tells you exactly what was changed or set aside.
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "notebook.ipynb")
print("cells:", len(C))
