"""
plots.py - figure functions for the results section.

Design rules applied to every figure
  * Colour-blind-safe palette (Okabe-Ito): blue = more explicit (X), vermillion =
    more implicit (I). Colour always means condition, nothing else.
  * Every axis has a label with its unit; every panel states n (pairs).
  * Individual participants are shown (paired lines / dots), not only means.
  * A "DUMMY DATA" watermark is added while DUMMY_WATERMARK is True.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import pipeline as pl

DUMMY_WATERMARK = True
C_X, C_I = "#0072B2", "#D55E00"           # explicit, implicit
C_GREY, C_LIGHT = "#6b6b6b", "#d9d9d9"
COLORS = {"Explicit": C_X, "Implicit": C_I}
XLABELS = ["More explicit\n(X)", "More implicit\n(I)"]
OUTDIR = Path("figures")

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def _finish(fig, suptitle=None, note=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=1.02)
    bottom = []
    if note:
        bottom.append(note)
    if DUMMY_WATERMARK:
        bottom.append("DUMMY DATA - illustrative only, not study results")
    if bottom:
        fig.text(0.5, -0.04, "\n".join(bottom), ha="center", va="top", fontsize=8,
                 color="#9a1b1b" if DUMMY_WATERMARK else C_GREY)
    return fig


def save_fig(fig, name, outdir=None):
    out = Path(outdir or OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    return out / f"{name}.png"


def label_of(m):
    lab, unit = pl.MEASURES.get(m, (m, ""))
    return lab, (f"{lab} ({unit})" if unit and unit != "count" else (f"{lab} (count)" if unit == "count" else lab))


# -----------------------------------------------------------------------------
def _jitter(ids):
    """Fixed horizontal offset per participant, so ties do not hide each other and
    a participant's two points and their connecting line stay aligned."""
    rng = np.random.default_rng(7)
    allp = {f"P{k:02d}": rng.uniform(-0.07, 0.07) for k in range(1, 100)}
    return np.array([allp.get(p, 0.0) for p in ids])


def paired_panel(ax, wide, m, long=None):
    """Paired plot: one grey line per participant, medians as thick bars."""
    d = wide[[f"{m}_X", f"{m}_I", "participant_id"]].dropna()
    x, i = d[f"{m}_X"].to_numpy(float), d[f"{m}_I"].to_numpy(float)
    j = _jitter(d["participant_id"])
    for a, b, jj in zip(x, i, j):
        ax.plot([0 + jj, 1 + jj], [a, b], color=C_GREY, alpha=0.4, lw=1.1, zorder=2)
    ax.scatter(0 + j, x, s=34, color=C_X, edgecolor="white", lw=0.6, zorder=3)
    ax.scatter(1 + j, i, s=34, color=C_I, edgecolor="white", lw=0.6, zorder=3)
    for pos, vals, col in ((0, x, C_X), (1, i, C_I)):
        med = np.median(vals) if len(vals) else np.nan
        ax.hlines(med, pos - 0.2, pos + 0.2, color=col, lw=3.2, zorder=4)
        ax.text(pos + (-0.25 if pos == 0 else 0.25), med, f"{med:.3g}", ha="right" if pos == 0 else "left",
                va="center", fontsize=8.5, color=col, fontweight="bold")
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], XLABELS)
    lab, full = label_of(m)
    ax.set_ylabel(full)
    ax.set_title(lab, loc="left")
    ax.text(0.99, 0.02, f"n = {len(d)} pairs", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=C_GREY)
    if pl.MEASURES.get(m, ("", ""))[1] == "count":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        top = max(np.nanmax(x), np.nanmax(i))
        ax.set_ylim(-0.15, top + 0.5)
    if m.startswith("ueqs"):
        ax.set_ylim(-3.1, 3.1)
        ax.axhline(0, color="black", lw=0.8, ls=":")
    elif m == "nasa_tlx_raw":
        ax.set_ylim(0, 100)
    elif pl.MEASURES.get(m, ("", ""))[1] != "count":
        ax.set_ylim(bottom=0)
    return ax


def fig_paired(wide, measures, title, ncols=3, long=None, cap_line=False):
    n = len(measures)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.9 * nrows), squeeze=False)
    for ax, m in zip(axes.ravel(), measures):
        paired_panel(ax, wide, m)
        if m == "completion_time_sec":
            if cap_line:
                ax.axhline(pl.TIME_CAP_SEC, color="black", ls="--", lw=0.9)
                ax.text(-0.55, pl.TIME_CAP_SEC + 8, "8-min cap", ha="left", fontsize=7.5)
            ax.set_ylim(0, pl.TIME_CAP_SEC * 1.12)
            if long is not None:
                to = long[long["timed_out"]]
                for _, r in to.iterrows():
                    jx = _jitter([r["participant_id"]])[0]
                    ax.scatter((0 if r["condition"] == "Explicit" else 1) + jx, r["completion_time_sec"], s=120,
                               marker="X", color="none", edgecolor="black", lw=1.3, zorder=5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=C_X, label="More explicit (X)"),
               Line2D([], [], marker="o", ls="", color=C_I, label="More implicit (I)"),
               Line2D([], [], color=C_GREY, alpha=0.6, label="One participant"),
               Line2D([], [], color="black", lw=3, label="Median")]
    if long is not None and "completion_time_sec" in measures and long["timed_out"].any():
        handles.append(Line2D([], [], marker="X", ls="", mfc="none", mec="black", ms=9,
                              label="Not completed (time = cap)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _finish(fig, title, "Darker connecting lines mean several participants share the same pair of values.")


def fig_tlx_subscales(long):
    """Median and IQR per NASA-TLX subscale and condition, with individual dots."""
    cols = pl.TLX_COLS
    names = {"mental_demand": "Mental demand", "physical_demand": "Physical demand",
             "temporal_demand": "Temporal demand", "performance": "Performance*",
             "effort": "Effort", "frustration": "Frustration"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    rng = np.random.default_rng(1)
    for k, c in enumerate(cols):
        for cond, off in (("Explicit", -0.17), ("Implicit", 0.17)):
            v = long.loc[long["condition"] == cond, c].astype(float).dropna().to_numpy()
            y = k + off
            ax.scatter(v, y + rng.uniform(-0.07, 0.07, len(v)), s=16, color=COLORS[cond], alpha=0.35, zorder=2)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.hlines(y, q1, q3, color=COLORS[cond], lw=5, alpha=0.55, zorder=3)
            ax.scatter([med], [y], s=70, color=COLORS[cond], edgecolor="black", lw=0.8, zorder=4)
    ax.set_yticks(range(len(cols)), [names[c] for c in cols])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rating (0 = very low, 100 = very high)")
    ax.set_title("NASA-TLX subscales by condition", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_X, label="More explicit (X)"),
                       Line2D([], [], marker="o", ls="", color=C_I, label="More implicit (I)"),
                       Line2D([], [], color=C_GREY, lw=5, alpha=0.55, label="IQR; dot with outline = median")],
              loc="lower right", fontsize=8.5)
    return _finish(fig, None, "*Performance is scored 0 = perfect, 100 = failure, so higher is worse on every subscale.")


def fig_differences(wide, measures, diffs):
    """Per-participant difference D = I - X with mean and bootstrap 95% CI."""
    n = len(measures)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows), squeeze=False)
    rng = np.random.default_rng(3)
    dd = diffs.set_index("measure")
    for ax, m in zip(axes.ravel(), measures):
        d = wide[f"{m}_diff"].astype(float).dropna().to_numpy()
        ax.axhline(0, color="black", lw=0.9, ls=":")
        ax.scatter(rng.uniform(-0.12, 0.12, len(d)), d, s=26, color=C_GREY, alpha=0.6, zorder=2)
        r = dd.loc[m]
        ax.errorbar([0.3], [r["mean_diff"]], yerr=[[r["mean_diff"] - r["ci95_low"]], [r["ci95_high"] - r["mean_diff"]]],
                    fmt="o", color="black", capsize=4, ms=6, zorder=4)
        ax.set_xlim(-0.4, 0.7)
        ax.set_xticks([])
        lab, full = label_of(m)
        ax.set_title(lab, loc="left", fontsize=9.5)
        unit = pl.MEASURES[m][1]
        ax.set_ylabel(f"I - X ({unit})")
        ax.text(0.98, 0.02, f"n = {len(d)}", transform=ax.transAxes, ha="right", fontsize=8, color=C_GREY)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_GREY, alpha=0.6, label="One participant"),
                        Line2D([], [], marker="o", ls="-", color="black", label="Mean difference, bootstrap 95% CI")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _finish(fig, "Within-participant differences (more implicit minus more explicit)",
                   "Above 0 = higher under the more implicit condition. Estimation only; no significance test.")


def fig_order_route(long):
    """Completion time and errors by trial position and by route, per condition."""
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2), sharey="row")
    for r, (m, ylab) in enumerate((("completion_time_sec", "Completion time (s)"),
                                   ("wayfinding_errors", "Wayfinding errors (count)"))):
        for c, (grp, levels, xl) in enumerate((("trial_position", [1, 2], ["Trial 1", "Trial 2"]),
                                               ("route_id", list(pl.ROUTES), ["Route A (R1)", "Route B (R2)"]))):
            ax = axes[r, c]
            for cond, off in (("Explicit", -0.14), ("Implicit", 0.14)):
                for k, lvl in enumerate(levels):
                    g = long[(long["condition"] == cond) & (long[grp].astype(str) == str(lvl))][m].astype(float).dropna()
                    xs = k + off + np.random.default_rng(k).uniform(-0.05, 0.05, len(g))
                    ax.scatter(xs, g, s=26, color=COLORS[cond], alpha=0.55, zorder=2)
                    if len(g):
                        ax.hlines(g.median(), k + off - 0.12, k + off + 0.12, color=COLORS[cond], lw=3, zorder=3)
                        ax.text(k + off, 0.01, f"n={len(g)}", ha="center", va="bottom", fontsize=7, color=C_GREY,
                                transform=ax.get_xaxis_transform())
            ax.set_xticks(range(2), xl)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim((0, pl.TIME_CAP_SEC * 1.1) if m == "completion_time_sec" else (-0.25, 4.6))
            if c == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=True)
            ax.set_title(("By trial position" if c == 0 else "By route"), loc="left")
            if m == "wayfinding_errors":
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_X, label="More explicit (X)"),
                        Line2D([], [], marker="o", ls="", color=C_I, label="More implicit (I)"),
                        Line2D([], [], color="black", lw=3, label="Median")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _finish(fig, "Learning and route checks: each dot is one trial",
                   "Counterbalancing puts each condition in each position and route equally often.")


def fig_code_counts(counts, codes, title, n_participants):
    """Participants who mentioned each code, by condition (grouped horizontal bars)."""
    c = counts.set_index("code").reindex(codes)
    y = np.arange(len(codes))
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(codes) + 1.8))
    ax.barh(y - 0.19, c["participants_X"], height=0.36, color=C_X, label="More explicit (X)")
    ax.barh(y + 0.19, c["participants_I"], height=0.36, color=C_I, label="More implicit (I)")
    for yy, a, b in zip(y, c["participants_X"], c["participants_I"]):
        ax.text(a + 0.2, yy - 0.19, str(int(a)), va="center", fontsize=8)
        ax.text(b + 0.2, yy + 0.19, str(int(b)), va="center", fontsize=8)
    ax.set_yticks(y, [pl.CODEBOOK.get(k, (0, 0, k))[2] for k in codes])
    ax.invert_yaxis()
    ax.set_xlim(0, n_participants + 1.5)
    ax.set_xlabel(f"Participants who mentioned it (out of {n_participants})")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    prev = None
    for k, code in enumerate(codes):
        grp = pl.CODEBOOK.get(code, (0, "", ""))[1]
        if prev is not None and grp != prev:
            ax.axhline(k - 0.5, color="black", lw=0.8)
        prev = grp
    ax.legend(loc="lower right", fontsize=8.5)
    return _finish(fig, None, "Reconciled interview codes; a participant is counted once per code and condition.")


def fig_covariates(wide, measure="completion_time_sec"):
    """Individual-case reading: I - X difference vs experience covariates."""
    covs = [("weekly_gaming_hours", "Gaming hours per week"), ("years_of_gaming", "Years of gaming")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    lab, full = label_of(measure)
    for ax, (cv, cl) in zip(axes, covs):
        d = wide[["participant_id", cv, f"{measure}_diff", "open_world_game_experience"]].dropna()
        ow = d["open_world_game_experience"].astype(bool)
        ax.scatter(d.loc[ow, cv], d.loc[ow, f"{measure}_diff"], s=48, marker="o", color=C_GREY, label="Open-world experience: yes")
        ax.scatter(d.loc[~ow, cv], d.loc[~ow, f"{measure}_diff"], s=58, marker="s", color="white",
                   edgecolor="black", label="Open-world experience: no")
        for _, r in d.iterrows():
            ax.annotate(r["participant_id"], (r[cv], r[f"{measure}_diff"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xlabel(cl)
    axes[0].set_ylabel(f"{lab}: I - X ({pl.MEASURES[measure][1]})")
    axes[0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    return _finish(fig, "Reading individual cases next to experience (descriptive)",
                   "Covariates describe participants; they are not used to adjust any measure.")


def fig_cleaning(clean):
    """Issue counts per rule and action, plus rows in/out per table."""
    log = clean.log.to_frame()
    log = log[log["table"] != "analysis_long"]
    colors = {"repaired": "#009E73", "set_missing": "#E69F00", "quarantined": "#CC3311", "flagged": "#56B4E9"}
    g = log.groupby(["rule", "action"]).size().rename("n").reset_index()
    g = g.sort_values(["action", "rule"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, max(3.6, 0.38 * len(g) + 1.2)),
                                 gridspec_kw={"width_ratios": [1.7, 1]})
    a1.barh(g["rule"], g["n"], color=[colors[a] for a in g["action"]])
    a1.invert_yaxis()
    a1.set_xlabel("Number of cases")
    a1.set_title("Cleaning actions by rule", loc="left")
    a1.xaxis.set_major_locator(MaxNLocator(integer=True))
    a1.grid(axis="y", visible=False)
    a1.legend(handles=[Line2D([], [], marker="s", ls="", color=c, label=a) for a, c in colors.items()],
              loc="lower right", fontsize=8)
    rc = clean.row_counts
    yy = np.arange(len(rc))
    a2.barh(yy, rc["rows_in"], color=C_LIGHT, label="Rows in (raw)")
    a2.barh(yy, rc["rows_out"], color=C_GREY, height=0.5, label="Rows out (clean)")
    a2.set_yticks(yy, rc.index)
    a2.invert_yaxis()
    a2.set_xlabel("Rows")
    a2.set_title("Rows before and after", loc="left")
    a2.grid(axis="y", visible=False)
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _finish(fig, None)


def fig_reliability(rel_summary):
    """Agreement per coded measure."""
    d = rel_summary.copy()
    fig, ax = plt.subplots(figsize=(7.6, 0.6 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.barh(y, d["percent_agreement"], color=C_GREY, height=0.5)
    for yy, v, k in zip(y, d["percent_agreement"], d["value"]):
        txt = f"{v:.0f}%" + (f"   kappa = {k:.2f}" if not pd.isna(k) else "")
        ax.text(min(v + 1, 70), yy, txt, va="center", fontsize=8.5, color="white" if v > 75 else "black")
    ax.set_yticks(y, [m.replace(" (per decision point)", "\n(per decision point)").replace(" (per trial count)", "\n(per trial count)") for m in d["measure"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exact agreement between coder 1 and coder 2 (%)")
    ax.set_title("Inter-coder agreement on the double-coded quarter", loc="left")
    ax.grid(axis="y", visible=False)
    return _finish(fig, None)


def fig_table(df, title=None, col_widths=None, font=9, wrap=38, note=None):
    """Render a DataFrame as a clean, slide-ready table image."""
    import textwrap
    cells = [[textwrap.fill(str(v), wrap) for v in row] for row in df.astype(object).fillna("").values.tolist()]
    heads = [textwrap.fill(str(c), 16) for c in df.columns]
    nlines = [max(c.count("\n") + 1 for c in row) for row in cells]
    height = 0.3 + 0.3 * (max(h.count("\n") + 1 for h in heads)) + sum(0.27 * n + 0.1 for n in nlines)
    width = col_widths and sum(col_widths) or max(6.5, 1.9 * len(df.columns))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=heads, loc="upper center", cellLoc="left",
                   colWidths=[w / sum(col_widths) for w in col_widths] if col_widths else None)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.3 * (max(h.count("\n") + 1 for h in heads)) / height)
        else:
            cell.set_facecolor("#f4f6f8" if r % 2 == 0 else "white")
            cell.set_height((0.27 * nlines[r - 1] + 0.1) / height)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    return _finish(fig, None, note)


def export_table(df, name, title=None, outdir="tables", image=True, **kw):
    """Save a table as CSV (for the repo / LaTeX) and as a slide-ready PNG."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    if image:
        fig = fig_table(df, title, **kw)
        save_fig(fig, name, out)
        plt.close(fig)
    return out / f"{name}.csv"
