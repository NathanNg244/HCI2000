"""
pipeline.py - cleaning, preparation and descriptive analysis for the study
"Explicit and Implicit Navigation Guidance Effects on Player Wayfinding in
Third-Person Open-World Games".

Everything that carries logic lives here (not in the notebook) so that it can
be unit-tested (see test_pipeline.py) and re-used unchanged when the real data
replace the dummy data.

Cleaning policy (every rule is logged in an issue log):
  repaired   deterministic and safe fix (whitespace, case, ID format, boolean
             spelling, exact duplicate row, integer-valued float -> int)
  set_missing  a value that cannot be trusted (non-numeric, out of range,
             impossible combination) becomes missing instead of silently
             entering an analysis; the rest of the row is kept
  quarantined  a row that cannot be used or whose correct version cannot be
             decided (unknown participant, invalid condition, conflicting
             duplicates, ineligible participant, broken trial design)
  flagged    kept as is, but a human should look (outliers, odd but possible)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# =============================================================================
# 1. Study configuration (single source of truth, mirrors the paper)
# =============================================================================
CONDITIONS = ("Explicit", "Implicit")          # X = more explicit, I = more implicit
COND_SHORT = {"Explicit": "X", "Implicit": "I"}
ROUTES = ("Route_A", "Route_B")                # Route_A = R1, Route_B = R2 in the paper
TIME_CAP_SEC = 480                             # 8-minute cap (paper, measure 1)

# Number of decision points D per route. PLACEHOLDER: the pilot fixes these.
# The paper requires D >= 3 and 0 <= E <= D. Replace with the real values.
ROUTE_DECISION_POINTS = {"Route_A": 4, "Route_B": 4}

TLX_COLS = ["mental_demand", "physical_demand", "temporal_demand",
            "performance", "effort", "frustration"]
UEQS_RAW_COLS = [f"ueqs_item{i}_raw_1_to_7" for i in range(1, 9)]
UEQS_PRAGMATIC_ITEMS = (1, 2, 3, 4)
UEQS_HEDONIC_ITEMS = (5, 6, 7, 8)
# Items whose positive pole is on the LEFT of the 7-point scale must be reversed
# before scoring. In the English UEQ-S all 8 items run negative -> positive, so
# the default is "none". Change only if your form differs.
UEQS_REVERSED_ITEMS: tuple[int, ...] = ()

# The four counterbalanced sequences (paper, Fig. "latin"): key = (condition,
# route) of trial 1; the second trial is always the other condition on the
# other route.
SEQUENCES = {
    ("Explicit", "Route_A"): 1,
    ("Implicit", "Route_B"): 2,
    ("Implicit", "Route_A"): 3,
    ("Explicit", "Route_B"): 4,
}
GENDERS = ("Woman", "Man", "Non-binary", "Prefer not to say", "Other")
INPUT_DEVICES = ("Keyboard and mouse", "Gamepad")

REQUIRED_COLUMNS = {
    "participants": ["participant_id", "age", "gender", "weekly_gaming_hours",
                     "years_of_gaming", "third_person_3d_game_past_12_months",
                     "previous_assassins_creed_play", "input_device",
                     "open_world_game_experience"],
    "trials": ["participant_id", "trial_number", "condition", "route_id",
               "trial_position", "elapsed_time_sec", "combat_cutscene_pause_sec",
               "completed", "wayfinding_errors", "lost_episodes", "map_openings",
               "total_map_open_duration_sec", "objective_board_openings",
               "compass_use_notes"],
    "questionnaires": ["participant_id", "condition"] + TLX_COLS + UEQS_RAW_COLS,
    "reliability": ["participant_id", "condition", "measure", "item",
                    "coder1", "coder2"],
    "interviews": ["segment_id", "participant_id", "condition", "interview_stage",
                   "code_family", "code", "excerpt"],
}
OPTIONAL_TABLES = ("reliability", "interviews")

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing",
             "#n/a", "n.a."}


class SchemaError(ValueError):
    """Raised when a required column is missing from an input table."""


# =============================================================================
# 2. Issue log
# =============================================================================
class IssueLog:
    """Collects every cleaning action so that nothing happens silently."""

    COLUMNS = ["table", "csv_row", "participant_id", "field", "rule", "action",
               "original", "new", "detail"]

    def __init__(self):
        self._rows: list[dict] = []

    def add(self, table, rule, action, csv_row=None, participant_id=None,
            field=None, original=None, new=None, detail=""):
        self._rows.append(dict(table=table, csv_row=csv_row,
                               participant_id=participant_id, field=field,
                               rule=rule, action=action, original=original,
                               new=new, detail=detail))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=self.COLUMNS)

    def summary(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return pd.DataFrame(columns=["table", "rule", "action", "n"])
        return (df.groupby(["table", "rule", "action"]).size()
                  .rename("n").reset_index()
                  .sort_values(["table", "action", "rule"]).reset_index(drop=True))

    def __len__(self):
        return len(self._rows)


# =============================================================================
# 3. Value normalisers (pure functions, unit-tested)
# =============================================================================
def _is_na(v) -> bool:
    return v is None or (not isinstance(v, str) and pd.isna(v))


def clean_text(v):
    """Strip, collapse inner whitespace, map NA tokens to pd.NA."""
    if _is_na(v):
        return pd.NA
    s = re.sub(r"\s+", " ", str(v)).strip()
    return pd.NA if s.lower() in NA_TOKENS else s


def normalize_id(v):
    """'p1', 'P 1', ' P01 ', 'P001' -> 'P01'; anything else -> None."""
    s = clean_text(v)
    if _is_na(s):
        return None
    m = re.fullmatch(r"[Pp]\s*0*(\d{1,3})", s)
    return f"P{int(m.group(1)):02d}" if m else None


_COND_MAP = {"explicit": "Explicit", "more explicit": "Explicit", "x": "Explicit",
             "guided on": "Explicit", "guided exploration on": "Explicit",
             "implicit": "Implicit", "more implicit": "Implicit", "i": "Implicit",
             "guided off": "Implicit", "guided exploration off": "Implicit"}


def normalize_condition(v):
    s = clean_text(v)
    return _COND_MAP.get(str(s).lower()) if not _is_na(s) else None


_ROUTE_MAP = {"route_a": "Route_A", "route a": "Route_A", "routea": "Route_A",
              "a": "Route_A", "r1": "Route_A", "route 1": "Route_A",
              "route_b": "Route_B", "route b": "Route_B", "routeb": "Route_B",
              "b": "Route_B", "r2": "Route_B", "route 2": "Route_B"}


def normalize_route(v):
    s = clean_text(v)
    return _ROUTE_MAP.get(str(s).lower()) if not _is_na(s) else None


_TRUE = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE = {"false", "f", "no", "n", "0", "0.0"}


def normalize_bool(v):
    """True/False for common spellings, None if unrecognised."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = clean_text(v)
    if _is_na(s):
        return None
    s = str(s).lower()
    return True if s in _TRUE else False if s in _FALSE else None


def normalize_yes_no(v):
    b = normalize_bool(v)
    return None if b is None else ("Yes" if b else "No")


def normalize_gender(v):
    s = clean_text(v)
    if _is_na(s):
        return None
    low = str(s).lower().replace("-", " ")
    table = {"woman": "Woman", "female": "Woman", "f": "Woman",
             "man": "Man", "male": "Man", "m": "Man",
             "non binary": "Non-binary", "nonbinary": "Non-binary",
             "prefer not to say": "Prefer not to say", "other": "Other"}
    return table.get(low)


def normalize_device(v):
    s = clean_text(v)
    if _is_na(s):
        return None
    low = str(s).lower()
    if low in {"keyboard and mouse", "keyboard & mouse", "kbm", "keyboard/mouse",
               "keyboard + mouse", "mouse and keyboard"}:
        return "Keyboard and mouse"
    if low in {"gamepad", "controller", "game pad"}:
        return "Gamepad"
    return None


def parse_number(v):
    """Parse a number from text ('5', ' 5.0 ', '1,5' is rejected). None if not numeric."""
    if _is_na(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v) if np.isfinite(v) else None
    s = clean_text(v)
    if _is_na(s):
        return None
    try:
        x = float(str(s))
    except ValueError:
        return None
    return x if np.isfinite(x) else None


# =============================================================================
# 4. Generic cleaning helpers
# =============================================================================
def standardize_columns(df: pd.DataFrame, table: str, log: IssueLog) -> pd.DataFrame:
    """Normalise header spelling, verify the schema, drop unexpected columns."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    required = REQUIRED_COLUMNS[table]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{table}: missing required column(s) {missing}")
    for c in [c for c in df.columns if c not in required]:
        log.add(table, "unexpected_column", "repaired", field=c,
                detail="column not in the study schema was dropped")
    df = df[required].copy()
    df.insert(0, "_row", np.arange(len(df)) + 2)   # CSV line number (header = 1)
    pid_raw = df["participant_id"].copy()
    for c in required:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            before = df[c]
            after = before.map(clean_text)
            token = [(not _is_na(b)) and str(b).strip() != "" and _is_na(a)
                     for b, a in zip(before, after)]
            for i in np.flatnonzero(token):
                log.add(table, "na_token", "repaired", df.at[i, "_row"], pid_raw.iloc[i],
                        c, before.iloc[i], None, "missing-value token read as missing")
            df[c] = after
    return df.reset_index(drop=True)


def _normalise_col(df, col, func, table, log, rule, invalid_action="set_missing"):
    """Apply func to a column; log repairs and invalid values."""
    out = []
    for _, r in df.iterrows():
        old = r[col]
        if _is_na(old):
            out.append(pd.NA)
            continue
        new = func(old)
        if new is None:
            log.add(table, rule, invalid_action, r["_row"], r.get("participant_id"),
                    col, old, None, "unrecognised value")
            out.append(pd.NA)
        else:
            if str(new) != str(old):
                log.add(table, rule, "repaired", r["_row"], r.get("participant_id"),
                        col, old, new)
            out.append(new)
    df[col] = out
    return df


def _numeric_col(df, col, table, log, lo=None, hi=None, integer=False):
    """Coerce to float; non-numeric / out-of-range / non-integer -> NaN (logged)."""
    vals = []
    for _, r in df.iterrows():
        old = r[col]
        if _is_na(old):
            vals.append(np.nan)
            continue
        x = parse_number(old)
        problem = None
        if x is None:
            problem = "non_numeric"
        elif (lo is not None and x < lo) or (hi is not None and x > hi):
            problem = "out_of_range"
        elif integer and x is not None and abs(x - round(x)) > 1e-9:
            problem = "non_integer"
        if problem:
            log.add(table, f"{col}:{problem}", "set_missing", r["_row"],
                    r.get("participant_id"), col, old, None,
                    f"expected {'integer ' if integer else ''}in [{lo}, {hi}]")
            vals.append(np.nan)
        else:
            vals.append(float(round(x)) if integer else x)
    df[col] = np.asarray(vals, dtype=float)
    return df


def _quarantine(df, mask, table, rule, log, store: dict, detail="") -> pd.DataFrame:
    """Move rows where mask is True into the quarantine store."""
    mask = pd.Series(np.asarray(mask, dtype=bool), index=df.index)
    for _, r in df[mask].iterrows():
        log.add(table, rule, "quarantined", r["_row"], r.get("participant_id"),
                detail=detail)
    if mask.any():
        q = df[mask].copy()
        q["quarantine_reason"] = rule
        store[table] = pd.concat([store.get(table, pd.DataFrame()), q],
                                 ignore_index=True)
    return df[~mask].reset_index(drop=True)


def _drop_exact_duplicates(df, table, log):
    cols = [c for c in df.columns if c != "_row"]
    dup = df.duplicated(subset=cols, keep="first")
    for _, r in df[dup].iterrows():
        log.add(table, "exact_duplicate_row", "repaired", r["_row"],
                r.get("participant_id"), detail="identical to an earlier row; dropped")
    return df[~dup].reset_index(drop=True)


def _conflicting_keys(df, keys):
    return df.duplicated(subset=keys, keep=False)


# =============================================================================
# 5. Table-specific cleaning
# =============================================================================
def clean_participants(raw, log, store):
    t = "participants"
    df = standardize_columns(raw, t, log)
    df = _normalise_col(df, "participant_id", normalize_id, t, log, "participant_id:format",
                        invalid_action="set_missing")
    df = _quarantine(df, df["participant_id"].isna(), t, "invalid_participant_id", log, store)
    df = _drop_exact_duplicates(df, t, log)
    df = _quarantine(df, _conflicting_keys(df, ["participant_id"]), t,
                     "conflicting_duplicate_participant", log, store,
                     "same ID with different values; cannot decide which is right")
    df = _normalise_col(df, "gender", normalize_gender, t, log, "gender:category")
    df = _normalise_col(df, "input_device", normalize_device, t, log, "input_device:category")
    for c in ["third_person_3d_game_past_12_months", "previous_assassins_creed_play",
              "open_world_game_experience"]:
        df = _normalise_col(df, c, normalize_yes_no, t, log, f"{c}:yes_no")
    df = _numeric_col(df, "age", t, log, lo=0, hi=100, integer=True)
    df = _numeric_col(df, "weekly_gaming_hours", t, log, lo=0, hi=168)
    df = _numeric_col(df, "years_of_gaming", t, log, lo=0, hi=100)

    # impossible combination: more years of play than years of life
    bad = df["years_of_gaming"] > df["age"]
    for _, r in df[bad].iterrows():
        log.add(t, "years_of_gaming>age", "set_missing", r["_row"], r["participant_id"],
                "years_of_gaming", r["years_of_gaming"], None, "longer than the participant's life")
    df.loc[bad, "years_of_gaming"] = np.nan

    # plausibility flags (kept)
    for _, r in df[(df["weekly_gaming_hours"] > 80)].iterrows():
        log.add(t, "weekly_gaming_hours>80", "flagged", r["_row"], r["participant_id"],
                "weekly_gaming_hours", r["weekly_gaming_hours"], detail="verify with participant")
    for _, r in df[(df["age"] > 40)].iterrows():
        log.add(t, "age>40", "flagged", r["_row"], r["participant_id"], "age", r["age"],
                detail="unusual for a student sample")

    # screening rules from the paper (Section 'Screening')
    df = _quarantine(df, df["age"] < 18, t, "ineligible:age<18", log, store)
    df = _quarantine(df, df["third_person_3d_game_past_12_months"] == "No", t,
                     "ineligible:no_third_person_3d_play_12mo", log, store)
    return df.reset_index(drop=True)


def _validate_trial_numbers(df, log):
    t = "trials"
    df = _normalise_col(df, "participant_id", normalize_id, t, log, "participant_id:format")
    df = _normalise_col(df, "condition", normalize_condition, t, log, "condition:category")
    df = _normalise_col(df, "route_id", normalize_route, t, log, "route_id:category")
    df = _normalise_col(df, "completed", normalize_bool, t, log, "completed:boolean")
    for c, lo, hi, integer in [("trial_number", 1, 2, True), ("trial_position", 1, 2, True),
                               ("elapsed_time_sec", 1, 3600, False),
                               ("combat_cutscene_pause_sec", 0, 3600, False),
                               ("wayfinding_errors", 0, 50, True), ("lost_episodes", 0, 50, True),
                               ("map_openings", 0, 200, True),
                               ("total_map_open_duration_sec", 0, 3600, False),
                               ("objective_board_openings", 0, 200, True)]:
        df = _numeric_col(df, c, t, log, lo, hi, integer)
    return df


def clean_trials(raw, valid_ids, log, store):
    t = "trials"
    df = standardize_columns(raw, t, log)
    df = _validate_trial_numbers(df, log)
    # rows we cannot place in the design at all
    df = _quarantine(df, df["participant_id"].isna(), t, "invalid_participant_id", log, store)
    df = _quarantine(df, df["condition"].isna(), t, "invalid_condition", log, store)
    df = _quarantine(df, df["route_id"].isna(), t, "invalid_route", log, store)
    df = _drop_exact_duplicates(df, t, log)
    df = _quarantine(df, ~df["participant_id"].isin(valid_ids), t,
                     "unknown_or_excluded_participant", log, store,
                     "no eligible participant record")

    # trial_position is canonical (recording order); trial_number must agree
    mism = df["trial_number"].notna() & df["trial_position"].notna() & (df["trial_number"] != df["trial_position"])
    for _, r in df[mism].iterrows():
        log.add(t, "trial_number!=trial_position", "flagged", r["_row"], r["participant_id"],
                detail="trial_position (recording order) is used")
    fill = df["trial_position"].isna() & df["trial_number"].notna()
    for _, r in df[fill].iterrows():
        log.add(t, "trial_position_filled_from_trial_number", "repaired", r["_row"],
                r["participant_id"], "trial_position", None, r["trial_number"])
    df.loc[fill, "trial_position"] = df.loc[fill, "trial_number"]
    df = _quarantine(df, df["trial_position"].isna(), t, "missing_trial_position", log, store)

    # conflicting duplicates (same participant+condition, or same participant+position)
    for keys, name in [(["participant_id", "condition"], "conflicting_duplicate_condition"),
                       (["participant_id", "trial_position"], "conflicting_duplicate_position")]:
        df = _quarantine(df, _conflicting_keys(df, keys), t, name, log, store,
                         "cannot decide which record is correct; check recordings")

    # cross-field checks
    bad_pause = df["combat_cutscene_pause_sec"] >= df["elapsed_time_sec"]
    for _, r in df[bad_pause].iterrows():
        log.add(t, "pause>=elapsed", "set_missing", r["_row"], r["participant_id"],
                "elapsed_time_sec;combat_cutscene_pause_sec",
                f"{r['elapsed_time_sec']};{r['combat_cutscene_pause_sec']}", None,
                "pauses cannot cover the whole trial; cannot tell which value is wrong")
    df.loc[bad_pause, ["elapsed_time_sec", "combat_cutscene_pause_sec"]] = np.nan

    d = df["route_id"].map(ROUTE_DECISION_POINTS)
    over = df["wayfinding_errors"] > d
    for _, r in df[over].iterrows():
        log.add(t, "errors>decision_points", "set_missing", r["_row"], r["participant_id"],
                "wayfinding_errors", r["wayfinding_errors"], None,
                f"E must satisfy 0 <= E <= D = {ROUTE_DECISION_POINTS[r['route_id']]}")
    df.loc[over, "wayfinding_errors"] = np.nan

    inc = (df["map_openings"] == 0) & (df["total_map_open_duration_sec"] > 0)
    for _, r in df[inc].iterrows():
        log.add(t, "map_duration>0_with_0_openings", "set_missing", r["_row"], r["participant_id"],
                "total_map_open_duration_sec", r["total_map_open_duration_sec"], None,
                "inconsistent with map_openings = 0")
    df.loc[inc, "total_map_open_duration_sec"] = np.nan

    too_long = df["total_map_open_duration_sec"] > df["elapsed_time_sec"]
    for _, r in df[too_long].iterrows():
        log.add(t, "map_duration>elapsed", "set_missing", r["_row"], r["participant_id"],
                "total_map_open_duration_sec", r["total_map_open_duration_sec"], None,
                "map open longer than the whole trial")
    df.loc[too_long, "total_map_open_duration_sec"] = np.nan

    # completion consistency (paper: not completed by the cap => not completed)
    active = df["elapsed_time_sec"] - df["combat_cutscene_pause_sec"]
    over_cap = (df["completed"] == True) & (active > TIME_CAP_SEC)  # noqa: E712
    for _, r in df[over_cap].iterrows():
        log.add(t, "completed_but_over_cap", "repaired", r["_row"], r["participant_id"],
                "completed", True, False, "active time exceeds the cap; counted as not completed")
    df.loc[over_cap, "completed"] = False
    early_stop = (df["completed"] == False) & (active < TIME_CAP_SEC)  # noqa: E712
    for _, r in df[early_stop].iterrows():
        log.add(t, "not_completed_before_cap", "flagged", r["_row"], r["participant_id"],
                "completed", False, detail="stopped before the cap; check for technical failure "
                                          "(paper: replace participant from waitlist)")
    df["completed"] = df["completed"].astype("boolean")
    return df.reset_index(drop=True)


def check_design(trials, log, store):
    """Every participant must have exactly two trials: one per condition, one per
    route, positions 1 and 2, forming one of the four counterbalanced sequences."""
    t = "trials"
    bad = {}
    for pid, g in trials.groupby("participant_id"):
        g = g.sort_values("trial_position")
        why = None
        if len(g) != 2:
            why = f"{len(g)} valid trial(s) instead of 2"
        elif set(g["condition"]) != set(CONDITIONS):
            why = "trials are not one per condition"
        elif set(g["route_id"]) != set(ROUTES):
            why = "a route was repeated"
        elif list(g["trial_position"]) != [1, 2]:
            why = "trial positions are not 1 and 2"
        else:
            first = (g.iloc[0]["condition"], g.iloc[0]["route_id"])
            second = (g.iloc[1]["condition"], g.iloc[1]["route_id"])
            if first not in SEQUENCES:
                why = "first trial is not part of a defined sequence"
            elif second != (CONDITIONS[1 - CONDITIONS.index(first[0])],
                            ROUTES[1 - ROUTES.index(first[1])]):
                why = "second trial does not complete the counterbalanced pair"
        if why:
            bad[pid] = why
    for pid, why in bad.items():
        log.add(t, "design_violation", "quarantined", participant_id=pid, detail=why)
    return bad


def clean_questionnaires(raw, valid_ids, log, store):
    t = "questionnaires"
    df = standardize_columns(raw, t, log)
    df = _normalise_col(df, "participant_id", normalize_id, t, log, "participant_id:format")
    df = _normalise_col(df, "condition", normalize_condition, t, log, "condition:category")
    df = _quarantine(df, df["participant_id"].isna(), t, "invalid_participant_id", log, store)
    df = _quarantine(df, df["condition"].isna(), t, "invalid_condition", log, store)
    for c in TLX_COLS:
        df = _numeric_col(df, c, t, log, lo=0, hi=100)
    for c in UEQS_RAW_COLS:
        df = _numeric_col(df, c, t, log, lo=1, hi=7, integer=True)
    df = _drop_exact_duplicates(df, t, log)
    df = _quarantine(df, ~df["participant_id"].isin(valid_ids), t,
                     "unknown_or_excluded_participant", log, store,
                     "no eligible participant record")
    df = _quarantine(df, _conflicting_keys(df, ["participant_id", "condition"]), t,
                     "conflicting_duplicate_condition", log, store,
                     "two different questionnaires for one participant and condition")
    # straight-lining is possible but suspicious
    items = df[UEQS_RAW_COLS]
    sl = (items.nunique(axis=1) == 1) & items.notna().all(axis=1)
    for _, r in df[sl].iterrows():
        log.add(t, "ueqs_straight_lining", "flagged", r["_row"], r["participant_id"],
                detail="all eight UEQ-S items identical")
    return df.reset_index(drop=True)


def clean_reliability(raw, valid_ids, log, store):
    t = "reliability"
    df = standardize_columns(raw, t, log)
    df = _normalise_col(df, "participant_id", normalize_id, t, log, "participant_id:format")
    df = _normalise_col(df, "condition", normalize_condition, t, log, "condition:category")
    df = _quarantine(df, df["participant_id"].isna() | df["condition"].isna(), t,
                     "invalid_key", log, store)
    df = _quarantine(df, ~df["participant_id"].isin(valid_ids), t,
                     "unknown_or_excluded_participant", log, store)
    df = _numeric_col(df, "coder1", t, log, lo=0, hi=200, integer=True)
    df = _numeric_col(df, "coder2", t, log, lo=0, hi=200, integer=True)
    df["item"] = df["item"].map(lambda v: pd.NA if _is_na(v) else str(v).strip().lower().removesuffix(".0"))
    df["measure"] = df["measure"].map(lambda v: pd.NA if _is_na(v) else str(v).strip().lower())
    df = _drop_exact_duplicates(df, t, log)
    df = _quarantine(df, _conflicting_keys(df, ["participant_id", "condition", "measure", "item"]),
                     t, "conflicting_duplicate_item", log, store)
    err = (df["measure"] == "error_code") & ((df["coder1"] > 1) | (df["coder2"] > 1))
    for _, r in df[err].iterrows():
        log.add(t, "error_code_not_binary", "set_missing", r["_row"], r["participant_id"],
                detail="per-decision-point error code must be 0 or 1")
    df.loc[err, ["coder1", "coder2"]] = np.nan
    df = _quarantine(df, df["coder1"].isna() | df["coder2"].isna(), t,
                     "incomplete_coder_pair", log, store,
                     "agreement needs both coders' values")
    return df.reset_index(drop=True)


def clean_interviews(raw, valid_ids, log, store):
    t = "interviews"
    df = standardize_columns(raw, t, log)
    df = _normalise_col(df, "participant_id", normalize_id, t, log, "participant_id:format")
    df = _normalise_col(df, "condition", normalize_condition, t, log, "condition:category")
    df["interview_stage"] = df["interview_stage"].map(lambda v: pd.NA if _is_na(v) else str(v).lower())
    for c in ["code_family", "code"]:
        df[c] = df[c].map(lambda v: pd.NA if _is_na(v) else str(v).strip().lower().replace(" ", "_"))
    df = _quarantine(df, df["participant_id"].isna() | df["code"].isna(), t,
                     "invalid_key_or_code", log, store)
    df = _quarantine(df, ~df["participant_id"].isin(valid_ids), t,
                     "unknown_or_excluded_participant", log, store)
    # the final interview covers both conditions, so condition may be missing there
    stage_ok = df["interview_stage"].isin(["trial", "final"])
    df = _quarantine(df, ~stage_ok, t, "invalid_interview_stage", log, store)
    no_cond = (df["interview_stage"] == "trial") & df["condition"].isna()
    df = _quarantine(df, no_cond, t, "trial_interview_without_condition", log, store)
    df = _drop_exact_duplicates(df, t, log)
    df = _quarantine(df, df["segment_id"].isna(), t, "missing_segment_id", log, store)
    df = _quarantine(df, df.duplicated(subset=["segment_id"], keep=False), t,
                     "conflicting_duplicate_segment_id", log, store,
                     "one segment ID with different content")
    return df.reset_index(drop=True)


@dataclass
class CleaningResult:
    tables: dict
    quarantine: dict
    log: IssueLog
    row_counts: pd.DataFrame
    excluded_participants: dict = field(default_factory=dict)

    @property
    def participants(self): return self.tables["participants"]
    @property
    def trials(self): return self.tables["trials"]
    @property
    def questionnaires(self): return self.tables["questionnaires"]


def clean_all(raw: dict) -> CleaningResult:
    """Run every cleaning step in dependency order and return cleaned tables,
    the quarantine, the issue log and a before/after row count."""
    log, store = IssueLog(), {}
    counts = {k: [len(v), None] for k, v in raw.items()}

    parts = clean_participants(raw["participants"], log, store)
    ids = set(parts["participant_id"])
    trials = clean_trials(raw["trials"], ids, log, store)
    bad = check_design(trials, log, store)
    if bad:
        trials = _quarantine(trials, trials["participant_id"].isin(bad), "trials",
                             "design_violation", log, store)
        # a participant whose trial design is broken is excluded everywhere
        ids -= set(bad)
        parts = parts[parts["participant_id"].isin(ids)].reset_index(drop=True)
    qs = clean_questionnaires(raw["questionnaires"], ids, log, store)

    # participants without any trial cannot be analysed
    no_trials = ids - set(trials["participant_id"])
    for pid in sorted(no_trials):
        log.add("participants", "participant_without_trials", "quarantined", participant_id=pid)
    parts = parts[~parts["participant_id"].isin(no_trials)].reset_index(drop=True)
    ids -= no_trials

    tables = {"participants": parts, "trials": trials, "questionnaires": qs}
    for name, fn in (("reliability", clean_reliability), ("interviews", clean_interviews)):
        if name in raw and raw[name] is not None:
            tables[name] = fn(raw[name], ids, log, store)

    # a questionnaire / trial present for only one of the two conditions
    for pid in sorted(ids):
        for tname in ("trials", "questionnaires"):
            have = set(tables[tname].loc[tables[tname]["participant_id"] == pid, "condition"])
            for c in set(CONDITIONS) - have:
                log.add(tname, "missing_condition_record", "flagged", participant_id=pid,
                        detail=f"no {c} row; paired analysis uses pairwise-complete data")

    for k in counts:
        counts[k][1] = len(tables[k])
    rc = pd.DataFrame(counts, index=["rows_in", "rows_out"]).T
    rc["rows_removed"] = rc["rows_in"] - rc["rows_out"]
    return CleaningResult(tables, store, log, rc, bad)


def flag_outliers(long: pd.DataFrame, measures, log: IssueLog, threshold=3.5):
    """Modified z-score (median/MAD) outliers within condition. Flag only: with
    n = 16 an extreme value may be a real player, so it is checked, not removed."""
    for m in measures:
        for cond, g in long.groupby("condition", observed=True):
            x = g[m].astype(float).dropna()
            if len(x) < 5:
                continue
            mad = np.median(np.abs(x - x.median()))
            if mad == 0:
                continue
            z = 0.6745 * (x - x.median()) / mad
            for idx in z[np.abs(z) > threshold].index:
                log.add("analysis_long", f"{m}:outlier", "flagged",
                        participant_id=g.loc[idx, "participant_id"], field=m,
                        original=g.loc[idx, m], detail=f"{cond}: modified z = {z[idx]:.1f}")


# =============================================================================
# 6. Data preparation (derived variables, analysis-ready tables)
# =============================================================================
def active_time(elapsed, pause):
    """Active play time: elapsed minus combat/cutscene pauses (paper, measure 1)."""
    return elapsed - pause


def completion_time(active, completed, cap=TIME_CAP_SEC):
    """T = active time capped at `cap`; a trial that was not completed is assigned
    the cap (paper, measure 1). Missing `completed` is treated as completed."""
    active = np.asarray(active, dtype=float)
    done = np.asarray(pd.Series(completed).astype("boolean").fillna(True), dtype=bool)
    return np.where(done, np.minimum(active, cap), float(cap))


def code_compass_use(note):
    """Interview-derived ordinal: 0 none, 1 brief, 2 frequent, 3 relied on."""
    if _is_na(note):
        return np.nan
    s = str(note).lower()
    if re.search(r"\b(did not|didn't|never|no) (use|used|check)", s) or "ignored" in s:
        return 0.0
    if re.search(r"relied|followed|constantly|continuous", s):
        return 3.0
    if re.search(r"several|frequent|often|repeated", s):
        return 2.0
    if re.search(r"brief|glanc|once|occasional|twice", s):
        return 1.0
    return np.nan


COMPASS_LABELS = {0: "None", 1: "Brief", 2: "Frequent", 3: "Relied on"}


def tlx_raw(df: pd.DataFrame) -> pd.Series:
    """Raw (unweighted) NASA-TLX = mean of 6 subscales; NaN if any is missing."""
    return df[TLX_COLS].mean(axis=1, skipna=False)


def ueqs_scores(df: pd.DataFrame, reversed_items=UEQS_REVERSED_ITEMS) -> pd.DataFrame:
    """UEQ-S: raw 1..7 -> -3..+3 (raw - 4; reversed items flipped); pragmatic =
    items 1-4, hedonic = items 5-8, overall = mean of all eight."""
    sc = pd.DataFrame(index=df.index)
    for i, col in enumerate(UEQS_RAW_COLS, start=1):
        v = df[col].astype(float) - 4.0
        sc[f"i{i}"] = -v if i in reversed_items else v
    out = pd.DataFrame(index=df.index)
    out["ueqs_pragmatic"] = sc[[f"i{i}" for i in UEQS_PRAGMATIC_ITEMS]].mean(axis=1, skipna=False)
    out["ueqs_hedonic"] = sc[[f"i{i}" for i in UEQS_HEDONIC_ITEMS]].mean(axis=1, skipna=False)
    out["ueqs_overall"] = sc.mean(axis=1, skipna=False)
    return out


MEASURES = {   # column -> (label, unit)
    "completion_time_sec": ("Completion time", "s"),
    "wayfinding_errors": ("Wayfinding errors", "count"),
    "lost_episodes": ("Lost episodes", "count"),
    "map_openings": ("Map openings", "count"),
    "map_open_duration_sec": ("Map open duration", "s"),
    "objective_board_openings": ("Objective-board openings", "count"),
    "nasa_tlx_raw": ("NASA-TLX (raw)", "0-100"),
    "ueqs_overall": ("UEQ-S overall", "-3 to +3"),
    "ueqs_pragmatic": ("UEQ-S pragmatic", "-3 to +3"),
    "ueqs_hedonic": ("UEQ-S hedonic", "-3 to +3"),
}
RQ_MEASURES = {
    "RQ3": ["completion_time_sec", "wayfinding_errors", "lost_episodes"],
    "RQ2": ["map_openings", "map_open_duration_sec", "objective_board_openings"],
    "RQ4": ["nasa_tlx_raw", "ueqs_overall", "ueqs_pragmatic", "ueqs_hedonic"],
}
COUNT_COLS = ["wayfinding_errors", "lost_episodes", "map_openings", "objective_board_openings"]


def prepare(clean: CleaningResult):
    """Return (analysis_long, analysis_wide). Long = one row per trial; wide =
    one row per participant with both conditions side by side (paper, 'Coding and
    data handling')."""
    trials = clean.trials.copy()
    qs = clean.questionnaires.copy()
    parts = clean.participants.copy()

    trials["active_time_sec"] = active_time(trials["elapsed_time_sec"], trials["combat_cutscene_pause_sec"])
    comp = trials["completed"].astype("boolean")
    trials["timed_out"] = ~comp.fillna(True).astype(bool)
    trials["completion_time_sec"] = completion_time(trials["active_time_sec"], comp)
    trials.loc[trials["active_time_sec"].isna() & ~trials["timed_out"], "completion_time_sec"] = np.nan
    trials["decision_points"] = trials["route_id"].map(ROUTE_DECISION_POINTS).astype(float)
    trials["error_rate"] = trials["wayfinding_errors"] / trials["decision_points"]
    trials = trials.rename(columns={"total_map_open_duration_sec": "map_open_duration_sec"})
    trials["map_time_share"] = trials["map_open_duration_sec"] / trials["completion_time_sec"]
    trials["compass_use"] = trials["compass_use_notes"].map(code_compass_use)

    first = trials[trials["trial_position"] == 1].set_index("participant_id")
    seq = pd.Series({p: SEQUENCES[(r["condition"], r["route_id"])] for p, r in first.iterrows()}, name="sequence")

    qs["nasa_tlx_raw"] = tlx_raw(qs)
    qs = pd.concat([qs, ueqs_scores(qs)], axis=1)

    q_keep = ["participant_id", "condition"] + TLX_COLS + ["nasa_tlx_raw", "ueqs_pragmatic",
                                                          "ueqs_hedonic", "ueqs_overall"]
    t_keep = ["participant_id", "condition", "route_id", "trial_position", "active_time_sec",
              "completion_time_sec", "completed", "timed_out", "wayfinding_errors", "error_rate",
              "lost_episodes", "map_openings", "map_open_duration_sec", "map_time_share",
              "objective_board_openings", "compass_use"]
    long = trials[t_keep].merge(qs[q_keep], on=["participant_id", "condition"], how="left",
                                validate="one_to_one")
    p_keep = ["participant_id", "age", "gender", "weekly_gaming_hours", "years_of_gaming",
              "open_world_game_experience", "previous_assassins_creed_play", "input_device"]
    long = long.merge(parts[p_keep], on="participant_id", how="left", validate="many_to_one")
    long = long.merge(seq, left_on="participant_id", right_index=True, how="left")

    long["condition"] = pd.Categorical(long["condition"], list(CONDITIONS), ordered=True)
    long["route_id"] = pd.Categorical(long["route_id"], list(ROUTES), ordered=True)
    for c in ["gender", "input_device"]:
        long[c] = long[c].astype("category")
    for c in ["open_world_game_experience", "previous_assassins_creed_play"]:
        long[c] = long[c].map({"Yes": True, "No": False}).astype("boolean")
    for c in COUNT_COLS + ["trial_position", "age", "compass_use", "sequence"]:
        long[c] = long[c].round().astype("Int64")
    long["completed"] = long["completed"].astype("boolean")
    long = long.sort_values(["participant_id", "trial_position"]).reset_index(drop=True)

    wide = make_wide(long)
    return long, wide


WIDE_MEASURES = list(MEASURES) + ["error_rate", "map_time_share", "compass_use"] + TLX_COLS


def make_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per participant; <measure>_X / _I / _diff (I minus X)."""
    base = ["participant_id", "sequence", "age", "gender", "weekly_gaming_hours",
            "years_of_gaming", "open_world_game_experience", "previous_assassins_creed_play",
            "input_device"]
    wide = long.drop_duplicates("participant_id")[base].set_index("participant_id")
    for m in WIDE_MEASURES:
        for cond in CONDITIONS:
            s = (long[long["condition"] == cond].set_index("participant_id")[m]).astype(float)
            wide[f"{m}_{COND_SHORT[cond]}"] = s
        wide[f"{m}_diff"] = wide[f"{m}_I"] - wide[f"{m}_X"]
    for cond in CONDITIONS:
        g = long[long["condition"] == cond].set_index("participant_id")
        wide[f"trial_position_{COND_SHORT[cond]}"] = g["trial_position"].astype(float)
        wide[f"route_{COND_SHORT[cond]}"] = g["route_id"].astype(str)
        wide[f"completed_{COND_SHORT[cond]}"] = g["completed"]
    return wide.reset_index()


# =============================================================================
# 7. Descriptive statistics (paper, Analysis plan, Step 1)
# =============================================================================
def fmt_num(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{x:.0f}"


def describe_by_condition(long: pd.DataFrame, wide: pd.DataFrame, measures) -> pd.DataFrame:
    """Median [min, max], mean (SD) per condition and paired direction counts."""
    rows = []
    for m in measures:
        r = {"measure": m, "label": MEASURES.get(m, (m, ""))[0], "unit": MEASURES.get(m, ("", ""))[1]}
        for cond in CONDITIONS:
            x = long.loc[long["condition"] == cond, m].astype(float).dropna()
            k = COND_SHORT[cond]
            r.update({f"n_{k}": len(x), f"median_{k}": x.median() if len(x) else np.nan,
                      f"min_{k}": x.min() if len(x) else np.nan,
                      f"max_{k}": x.max() if len(x) else np.nan,
                      f"mean_{k}": x.mean() if len(x) else np.nan,
                      f"sd_{k}": x.std(ddof=1) if len(x) > 1 else np.nan})
        d = wide[[f"{m}_X", f"{m}_I"]].dropna()
        r["n_pairs"] = len(d)
        r["n_higher_X"] = int((d[f"{m}_X"] > d[f"{m}_I"]).sum())
        r["n_higher_I"] = int((d[f"{m}_I"] > d[f"{m}_X"]).sum())
        r["n_equal"] = int((d[f"{m}_I"] == d[f"{m}_X"]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def format_descriptives(desc: pd.DataFrame, digits=1) -> pd.DataFrame:
    """Presentation table: 'median [min, max]' and 'mean (SD)' per condition."""
    out = pd.DataFrame({"Measure": desc["label"] + " (" + desc["unit"] + ")"})
    for k, name in (("X", "X"), ("I", "I")):
        out[f"{name} median [min, max]"] = [
            f"{fmt_num(a, digits)} [{fmt_num(b, digits)}, {fmt_num(c, digits)}]"
            for a, b, c in zip(desc[f"median_{k}"], desc[f"min_{k}"], desc[f"max_{k}"])]
        out[f"{name} mean (SD)"] = [f"{fmt_num(a, digits)} ({fmt_num(b, digits)})"
                                     for a, b in zip(desc[f"mean_{k}"], desc[f"sd_{k}"])]
    out["Higher under X / I / equal (n)"] = [f"{a} / {b} / {c}" for a, b, c in
                                         zip(desc["n_higher_X"], desc["n_higher_I"], desc["n_equal"])]
    out["Pairs (n)"] = desc["n_pairs"]
    return out


def completion_table(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("condition", observed=True)
    out = g.agg(trials=("participant_id", "count"),
                completed=("completed", lambda s: int(s.astype("boolean").sum())),
                timeouts=("timed_out", "sum")).reset_index()
    out["completion_rate_pct"] = (100 * out["completed"] / out["trials"]).round(1)
    return out


def describe_by_group(long: pd.DataFrame, group: str, measures, digits=1) -> pd.DataFrame:
    """Median [min, max] and n for each level of `group` (trial_position / route_id)
    within each condition; descriptive check of learning and route effects."""
    rows = []
    for (cond, lvl), g in long.groupby(["condition", group], observed=True):
        r = {"condition": str(cond), group: str(lvl), "n": len(g)}
        for m in measures:
            x = g[m].astype(float).dropna()
            r[m] = (f"{fmt_num(x.median(), digits)} [{fmt_num(x.min(), digits)}, {fmt_num(x.max(), digits)}]"
                    if len(x) else "n/a")
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_ci(x, stat=np.mean, n_boot=10_000, alpha=0.05, seed=2026):
    """Percentile bootstrap CI for a statistic of paired differences."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def paired_differences(wide: pd.DataFrame, measures, n_boot=10_000, seed=2026) -> pd.DataFrame:
    """Estimation (not significance testing): per-participant difference
    D = value(I) - value(X), with median, mean, bootstrap 95% CI of the mean and
    the standardised paired effect size dz = mean(D) / SD(D)."""
    rows = []
    for m in measures:
        d = wide[f"{m}_diff"].astype(float).dropna()
        lo, hi = bootstrap_ci(d, np.mean, n_boot, seed=seed)
        sd = d.std(ddof=1) if len(d) > 1 else np.nan
        rows.append({"measure": m, "label": MEASURES.get(m, (m, ""))[0], "n_pairs": len(d),
                     "median_diff": d.median() if len(d) else np.nan,
                     "mean_diff": d.mean() if len(d) else np.nan,
                     "ci95_low": lo, "ci95_high": hi,
                     "dz": d.mean() / sd if sd and sd > 0 else np.nan})
    return pd.DataFrame(rows)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone step-down)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def exploratory_wilcoxon(wide: pd.DataFrame, measures) -> pd.DataFrame:
    """OPTIONAL sensitivity analysis, NOT part of the analysis plan in the paper
    (which commits to descriptive reporting only). Paired Wilcoxon signed-rank
    with Holm correction. Report only if the paper is amended."""
    from scipy import stats
    rows = []
    for m in measures:
        d = wide[[f"{m}_X", f"{m}_I"]].dropna()
        diff = d[f"{m}_I"] - d[f"{m}_X"]
        if len(d) < 6 or (diff == 0).all():
            rows.append({"measure": m, "n_pairs": len(d), "W": np.nan, "p": np.nan})
            continue
        res = stats.wilcoxon(d[f"{m}_I"], d[f"{m}_X"], zero_method="wilcox", method="auto")
        rows.append({"measure": m, "n_pairs": len(d), "W": float(res.statistic), "p": float(res.pvalue)})
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out["p_holm"] = np.nan
    if ok.any():
        out.loc[ok, "p_holm"] = holm(out.loc[ok, "p"])
    return out


def counter_cases(wide: pd.DataFrame, measures) -> pd.DataFrame:
    """Participants whose direction of difference is opposite to (or tied with)
    the majority, per measure. These are the individual cases to read next to
    their covariates and interviews (paper, Analysis plan, Step 3)."""
    rows = []
    for m in measures:
        d = wide.set_index("participant_id")[f"{m}_diff"].astype(float).dropna()
        if d.empty:
            continue
        majority = 1 if (d > 0).sum() >= (d < 0).sum() else -1
        for pid, v in d.items():
            if v == 0 or np.sign(v) != majority:
                rows.append({"measure": MEASURES[m][0], "participant_id": pid, "difference_I_minus_X": v,
                             "direction": "no difference" if v == 0 else
                                          ("higher under I" if v > 0 else "higher under X")})
    return pd.DataFrame(rows, columns=["measure", "participant_id", "difference_I_minus_X", "direction"])


def headline_table(desc: pd.DataFrame, diffs: pd.DataFrame) -> pd.DataFrame:
    """One neutral descriptive sentence per measure, rebuilt from the data so the
    wording can never drift away from the numbers. No causal language."""
    d, e = desc.set_index("measure"), diffs.set_index("measure")
    rows = []
    for rq in sorted(RQ_MEASURES):
        for m in RQ_MEASURES[rq]:
            r, x = d.loc[m], e.loc[m]
            unit = MEASURES[m][1]
            rows.append({"RQ": rq, "measure": r["label"],
                         "sentence": (f"Median {fmt_num(r['median_I'])} under the more implicit condition vs "
                                      f"{fmt_num(r['median_X'])} under the more explicit one ({unit}); "
                                      f"{r['n_higher_I']} of {r['n_pairs']} participants scored higher under I, "
                                      f"{r['n_higher_X']} under X, {r['n_equal']} equal. Mean difference (I - X) "
                                      f"{fmt_num(x['mean_diff'], 2)} (bootstrap 95% CI {fmt_num(x['ci95_low'], 2)} "
                                      f"to {fmt_num(x['ci95_high'], 2)}).")})
    return pd.DataFrame(rows)


# =============================================================================
# 8. Participant description, covariates
# =============================================================================
def participant_table(parts: pd.DataFrame) -> pd.DataFrame:
    n = len(parts)
    rows = []

    def num(label, col):
        x = parts[col].astype(float)
        rows.append((label, f"median {fmt_num(x.median())} [{fmt_num(x.min())}, {fmt_num(x.max())}]; "
                            f"mean {fmt_num(x.mean())} (SD {fmt_num(x.std(ddof=1))})"))

    def cat(label, col):
        vc = parts[col].astype(str).value_counts()
        rows.append((label, "; ".join(f"{k}: {v} ({100*v/n:.0f}%)" for k, v in vc.items())))

    rows.append(("Participants (n)", str(n)))
    num("Age (years)", "age")
    cat("Gender", "gender")
    num("Gaming hours per week", "weekly_gaming_hours")
    num("Years of gaming", "years_of_gaming")
    cat("Open-world experience", "open_world_game_experience")
    cat("Prior Assassin's Creed play", "previous_assassins_creed_play")
    cat("Input device", "input_device")
    return pd.DataFrame(rows, columns=["Characteristic", "Summary"])


def sequence_table(wide: pd.DataFrame) -> pd.DataFrame:
    desc = {1: "R1 explicit, then R2 implicit", 2: "R2 implicit, then R1 explicit",
            3: "R1 implicit, then R2 explicit", 4: "R2 explicit, then R1 implicit"}
    vc = wide["sequence"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    target = len(wide) / 4
    return pd.DataFrame({"Sequence": [f"Sequence {i}" for i in vc.index],
                         "Trials": [desc[i] for i in vc.index],
                         "Participants": vc.values,
                         "Target": [f"{target:g}"] * 4})


def covariate_table(wide: pd.DataFrame, diff_measures, covariates) -> pd.DataFrame:
    """Descriptive Spearman rho between covariates and participants' I-X
    difference. Reported without p-values (n is small; paper: covariates are for
    reading individual cases, not for adjustment)."""
    from scipy import stats
    rows = []
    for m in diff_measures:
        for c in covariates:
            d = wide[[f"{m}_diff", c]].astype(float).dropna()
            rho = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 5 and d.nunique().min() > 1 else np.nan
            rows.append({"measure": MEASURES[m][0], "covariate": c, "n": len(d), "spearman_rho": rho})
    return pd.DataFrame(rows)


# =============================================================================
# 9. Inter-coder reliability (paper, 'Coding and data handling')
# =============================================================================
def cohens_kappa(a, b):
    """Cohen's kappa for two raters on the same items (nominal labels).
    Returns NaN when chance agreement is 1 (kappa undefined)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(a) != len(b):
        return np.nan
    cats = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return np.nan if np.isclose(pe, 1.0) else float((po - pe) / (1 - pe))


def kappa_label(k):
    """Landis & Koch (1977) verbal labels."""
    if k is None or np.isnan(k):
        return "undefined"
    if k < 0:
        return "poor"
    for upper, lab in [(0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if k <= upper:
            return lab
    return "almost perfect"


def percent_agreement(a, b, tol=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b) <= tol)) if len(a) else np.nan


def reliability_summary(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    e = rel[rel["measure"] == "error_code"]
    if len(e):
        rows.append({"measure": "Wayfinding error code (per decision point)", "method": "Cohen's kappa",
                     "units": len(e), "value": cohens_kappa(e["coder1"], e["coder2"]),
                     "percent_agreement": 100 * percent_agreement(e["coder1"], e["coder2"])})
    for m, g in rel[rel["measure"] != "error_code"].groupby("measure"):
        rows.append({"measure": m.replace("_", " ").capitalize() + " (per trial count)",
                     "method": "Exact % agreement", "units": len(g), "value": np.nan,
                     "percent_agreement": 100 * percent_agreement(g["coder1"], g["coder2"]),
                     "within_1_pct": 100 * percent_agreement(g["coder1"], g["coder2"], tol=1),
                     "mean_abs_diff": float(np.mean(np.abs(g["coder1"] - g["coder2"])))})
    out = pd.DataFrame(rows)
    if "value" in out:
        out["kappa_label"] = out["value"].map(lambda k: kappa_label(k) if not pd.isna(k) else "")
    return out


def reliability_consistency(rel: pd.DataFrame, trials: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    """Coder 1 values must reproduce the main trials table (sum of error codes ->
    wayfinding_errors; counts -> map_openings etc.)."""
    problems = []
    t = trials.set_index(["participant_id", "condition"])
    err = rel[rel["measure"] == "error_code"].groupby(["participant_id", "condition"])["coder1"].sum()
    for key, val in err.items():
        if key in t.index and t.loc[key, "wayfinding_errors"] != val:
            problems.append((*key, "wayfinding_errors", val, t.loc[key, "wayfinding_errors"]))
    for _, r in rel[rel["measure"] != "error_code"].iterrows():
        key = (r["participant_id"], r["condition"])
        if key in t.index and r["measure"] in t.columns and t.loc[key, r["measure"]] != r["coder1"]:
            problems.append((*key, r["measure"], r["coder1"], t.loc[key, r["measure"]]))
    for p in problems:
        log.add("reliability", "coder1_differs_from_trials", "flagged", participant_id=p[0],
                field=p[2], original=p[3], new=p[4], detail=f"{p[1]}: coder 1 {p[3]} vs trials table {p[4]}")
    return pd.DataFrame(problems, columns=["participant_id", "condition", "measure",
                                           "coder1", "trials_table"])


# =============================================================================
# 10. Interviews: codebook, thematic tables, joint display (Analysis plan, 2-3)
# =============================================================================
CODEBOOK = {   # code -> (family, group, label)
    "marker": ("source", "External aids", "Objective marker"),
    "compass": ("source", "External aids", "Compass"),
    "map": ("source", "External aids", "Map"),
    "clue_text": ("source", "Environment and narrative", "Clue / quest text"),
    "landmarks_terrain": ("source", "Environment and narrative", "Landmarks and terrain"),
    "lighting_colour": ("source", "Environment and narrative", "Lighting and colour"),
    "character_clues": ("source", "Environment and narrative", "Character-given clues"),
    "ambiguous_clue": ("difficulty", "Wayfinding difficulty", "Clue was ambiguous"),
    "missed_branch": ("difficulty", "Wayfinding difficulty", "Missed or misread a branch"),
    "lost_orientation": ("difficulty", "Wayfinding difficulty", "Lost orientation"),
    "map_hard_to_relate": ("difficulty", "Wayfinding difficulty", "Hard to relate map to 3D view"),
    "more_confident": ("experience", "Experience", "Felt confident"),
    "more_effortful": ("experience", "Experience", "Felt effortful"),
    "more_immersive": ("experience", "Experience", "Felt immersive / enjoyable"),
    "frustrating": ("experience", "Experience", "Felt frustrating"),
}
# Candidate themes (to be replaced by the researchers' own themes after coding).
THEMES = {
    "Marker-led following": ["marker", "more_confident"],
    "Interpreting clues with the environment": ["clue_text", "landmarks_terrain",
                                                "lighting_colour", "character_clues"],
    "Map and compass as fallback": ["map", "compass", "map_hard_to_relate"],
    "Getting lost and recovering": ["ambiguous_clue", "missed_branch", "lost_orientation"],
    "Effort versus immersion": ["more_effortful", "more_immersive", "frustrating"],
}


def code_participant_counts(codes: pd.DataFrame, n_participants: int) -> pd.DataFrame:
    """Number of distinct participants (and segments) per code and condition.
    Final-interview segments with a condition are included; segments without a
    condition are excluded from per-condition counts."""
    c = codes.dropna(subset=["condition"])
    pc = (c.groupby(["code", "condition"])["participant_id"].nunique().unstack(fill_value=0)
            .reindex(columns=list(CONDITIONS), fill_value=0))
    seg = (c.groupby(["code", "condition"]).size().unstack(fill_value=0)
             .reindex(columns=list(CONDITIONS), fill_value=0))
    out = pd.DataFrame({"participants_X": pc["Explicit"], "participants_I": pc["Implicit"],
                        "segments_X": seg["Explicit"], "segments_I": seg["Implicit"]})
    out["n_participants"] = n_participants
    out = out.reset_index()
    out["family"] = out["code"].map(lambda k: CODEBOOK.get(k, ("emergent", "Emergent", k))[0])
    out["group"] = out["code"].map(lambda k: CODEBOOK.get(k, ("emergent", "Emergent", k))[1])
    out["label"] = out["code"].map(lambda k: CODEBOOK.get(k, ("emergent", "Emergent", k))[2])
    return out


def theme_participants(codes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    c = codes.dropna(subset=["condition"])
    for theme, cl in THEMES.items():
        r = {"Theme": theme, "Codes": ", ".join(cl)}
        for cond in CONDITIONS:
            s = c[(c["condition"] == cond) & (c["code"].isin(cl))]
            r[f"participants_{COND_SHORT[cond]}"] = s["participant_id"].nunique()
            r[f"segments_{COND_SHORT[cond]}"] = len(s)
        rows.append(r)
    return pd.DataFrame(rows)


RELATED_WORK = {   # only sources cited in the paper
    "RQ2": "Xiao (2020): HUD aids vs. cues built into the environment; reading a map means relating it "
           "to the 3D view. Moura & Bartram (2014): responses to cues depend on game context.",
    "RQ3": "Lankes et al. (2026): extra movement alone cannot show exploring vs. struggling, so time is "
           "read with errors, lost episodes and interviews. Johanson et al. (2017): assistance affects "
           "performance.",
    "RQ4": "Iacovides et al. (2015): removing interface elements changed involvement, depending on "
           "expertise. Gardony et al. (2013): aids can divert attention from the environment.",
}


def joint_display(desc: pd.DataFrame, theme_cnt: pd.DataFrame) -> pd.DataFrame:
    """Step 3 of the analysis plan: one row per RQ with the descriptive pattern,
    supporting/contradicting themes and related work. A pattern with no coded
    theme at all in the relevant theme set is reported as 'unexplained'."""
    rq_themes = {"RQ2": ["Marker-led following", "Interpreting clues with the environment",
                         "Map and compass as fallback"],
                 "RQ3": ["Getting lost and recovering", "Interpreting clues with the environment"],
                 "RQ4": ["Effort versus immersion", "Marker-led following"]}
    rows = []
    d = desc.set_index("measure")
    for rq in sorted(RQ_MEASURES):
        measures = RQ_MEASURES[rq]
        pat = []
        for m in measures:
            r = d.loc[m]
            pat.append(f"{r['label']}: median {fmt_num(r['median_X'])} (X) vs {fmt_num(r['median_I'])} (I); "
                       f"higher under I in {r['n_higher_I']}/{r['n_pairs']}")
        th = theme_cnt.set_index("Theme")
        sup = [f"{t}: {th.loc[t,'participants_X']} (X) / {th.loc[t,'participants_I']} (I) participants"
               for t in rq_themes[rq]]
        any_coded = sum(int(th.loc[t, "participants_X"] + th.loc[t, "participants_I"]) for t in rq_themes[rq]) > 0
        rows.append({"RQ": rq, "Descriptive pattern": "\n".join(pat),
                     "Interview themes": "\n".join(sup) if any_coded else "unexplained: no coded theme",
                     "Related work": RELATED_WORK[rq]})
    return pd.DataFrame(rows)


# =============================================================================
# 11. Input / output
# =============================================================================
def load_raw(data_dir, suffix="_dummy") -> dict:
    """Read the study CSVs as text (dtype=str) so that nothing is silently
    coerced before cleaning. Optional tables are skipped if absent."""
    from pathlib import Path
    d = Path(data_dir)
    names = {"participants": "participants", "trials": "trials", "questionnaires": "questionnaires",
             "reliability": "reliability", "interviews": "interviews_coded"}
    raw = {}
    for key, stem in names.items():
        p = d / f"{stem}{suffix}.csv"
        if p.exists():
            raw[key] = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[])
        elif key not in OPTIONAL_TABLES:
            raise FileNotFoundError(f"Required file not found: {p}")
    return raw


def run_pipeline(data_dir=".", suffix="_dummy"):
    """Convenience wrapper: load -> clean -> prepare -> outlier flags."""
    raw = load_raw(data_dir, suffix)
    clean = clean_all(raw)
    long, wide = prepare(clean)
    flag_outliers(long, ["completion_time_sec", "wayfinding_errors", "map_openings",
                         "nasa_tlx_raw", "ueqs_overall"], clean.log)
    if "reliability" in clean.tables:
        reliability_consistency(clean.tables["reliability"], clean.trials, clean.log)
    return raw, clean, long, wide
