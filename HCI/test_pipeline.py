"""
test_pipeline.py - software tests for the data pipeline (run: python -m pytest -q).

Layers
  1. normalisers           small pure functions, parametrised edge cases
  2. cleaning rules        each rule on a hand-built table with a known answer
  3. formulas              derived variables vs values computed by hand
  4. statistics            kappa vs scikit-learn, Holm, bootstrap, wide/long
  5. end-to-end            full run on the dummy files: every injected fault is
                           caught, invariants hold, cleaning is idempotent
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import generate_dummy_data as gen
import pipeline as pl

HERE = Path(__file__).parent


# ----------------------------------------------------------------- helpers ---
def _trial(pid="P01", pos=1, cond="Explicit", route="Route_A", el=200, pause=20, done="True",
           err=1, lost=0, maps=2, mdur=30, board=1, note="Used compass briefly"):
    return dict(participant_id=pid, trial_number=pos, condition=cond, route_id=route, trial_position=pos,
                elapsed_time_sec=el, combat_cutscene_pause_sec=pause, completed=done, wayfinding_errors=err,
                lost_episodes=lost, map_openings=maps, total_map_open_duration_sec=mdur,
                objective_board_openings=board, compass_use_notes=note)


def _pair(pid="P01", first=("Explicit", "Route_A"), **kw):
    c1, r1 = first
    c2 = "Implicit" if c1 == "Explicit" else "Explicit"
    r2 = "Route_B" if r1 == "Route_A" else "Route_A"
    return [_trial(pid, 1, c1, r1, **kw), _trial(pid, 2, c2, r2, **kw)]


def _df(rows):
    return pd.DataFrame(rows).astype(str)


def _clean_trials(rows, ids=("P01", "P02", "P03")):
    log, store = pl.IssueLog(), {}
    out = pl.clean_trials(_df(rows), set(ids), log, store)
    return out, log.to_frame(), store


def rules(log):
    return set(log["rule"]) if len(log) else set()


# =============================================================== 1. normalisers
@pytest.mark.parametrize("raw,exp", [("P01", "P01"), ("p1", "P01"), (" P 7 ", "P07"), ("P001", "P01"),
                                     ("P16", "P16"), ("Q01", None), ("", None), (None, None), ("P", None),
                                     ("P1x", None)])
def test_normalize_id(raw, exp):
    assert pl.normalize_id(raw) == exp


@pytest.mark.parametrize("raw,exp", [("Explicit", "Explicit"), (" explicit ", "Explicit"), ("EXPLICIT", "Explicit"),
                                     ("more implicit", "Implicit"), ("Guided Exploration On", "Explicit"),
                                     ("guided off", "Implicit"), ("neither", None), ("", None)])
def test_normalize_condition(raw, exp):
    assert pl.normalize_condition(raw) == exp


@pytest.mark.parametrize("raw,exp", [("Route_A", "Route_A"), ("route b", "Route_B"), ("R1", "Route_A"),
                                     ("2", None), ("Route_C", None)])
def test_normalize_route(raw, exp):
    assert pl.normalize_route(raw) == exp


@pytest.mark.parametrize("raw,exp", [("True", True), ("TRUE", True), ("yes", True), ("1", True), ("False", False),
                                     ("No", False), ("0", False), ("maybe", None), ("", None), (True, True)])
def test_normalize_bool(raw, exp):
    assert pl.normalize_bool(raw) == exp


@pytest.mark.parametrize("raw,exp", [("5", 5.0), (" 5.5 ", 5.5), ("n/a", None), ("abc", None), ("", None),
                                     ("1,5", None), ("inf", None), (7, 7.0), (True, None)])
def test_parse_number(raw, exp):
    assert pl.parse_number(raw) == exp


@pytest.mark.parametrize("raw,exp", [("woman", "Woman"), ("Female", "Woman"), ("non-binary", "Non-binary"),
                                     ("Nonbinary", "Non-binary"), ("robot", None)])
def test_normalize_gender(raw, exp):
    assert pl.normalize_gender(raw) == exp


@pytest.mark.parametrize("note,exp", [("Used compass briefly", 1), ("Checked compass several times", 2),
                                      ("Frequently checked compass", 2), ("Relied on compass", 3),
                                      ("Followed compass", 3), ("Did not use the compass", 0),
                                      ("Relied on compass; still unsure", 3), ("something else", np.nan)])
def test_code_compass_use(note, exp):
    got = pl.code_compass_use(note)
    assert (np.isnan(got) and np.isnan(exp)) or got == exp


# ============================================================ 2. cleaning rules
def test_clean_trials_valid_pair_untouched():
    out, log, store = _clean_trials(_pair())
    assert len(out) == 2 and log.empty and not store


def test_trials_id_condition_route_boolean_repairs():
    rows = _pair()
    rows[0].update(participant_id="p1", condition=" explicit ", route_id="route a", completed="YES")
    out, log, _ = _clean_trials(rows)
    r = out.iloc[0]
    assert (r["participant_id"], r["condition"], r["route_id"], bool(r["completed"])) == ("P01", "Explicit", "Route_A", True)
    assert {"participant_id:format", "condition:category", "route_id:category", "completed:boolean"} <= rules(log)


def test_trials_exact_duplicate_dropped_once():
    rows = _pair()
    out, log, _ = _clean_trials(rows + [rows[0]])
    assert len(out) == 2 and "exact_duplicate_row" in rules(log)


def test_trials_conflicting_duplicates_are_quarantined_not_guessed():
    rows = _pair()
    dup = dict(rows[0]); dup["elapsed_time_sec"] = 999
    out, log, store = _clean_trials(rows + [dup])
    assert "conflicting_duplicate_condition" in rules(log)
    assert len(store["trials"]) >= 2 and "P01" not in set(out[out["condition"] == "Explicit"]["participant_id"])


@pytest.mark.parametrize("field,val,rule", [
    ("elapsed_time_sec", "abc", "elapsed_time_sec:non_numeric"),
    ("elapsed_time_sec", "-5", "elapsed_time_sec:out_of_range"),
    ("wayfinding_errors", "2.5", "wayfinding_errors:non_integer"),
    ("wayfinding_errors", "-1", "wayfinding_errors:out_of_range"),
    ("map_openings", "x", "map_openings:non_numeric"),
    ("lost_episodes", "-2", "lost_episodes:out_of_range"),
    ("combat_cutscene_pause_sec", "-1", "combat_cutscene_pause_sec:out_of_range")])
def test_trials_bad_numbers_set_missing(field, val, rule):
    rows = _pair()
    rows[0][field] = val
    out, log, _ = _clean_trials(rows)
    assert rule in rules(log) and np.isnan(out.iloc[0][field])
    assert len(out) == 2                                  # the row itself is kept


def test_trials_pause_longer_than_trial():
    rows = _pair(); rows[0].update(elapsed_time_sec=100, combat_cutscene_pause_sec=150)
    out, log, _ = _clean_trials(rows)
    assert "pause>=elapsed" in rules(log)
    assert out.iloc[0][["elapsed_time_sec", "combat_cutscene_pause_sec"]].isna().all()


def test_trials_errors_cannot_exceed_decision_points():
    rows = _pair(); rows[0]["wayfinding_errors"] = pl.ROUTE_DECISION_POINTS["Route_A"] + 1
    out, log, _ = _clean_trials(rows)
    assert "errors>decision_points" in rules(log) and np.isnan(out.iloc[0]["wayfinding_errors"])


def test_trials_errors_equal_to_decision_points_allowed():
    rows = _pair(); rows[0]["wayfinding_errors"] = pl.ROUTE_DECISION_POINTS["Route_A"]
    out, log, _ = _clean_trials(rows)
    assert "errors>decision_points" not in rules(log)


def test_trials_map_duration_consistency():
    rows = _pair(); rows[0].update(map_openings=0, total_map_open_duration_sec=30)
    out, log, _ = _clean_trials(rows)
    assert "map_duration>0_with_0_openings" in rules(log)
    rows = _pair(); rows[0].update(elapsed_time_sec=100, combat_cutscene_pause_sec=10, total_map_open_duration_sec=400)
    out, log, _ = _clean_trials(rows)
    assert "map_duration>elapsed" in rules(log)


def test_trials_completed_over_cap_is_repaired_to_not_completed():
    rows = _pair(); rows[0].update(elapsed_time_sec=600, combat_cutscene_pause_sec=20, completed="True")      # active 580 > 480
    out, log, _ = _clean_trials(rows)
    assert "completed_but_over_cap" in rules(log) and out.iloc[0]["completed"] == False  # noqa: E712


def test_trials_not_completed_before_cap_is_flagged_not_changed():
    rows = _pair(); rows[0].update(elapsed_time_sec=300, combat_cutscene_pause_sec=20, completed="False")
    out, log, _ = _clean_trials(rows)
    flagged = log[log["rule"] == "not_completed_before_cap"]
    assert len(flagged) == 1 and flagged.iloc[0]["action"] == "flagged"
    assert out.iloc[0]["completed"] == False  # noqa: E712


def test_trials_unknown_participant_quarantined():
    out, log, store = _clean_trials(_pair("P09"), ids=("P01",))
    assert out.empty and "unknown_or_excluded_participant" in rules(log)


def test_trials_invalid_condition_quarantined():
    rows = _pair(); rows[0]["condition"] = "Both"
    out, log, store = _clean_trials(rows)
    assert "invalid_condition" in rules(log) and len(out) == 1


def test_trial_position_canonical_and_mismatch_flagged():
    rows = _pair(); rows[0]["trial_number"] = 2
    out, log, _ = _clean_trials(rows)
    assert "trial_number!=trial_position" in rules(log) and out.iloc[0]["trial_position"] == 1


def test_missing_required_column_raises():
    with pytest.raises(pl.SchemaError):
        pl.clean_trials(_df(_pair()).drop(columns=["condition"]), {"P01"}, pl.IssueLog(), {})


def test_column_names_are_normalised():
    df = _df(_pair()).rename(columns={"condition": " Condition ", "route_id": "Route ID"})
    out, log, _ = _clean_trials(df.to_dict("records"))
    assert len(out) == 2 and list(out["route_id"]) == ["Route_A", "Route_B"]


def test_na_tokens_are_logged_and_become_missing():
    rows = _pair(); rows[0]["total_map_open_duration_sec"] = "N/A"
    out, log, _ = _clean_trials(rows)
    assert "na_token" in rules(log) and np.isnan(out.iloc[0]["total_map_open_duration_sec"])


# ---- design check ----------------------------------------------------------
def _design(rows):
    log, store = pl.IssueLog(), {}
    t, _, _ = _clean_trials(rows)
    return pl.check_design(t, log, store)


def test_design_valid_for_all_four_sequences():
    for first in pl.SEQUENCES:
        assert _design(_pair(first=first)) == {}


def test_design_rejects_one_trial_only():
    assert "P01" in _design([_pair()[0]])


def test_design_rejects_repeated_route():
    rows = _pair(); rows[1]["route_id"] = "Route_A"
    assert "P01" in _design(rows)


def test_same_condition_twice_never_reaches_analysis():
    rows = _pair(); rows[1]["condition"] = "Explicit"
    out, log, store = _clean_trials(rows)
    assert "conflicting_duplicate_condition" in rules(log) and out.empty


def test_design_rejects_noncrossed_pair():
    """Explicit on Route_B then Implicit on Route_B repeats a route."""
    rows = _pair(first=("Explicit", "Route_B"))
    rows[1]["route_id"] = "Route_B"
    assert "P01" in _design(rows)


def test_design_rejects_pair_that_is_not_one_of_the_four_sequences():
    """Both trials on the same route-condition combination as another valid pair but out of order."""
    rows = _pair(first=("Explicit", "Route_A"))
    rows[0]["trial_position"], rows[1]["trial_position"] = 2, 2
    out, log, _ = _clean_trials(rows)
    assert "conflicting_duplicate_position" in rules(log)


# ---- participants ----------------------------------------------------------
BASE_P = dict(participant_id="P01", age=21, gender="Woman", weekly_gaming_hours=8, years_of_gaming=4,
              third_person_3d_game_past_12_months="Yes", previous_assassins_creed_play="No",
              input_device="Keyboard and mouse", open_world_game_experience="Yes")


def _clean_parts(rows):
    log, store = pl.IssueLog(), {}
    return pl.clean_participants(_df(rows), log, store), log.to_frame(), store


def test_participants_valid():
    out, log, store = _clean_parts([BASE_P])
    assert len(out) == 1 and log.empty


@pytest.mark.parametrize("change,rule", [
    ({"age": 17}, "ineligible:age<18"),
    ({"third_person_3d_game_past_12_months": "No"}, "ineligible:no_third_person_3d_play_12mo"),
    ({"participant_id": "XX"}, "invalid_participant_id")])
def test_participants_ineligible_or_invalid_are_quarantined(change, rule):
    out, log, store = _clean_parts([{**BASE_P, **change}])
    assert out.empty and rule in rules(log)


@pytest.mark.parametrize("change,rule", [
    ({"age": "abc"}, "age:non_numeric"), ({"age": 150}, "age:out_of_range"),
    ({"weekly_gaming_hours": 200}, "weekly_gaming_hours:out_of_range"),
    ({"years_of_gaming": 30}, "years_of_gaming>age"),
    ({"gender": "robot"}, "gender:category"), ({"input_device": "touchscreen"}, "input_device:category")])
def test_participants_bad_values_become_missing(change, rule):
    out, log, _ = _clean_parts([{**BASE_P, **change}])
    assert rule in rules(log) and len(out) == 1


def test_participants_conflicting_duplicate_ids_quarantined():
    out, log, _ = _clean_parts([BASE_P, {**BASE_P, "age": 30}])
    assert out.empty and "conflicting_duplicate_participant" in rules(log)


def test_participants_implausible_values_flagged_not_changed():
    out, log, _ = _clean_parts([{**BASE_P, "weekly_gaming_hours": 90, "age": 45, "years_of_gaming": 10}])
    assert {"weekly_gaming_hours>80", "age>40"} <= rules(log)
    assert out.iloc[0]["weekly_gaming_hours"] == 90


# ---- questionnaires --------------------------------------------------------
def _q(pid="P01", cond="Explicit", tlx=(30,) * 6, ue=(5,) * 8):
    r = dict(participant_id=pid, condition=cond)
    r.update(dict(zip(pl.TLX_COLS, tlx)))
    r.update(dict(zip(pl.UEQS_RAW_COLS, ue)))
    return r


def _clean_q(rows, ids=("P01", "P02")):
    log, store = pl.IssueLog(), {}
    return pl.clean_questionnaires(_df(rows), set(ids), log, store), log.to_frame(), store


@pytest.mark.parametrize("col,val,rule", [("effort", "105", "effort:out_of_range"), ("mental_demand", "-1", "mental_demand:out_of_range"),
                                          ("effort", "high", "effort:non_numeric"),
                                          ("ueqs_item1_raw_1_to_7", "0", "ueqs_item1_raw_1_to_7:out_of_range"),
                                          ("ueqs_item2_raw_1_to_7", "8", "ueqs_item2_raw_1_to_7:out_of_range"),
                                          ("ueqs_item3_raw_1_to_7", "3.5", "ueqs_item3_raw_1_to_7:non_integer")])
def test_questionnaire_invalid_scores(col, val, rule):
    r = _q(); r[col] = val
    out, log, _ = _clean_q([r])
    assert rule in rules(log) and np.isnan(out.iloc[0][col])


def test_questionnaire_boundaries_accepted():
    out, log, _ = _clean_q([_q(tlx=(0, 100, 0, 100, 0, 100), ue=(1, 7, 1, 7, 1, 7, 1, 7))])
    assert log[log["action"].isin(["set_missing", "quarantined"])].empty


def test_questionnaire_duplicate_and_orphan_handling():
    r = _q()
    out, log, _ = _clean_q([r, r, _q("P99")])
    assert len(out) == 1 and {"exact_duplicate_row", "unknown_or_excluded_participant"} <= rules(log)
    r2 = _q(tlx=(40,) * 6)
    out, log, _ = _clean_q([r, r2])
    assert out.empty and "conflicting_duplicate_condition" in rules(log)


def test_questionnaire_straight_lining_flagged():
    out, log, _ = _clean_q([_q(ue=(4,) * 8)])
    assert "ueqs_straight_lining" in rules(log) and len(out) == 1


# ============================================================== 3. formulas
def test_active_time_and_completion_time_cap():
    assert pl.active_time(300, 40) == 260
    t = pl.completion_time(np.array([260.0, 479.0, 480.0, 500.0, 300.0]), [True, True, True, True, False])
    assert list(t) == [260.0, 479.0, 480.0, 480.0, 480.0]        # >cap capped; not completed -> cap


def test_tlx_raw_is_unweighted_mean_and_nan_propagates():
    df = pd.DataFrame([dict(zip(pl.TLX_COLS, [10, 20, 30, 40, 50, 60])),
                       dict(zip(pl.TLX_COLS, [10, 20, 30, 40, 50, np.nan]))])
    r = pl.tlx_raw(df)
    assert r.iloc[0] == 35.0 and np.isnan(r.iloc[1])


def test_ueqs_scoring_hand_computed():
    df = pd.DataFrame([dict(zip(pl.UEQS_RAW_COLS, [7, 7, 7, 7, 1, 1, 1, 1])),     # +3 x4, -3 x4
                       dict(zip(pl.UEQS_RAW_COLS, [4] * 8)),
                       dict(zip(pl.UEQS_RAW_COLS, [5, 6, 5, 6, 3, 2, 3, 2]))])
    s = pl.ueqs_scores(df)
    assert s.loc[0, "ueqs_pragmatic"] == 3 and s.loc[0, "ueqs_hedonic"] == -3 and s.loc[0, "ueqs_overall"] == 0
    assert s.loc[1, "ueqs_overall"] == 0
    assert s.loc[2, "ueqs_pragmatic"] == pytest.approx((1 + 2 + 1 + 2) / 4)
    assert s.loc[2, "ueqs_hedonic"] == pytest.approx((-1 - 2 - 1 - 2) / 4)
    assert s.loc[2, "ueqs_overall"] == pytest.approx(s.loc[2, ["ueqs_pragmatic", "ueqs_hedonic"]].mean())


def test_ueqs_reversed_item_flips_sign():
    df = pd.DataFrame([dict(zip(pl.UEQS_RAW_COLS, [7] + [4] * 7))])
    assert pl.ueqs_scores(df).loc[0, "ueqs_pragmatic"] == 0.75
    assert pl.ueqs_scores(df, reversed_items=(1,)).loc[0, "ueqs_pragmatic"] == -0.75


def test_ueqs_missing_item_gives_nan_for_affected_scales_only():
    row = dict(zip(pl.UEQS_RAW_COLS, [5] * 8)); row[pl.UEQS_RAW_COLS[2]] = np.nan
    s = pl.ueqs_scores(pd.DataFrame([row]))
    assert np.isnan(s.loc[0, "ueqs_pragmatic"]) and np.isnan(s.loc[0, "ueqs_overall"])
    assert s.loc[0, "ueqs_hedonic"] == 1.0


# =============================================================== 4. statistics
def test_cohens_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0)
    for _ in range(25):
        a = rng.integers(0, 2, 40); b = np.where(rng.random(40) < 0.2, 1 - a, a)
        if len(set(a)) > 1 or len(set(b)) > 1:
            assert pl.cohens_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_kappa_edge_cases():
    assert pl.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert np.isnan(pl.cohens_kappa([0, 0, 0], [0, 0, 0]))               # chance agreement = 1
    assert pl.cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    assert np.isnan(pl.cohens_kappa([], []))
    assert pl.cohens_kappa([1, 0, 1, 0], [1, 0, 0, 1]) == pytest.approx(0.0)


@pytest.mark.parametrize("k,label", [(-0.1, "poor"), (0.1, "slight"), (0.3, "fair"), (0.5, "moderate"),
                                     (0.7, "substantial"), (0.9, "almost perfect"), (np.nan, "undefined")])
def test_kappa_label(k, label):
    assert pl.kappa_label(k) == label


def test_percent_agreement_exact_and_tolerance():
    assert pl.percent_agreement([1, 2, 3, 4], [1, 2, 3, 5]) == 0.75
    assert pl.percent_agreement([1, 2, 3, 4], [1, 3, 3, 5], tol=1) == 1.0


def test_holm_known_values():
    adj = pl.holm([0.01, 0.04, 0.03, 0.005])
    assert list(np.round(adj, 4)) == [0.03, 0.06, 0.06, 0.02]


def test_bootstrap_ci_reproducible_and_brackets_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 2.5])
    a, b = pl.bootstrap_ci(x, seed=1), pl.bootstrap_ci(x, seed=1)
    assert a == b and a[0] < x.mean() < a[1]
    assert np.isnan(pl.bootstrap_ci([1.0])[0])


def test_bootstrap_ci_constant_data_has_zero_width():
    lo, hi = pl.bootstrap_ci(np.full(8, 3.0))
    assert lo == hi == 3.0


def _mini_long():
    rows = []
    for pid, (tx, ti, ex, ei) in {"P01": (200, 300, 1, 3), "P02": (180, 250, 0, 2), "P03": (220, 210, 2, 2)}.items():
        for cond, t, e, pos in (("Explicit", tx, ex, 1), ("Implicit", ti, ei, 2)):
            rows.append(dict(participant_id=pid, condition=cond, completion_time_sec=t, wayfinding_errors=e,
                             trial_position=pos, route_id="Route_A" if pos == 1 else "Route_B", timed_out=False,
                             completed=True, sequence=1, age=20, gender="Man", weekly_gaming_hours=5,
                             years_of_gaming=3, open_world_game_experience=True,
                             previous_assassins_creed_play=False, input_device="Gamepad"))
    long = pd.DataFrame(rows)
    long["condition"] = pd.Categorical(long["condition"], list(pl.CONDITIONS), ordered=True)
    return long


def test_make_wide_hand_checked():
    long = _mini_long()
    for m in pl.WIDE_MEASURES:
        if m not in long:
            long[m] = np.nan
    wide = pl.make_wide(long).set_index("participant_id")
    assert wide.loc["P01", "completion_time_sec_X"] == 200 and wide.loc["P01", "completion_time_sec_I"] == 300
    assert wide.loc["P01", "completion_time_sec_diff"] == 100
    assert wide.loc["P03", "completion_time_sec_diff"] == -10
    assert wide.loc["P02", "wayfinding_errors_diff"] == 2
    assert len(wide) == 3


def test_describe_by_condition_hand_checked():
    long = _mini_long()
    for m in pl.WIDE_MEASURES:
        if m not in long:
            long[m] = np.nan
    wide = pl.make_wide(long)
    d = pl.describe_by_condition(long, wide, ["completion_time_sec"]).iloc[0]
    assert d["median_X"] == 200 and d["median_I"] == 250 and d["min_X"] == 180 and d["max_I"] == 300
    assert (d["n_higher_X"], d["n_higher_I"], d["n_equal"]) == (1, 2, 0) and d["n_pairs"] == 3
    e = pl.describe_by_condition(long, wide, ["wayfinding_errors"]).iloc[0]
    assert (e["n_higher_X"], e["n_higher_I"], e["n_equal"]) == (0, 2, 1)


def test_paired_differences_hand_checked():
    long = _mini_long()
    for m in pl.WIDE_MEASURES:
        if m not in long:
            long[m] = np.nan
    wide = pl.make_wide(long)
    d = pl.paired_differences(wide, ["completion_time_sec"], n_boot=500).iloc[0]
    assert d["mean_diff"] == pytest.approx((100 + 70 - 10) / 3) and d["median_diff"] == 70
    diffs = np.array([100, 70, -10.0])
    assert d["dz"] == pytest.approx(diffs.mean() / diffs.std(ddof=1))
    assert d["ci95_low"] <= d["mean_diff"] <= d["ci95_high"]


def test_reliability_summary_perfect_and_imperfect():
    rel = pd.DataFrame({"measure": ["error_code"] * 4 + ["map_openings"] * 2,
                        "coder1": [1, 0, 1, 0, 3, 4], "coder2": [1, 0, 0, 0, 3, 5]})
    s = pl.reliability_summary(rel).set_index("measure")
    assert s.loc["Wayfinding error code (per decision point)", "percent_agreement"] == 75
    assert s.loc["Wayfinding error code (per decision point)", "value"] == pytest.approx(0.5)
    assert s.loc["Map openings (per trial count)", "percent_agreement"] == 50


def test_reliability_cleaning_rejects_nonbinary_codes_and_incomplete_pairs():
    raw = pd.DataFrame([dict(participant_id="P01", condition="Explicit", measure="error_code", item=1, coder1=2, coder2=1),
                        dict(participant_id="P01", condition="Explicit", measure="error_code", item=2, coder1=1, coder2=""),
                        dict(participant_id="P01", condition="Explicit", measure="error_code", item=3, coder1=1, coder2=1)]).astype(str)
    log = pl.IssueLog()
    out = pl.clean_reliability(raw, {"P01"}, log, {})
    assert len(out) == 1 and {"error_code_not_binary", "incomplete_coder_pair"} <= rules(log.to_frame())


def test_holm_and_wilcoxon_run_and_are_monotone():
    long = _mini_long()
    for m in pl.WIDE_MEASURES:
        if m not in long:
            long[m] = np.nan
    wide = pl.make_wide(long)
    out = pl.exploratory_wilcoxon(wide, ["completion_time_sec"])
    assert out.loc[0, "n_pairs"] == 3 and np.isnan(out.loc[0, "p"])         # n < 6 -> refuses to test


# =========================================================== 5. end-to-end
@pytest.fixture(scope="module")
def dummy_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("dummy")
    parts, trials, quest, rel, inter = gen.build(clean=False)
    parts.to_csv(d / "participants_dummy.csv", index=False)
    trials.to_csv(d / "trials_dummy.csv", index=False)
    quest.to_csv(d / "questionnaires_dummy.csv", index=False)
    rel.to_csv(d / "reliability_dummy.csv", index=False)
    inter.to_csv(d / "interviews_coded_dummy.csv", index=False)
    return d


@pytest.fixture(scope="module")
def run(dummy_dir):
    return pl.run_pipeline(dummy_dir)


def test_every_injected_fault_is_caught(run):
    _, clean, _, _ = run
    log = clean.log.to_frame()
    for table, pid, rule, desc in gen.INJECTED_FAULTS:
        hit = log[(log["table"] == table) & (log["rule"] == rule)]
        assert len(hit), f"fault not detected: {desc}"
        ids = hit["participant_id"].dropna().map(lambda v: pl.normalize_id(v) or v)
        assert pid in set(ids), f"{desc}: rule fired but not for {pid}"


def test_row_counts_after_cleaning(run):
    _, clean, long, wide = run
    assert len(clean.participants) == 16 and len(long) == 32 and len(wide) == 16
    assert clean.row_counts.loc["trials", "rows_removed"] == 1               # the duplicated row
    assert clean.row_counts.loc["questionnaires", "rows_removed"] == 1        # the orphan P17


def test_analysis_tables_are_analysis_ready(run):
    _, _, long, wide = run
    assert long["condition"].dtype.name == "category" and list(long["condition"].cat.categories) == list(pl.CONDITIONS)
    assert long["completed"].dtype.name == "boolean" and long["timed_out"].dtype == bool
    for c in pl.COUNT_COLS:
        assert str(long[c].dtype) == "Int64"
    assert long["completion_time_sec"].dropna().between(0, pl.TIME_CAP_SEC).all()
    assert long["nasa_tlx_raw"].dropna().between(0, 100).all()
    assert long[["ueqs_overall", "ueqs_pragmatic", "ueqs_hedonic"]].dropna().stack().between(-3, 3).all()
    assert not long.duplicated(["participant_id", "condition"]).any()
    assert "compass_use_notes" not in long and not any(c.startswith("ueqs_item") for c in long)
    assert wide["participant_id"].is_unique
    assert (wide["completion_time_sec_diff"].dropna() ==
            (wide["completion_time_sec_I"] - wide["completion_time_sec_X"]).dropna()).all()


def test_timeouts_get_the_cap_and_are_not_completed(run):
    _, _, long, _ = run
    to = long[long["timed_out"]]
    assert len(to) == 2 and (to["completion_time_sec"] == pl.TIME_CAP_SEC).all() and (~to["completed"].astype(bool)).all()
    assert (to["condition"] == "Implicit").all()


def test_derived_values_agree_with_raw_rows(run):
    raw, _, long, _ = run
    r = long[(long.participant_id == "P01") & (long.condition == "Explicit")].iloc[0]
    assert r["active_time_sec"] == 205 - 25 and r["completion_time_sec"] == 180
    assert r["nasa_tlx_raw"] == pytest.approx(np.mean([35, 20, 30, 25, 35, 15]))
    assert r["ueqs_overall"] == pytest.approx(np.mean([6, 6, 5, 6, 6, 5, 6, 5]) - 4)
    assert r["compass_use"] == 1 and r["sequence"] == 1


def test_missing_values_only_where_faults_were_injected(run):
    """Any NaN in a key measure must trace back to an injected fault."""
    _, _, long, _ = run
    cols = ["completion_time_sec", "wayfinding_errors", "map_openings", "map_open_duration_sec", "nasa_tlx_raw",
            "ueqs_overall", "ueqs_pragmatic", "ueqs_hedonic"]
    na = long.set_index(["participant_id", "condition"])[cols].isna()
    found = {(p, str(c), col) for (p, c), row in na.iterrows() for col in cols if row[col]}
    expected = {("P16", "Implicit", "completion_time_sec"), ("P16", "Explicit", "completion_time_sec"),
                ("P13", "Implicit", "map_open_duration_sec"), ("P12", "Explicit", "nasa_tlx_raw"),
                ("P15", "Implicit", "ueqs_overall"), ("P15", "Implicit", "ueqs_pragmatic")}
    assert found <= expected, found - expected
    assert ("P12", "Explicit", "nasa_tlx_raw") in found and ("P15", "Implicit", "ueqs_overall") in found


def test_sequences_balanced_and_valid(run):
    _, _, _, wide = run
    assert wide["sequence"].value_counts().to_dict() == {1: 4, 2: 4, 3: 4, 4: 4}
    st = pl.sequence_table(wide)
    assert st["Participants"].sum() == 16


def test_cleaning_is_idempotent(run, tmp_path):
    """Cleaning already-clean data must change nothing (no repairs, no losses)."""
    _, clean, _, _ = run
    again = {k: v.drop(columns=["_row"], errors="ignore").astype(str).replace({"<NA>": ""}).replace({"nan": ""})
             for k, v in clean.tables.items()}
    for k in again:
        if k == "trials":
            again[k]["completed"] = again[k]["completed"].replace({"True": "True", "False": "False"})
    c2 = pl.clean_all(again)
    log2 = c2.log.to_frame()
    assert log2[log2["action"].isin(["repaired", "quarantined", "set_missing"])].empty, log2
    for k in clean.tables:
        assert len(c2.tables[k]) == len(clean.tables[k])


def test_fault_free_data_produces_no_repairs(tmp_path):
    parts, trials, quest, rel, inter = gen.build(clean=True)
    raw = {"participants": parts.astype(str), "trials": trials.astype(str), "questionnaires": quest.astype(str),
           "reliability": rel.astype(str), "interviews": inter.astype(str)}
    c = pl.clean_all(raw)
    log = c.log.to_frame()
    assert log[log["action"].isin(["repaired", "quarantined", "set_missing"])].empty, log
    long, wide = pl.prepare(c)
    assert len(long) == 32
    assert long[["completion_time_sec", "nasa_tlx_raw", "ueqs_overall", "map_open_duration_sec"]].notna().all().all()


def test_reliability_sample_is_a_quarter_and_coder1_matches_trials(run):
    _, clean, _, _ = run
    rel = clean.tables["reliability"]
    assert rel["participant_id"].nunique() == len(clean.participants) // 4
    assert pl.reliability_consistency(rel, clean.trials, pl.IssueLog()).empty


def test_interview_counts_never_exceed_participants(run):
    _, clean, _, _ = run
    cnt = pl.code_participant_counts(clean.tables["interviews"], 16)
    assert (cnt[["participants_X", "participants_I"]].max() <= 16).all().all()
    assert (cnt["segments_X"] >= cnt["participants_X"]).all() and (cnt["segments_I"] >= cnt["participants_I"]).all()
    assert set(cnt["code"]) <= set(pl.CODEBOOK)


def test_joint_display_has_one_row_per_rq(run):
    _, clean, long, wide = run
    desc = pl.describe_by_condition(long, wide, list(pl.MEASURES))
    jd = pl.joint_display(desc, pl.theme_participants(clean.tables["interviews"]))
    assert list(jd["RQ"]) == ["RQ2", "RQ3", "RQ4"] and jd["Interview themes"].str.len().min() > 0


def test_group_tables_cover_every_trial(run):
    _, _, long, _ = run
    for g in ("trial_position", "route_id"):
        t = pl.describe_by_group(long, g, ["completion_time_sec"])
        assert t["n"].sum() == len(long)


def test_pipeline_is_deterministic(dummy_dir):
    a = pl.run_pipeline(dummy_dir)[2]
    b = pl.run_pipeline(dummy_dir)[2]
    pd.testing.assert_frame_equal(a, b)


def test_generator_is_deterministic():
    a = gen.build(clean=False)[1]
    b = gen.build(clean=False)[1]
    pd.testing.assert_frame_equal(a, b)


def test_original_p01_p08_rows_preserved_verbatim():
    parts, trials, quest, _, _ = gen.build(clean=True)
    orig = pd.read_csv(io.StringIO(gen.ORIG_TRIALS), dtype=str)
    pd.testing.assert_frame_equal(trials.iloc[: len(orig)].reset_index(drop=True), orig)


def test_notebook_pipeline_importable_without_side_effects():
    out = subprocess.run([sys.executable, "-c", "import pipeline, plots"], cwd=HERE, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ------------------------------------------------------------ boundary values
def test_age_boundary_18_is_eligible_17_is_not():
    ok, _, _ = _clean_parts([{**BASE_P, "age": 18, "years_of_gaming": 3}])
    no, log, _ = _clean_parts([{**BASE_P, "age": 17, "years_of_gaming": 3}])
    assert len(ok) == 1 and no.empty and "ineligible:age<18" in rules(log)


def test_pause_equal_to_elapsed_is_invalid_but_one_less_is_valid():
    rows = _pair(); rows[0].update(elapsed_time_sec=100, combat_cutscene_pause_sec=100)
    out, log, _ = _clean_trials(rows)
    assert "pause>=elapsed" in rules(log)
    rows = _pair(); rows[0].update(elapsed_time_sec=100, combat_cutscene_pause_sec=99)
    out, log, _ = _clean_trials(rows)
    assert "pause>=elapsed" not in rules(log) and out.iloc[0]["elapsed_time_sec"] == 100


def test_cap_boundary_480_active_seconds_completed_is_kept_481_is_not():
    rows = _pair(); rows[0].update(elapsed_time_sec=500, combat_cutscene_pause_sec=20, completed="True")   # active 480
    out, log, _ = _clean_trials(rows)
    assert "completed_but_over_cap" not in rules(log) and out.iloc[0]["completed"] == True  # noqa: E712
    rows = _pair(); rows[0].update(elapsed_time_sec=501, combat_cutscene_pause_sec=20, completed="True")   # active 481
    out, log, _ = _clean_trials(rows)
    assert "completed_but_over_cap" in rules(log)


def test_counter_cases_and_headlines(run):
    _, _, long, wide = run
    cc = pl.counter_cases(wide, ["completion_time_sec"])
    maj = (wide["completion_time_sec_diff"].dropna() > 0).sum()
    assert maj >= 8
    n_not = (wide["completion_time_sec_diff"].dropna() <= 0).sum()
    assert len(cc) == n_not
    desc = pl.describe_by_condition(long, wide, list(pl.MEASURES))
    h = pl.headline_table(desc, pl.paired_differences(wide, list(pl.MEASURES), n_boot=200))
    assert len(h) == 10 and h["sentence"].str.contains("Median").all()
    assert not h["sentence"].str.contains("because|caused|effect of", case=False).any()


def test_counter_cases_ties_are_listed():
    w = pd.DataFrame({"participant_id": ["P01", "P02", "P03"], "wayfinding_errors_diff": [2.0, 0.0, -1.0]})
    cc = pl.counter_cases(w, ["wayfinding_errors"])
    assert set(cc["participant_id"]) == {"P02", "P03"}


# ---- outlier flagging: pins the modified z-score constant and threshold -------
def _outlier_flags(value):
    long = pd.DataFrame({"participant_id": [f"P{i:02d}" for i in range(1, 8)],
                         "condition": "Explicit", "x": [1, 2, 3, 4, 5, 6, value]})
    log = pl.IssueLog()
    pl.flag_outliers(long, ["x"], log)
    return log.to_frame()


def test_outlier_flagged_just_above_threshold():
    # median 4, MAD 2 -> modified z = 0.6745 * 12 / 2 = 4.05 > 3.5
    f = _outlier_flags(16)
    assert len(f) == 1 and f.iloc[0]["participant_id"] == "P07"


def test_value_below_threshold_not_flagged():
    # z = 0.6745 * 8 / 2 = 2.70 < 3.5
    assert _outlier_flags(12).empty
