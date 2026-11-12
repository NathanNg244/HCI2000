"""
generate_dummy_data.py - builds the DUMMY datasets used until data collection ends.

* P01-P08 are the original dummy rows, kept verbatim.
* P09-P16 are generated (seeded) so the dummy study reaches the planned n = 16
  with the four counterbalanced sequences balanced (4 participants each).
* A few realistic data-entry faults are injected on purpose (INJECTED_FAULTS) so
  that the cleaning pipeline has something real to detect and the unit tests can
  verify that every fault is caught. Run with --clean for fault-free files.

All numbers are invented. Do not interpret them.
Usage:  python generate_dummy_data.py [--clean] [--out DIR]
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261112

ORIG_PARTICIPANTS = """participant_id,age,gender,weekly_gaming_hours,years_of_gaming,third_person_3d_game_past_12_months,previous_assassins_creed_play,input_device,open_world_game_experience
P01,21,Woman,8,4,Yes,No,Keyboard and mouse,Yes
P02,22,Man,12,6,Yes,Yes,Gamepad,Yes
P03,20,Woman,5,3,Yes,No,Keyboard and mouse,Yes
P04,23,Man,10,7,Yes,No,Gamepad,Yes
P05,21,Man,6,4,Yes,Yes,Keyboard and mouse,Yes
P06,24,Woman,14,8,Yes,No,Gamepad,Yes
P07,22,Woman,7,5,Yes,No,Keyboard and mouse,Yes
P08,20,Man,9,2,Yes,Yes,Gamepad,Yes
"""
ORIG_TRIALS = """participant_id,trial_number,condition,route_id,trial_position,elapsed_time_sec,combat_cutscene_pause_sec,completed,wayfinding_errors,lost_episodes,map_openings,total_map_open_duration_sec,objective_board_openings,compass_use_notes
P01,1,Explicit,Route_A,1,205,25,True,1,0,2,38,1,Used compass briefly
P01,2,Implicit,Route_B,2,292,32,True,3,2,5,104,4,Checked compass several times
P02,1,Implicit,Route_A,1,310,40,True,2,1,5,115,3,Relied on compass
P02,2,Explicit,Route_B,2,225,20,True,0,0,2,31,1,Used compass briefly
P03,1,Explicit,Route_B,1,188,18,True,0,0,1,19,1,Followed compass
P03,2,Implicit,Route_A,2,275,25,True,2,1,4,92,3,Checked compass several times
P04,1,Implicit,Route_B,1,325,35,True,3,2,6,132,4,Relied on compass
P04,2,Explicit,Route_A,2,215,25,True,1,0,2,34,1,Followed compass
P05,1,Explicit,Route_A,1,230,30,True,1,0,2,42,2,Used compass briefly
P05,2,Implicit,Route_B,2,340,40,True,3,2,6,141,4,Frequently checked compass
P06,1,Implicit,Route_A,1,510,30,False,4,3,8,190,6,Relied on compass; still unsure
P06,2,Explicit,Route_B,2,250,30,True,1,0,3,48,2,Followed compass
P07,1,Explicit,Route_B,1,195,15,True,0,0,1,22,1,Followed compass
P07,2,Implicit,Route_A,2,285,25,True,2,1,4,98,3,Checked compass several times
P08,1,Implicit,Route_B,1,300,30,True,2,1,5,108,3,Relied on compass
P08,2,Explicit,Route_A,2,220,20,True,1,0,2,35,1,Used compass briefly
"""
ORIG_QUEST = """participant_id,condition,mental_demand,physical_demand,temporal_demand,performance,effort,frustration,ueqs_item1_raw_1_to_7,ueqs_item2_raw_1_to_7,ueqs_item3_raw_1_to_7,ueqs_item4_raw_1_to_7,ueqs_item5_raw_1_to_7,ueqs_item6_raw_1_to_7,ueqs_item7_raw_1_to_7,ueqs_item8_raw_1_to_7
P01,Explicit,35,20,30,25,35,15,6,6,5,6,6,5,6,5
P01,Implicit,65,30,60,55,70,50,4,4,3,4,4,3,4,3
P02,Implicit,55,25,50,45,55,35,4,5,4,4,5,4,5,4
P02,Explicit,30,20,25,30,30,15,6,6,5,6,6,5,6,6
P03,Explicit,25,15,20,25,25,10,6,6,6,5,6,6,5,6
P03,Implicit,50,25,45,40,50,30,4,4,4,3,4,4,3,4
P04,Implicit,70,35,65,60,75,55,3,3,4,3,3,4,3,3
P04,Explicit,35,20,30,30,35,15,6,5,6,6,5,6,6,5
P05,Explicit,40,25,35,30,40,20,5,5,5,4,5,5,4,5
P05,Implicit,75,40,70,65,80,65,3,3,3,2,3,3,2,3
P06,Implicit,80,45,85,75,90,80,2,2,2,2,2,2,2,2
P06,Explicit,45,25,40,35,45,25,5,5,4,5,5,4,5,5
P07,Explicit,25,15,20,20,25,10,6,6,6,6,6,5,6,6
P07,Implicit,60,30,55,50,65,45,4,4,4,3,4,4,3,4
P08,Implicit,55,30,50,45,60,40,4,4,5,4,4,5,4,4
P08,Explicit,35,20,30,25,35,15,6,6,5,6,6,5,6,6
"""

# (table, participant, rule that must appear in the issue log, description)
INJECTED_FAULTS = [
    ("trials", "P09", "participant_id:format", "ID typed as 'p9'"),
    ("questionnaires", "P10", "condition:category", "condition typed ' explicit ' (space, case)"),
    ("trials", "P11", "exact_duplicate_row", "trial row entered twice"),
    ("questionnaires", "P12", "effort:out_of_range", "NASA-TLX effort typed 105"),
    ("trials", "P13", "na_token", "map duration entered as 'n/a'"),
    ("trials", "P14", "completed:boolean", "'No' / 'TRUE' spellings in completed"),
    ("questionnaires", "P15", "na_token", "UEQ-S item 3 left as '-'"),
    ("trials", "P16", "pause>=elapsed", "pause longer than the whole trial"),
    ("questionnaires", "P17", "unknown_or_excluded_participant", "questionnaire for an unregistered ID"),
    ("participants", "P16", "gender:category", "gender typed 'woman' in lowercase"),
]

CODES_SOURCE = ["marker", "compass", "map", "clue_text", "landmarks_terrain", "lighting_colour", "character_clues"]
P_SOURCE = {"Explicit": [0.95, 0.55, 0.40, 0.15, 0.25, 0.10, 0.10],
            "Implicit": [0.45, 0.85, 0.90, 0.95, 0.85, 0.50, 0.60]}
EXCERPTS = {
    "marker": "[dummy] followed the marker", "compass": "[dummy] glanced at the compass",
    "map": "[dummy] opened the map to check", "clue_text": "[dummy] re-read the clue text",
    "landmarks_terrain": "[dummy] used a landmark to decide", "lighting_colour": "[dummy] noticed lighting",
    "character_clues": "[dummy] relied on what the character said", "ambiguous_clue": "[dummy] the clue was unclear",
    "missed_branch": "[dummy] took the wrong branch", "lost_orientation": "[dummy] was not sure where to go",
    "map_hard_to_relate": "[dummy] map did not match the view", "more_confident": "[dummy] felt sure of the way",
    "more_effortful": "[dummy] had to think harder", "more_immersive": "[dummy] felt more immersed",
    "frustrating": "[dummy] got frustrated",
}


def _round5(x):
    return int(np.clip(5 * round(x / 5), 0, 100))


def build(clean: bool = False):
    rng = np.random.default_rng(SEED)
    parts = pd.read_csv(StringIO(ORIG_PARTICIPANTS), dtype=str)
    trials = pd.read_csv(StringIO(ORIG_TRIALS), dtype=str)
    quest = pd.read_csv(StringIO(ORIG_QUEST), dtype=str)

    # ---- P09-P16: sequences balanced (each sequence twice more) -------------
    seq_first = {1: ("Explicit", "Route_A"), 2: ("Implicit", "Route_B"),
                 3: ("Implicit", "Route_A"), 4: ("Explicit", "Route_B")}
    seqs = rng.permutation([1, 2, 3, 4, 1, 2, 3, 4])
    new_p, new_t, new_q = [], [], []
    compass_x = ["Used compass briefly", "Followed compass", "Followed compass"]
    compass_i = ["Checked compass several times", "Relied on compass", "Frequently checked compass"]
    for k, s in enumerate(seqs):
        pid = f"P{9 + k:02d}"
        age = int(rng.integers(19, 26))
        new_p.append(dict(participant_id=pid, age=age,
                          gender=rng.choice(["Woman", "Man", "Man", "Woman", "Non-binary"]),
                          weekly_gaming_hours=int(rng.integers(3, 16)),
                          years_of_gaming=int(rng.integers(2, min(10, age - 12))),
                          third_person_3d_game_past_12_months="Yes",
                          previous_assassins_creed_play=rng.choice(["Yes", "No", "No"]),
                          input_device=rng.choice(["Keyboard and mouse", "Gamepad"]),
                          open_world_game_experience=rng.choice(["Yes", "Yes", "Yes", "No"])))
        skill = rng.normal(0, 18)                      # participant-level speed offset
        c1, r1 = seq_first[s]
        order = [(c1, r1), ("Implicit" if c1 == "Explicit" else "Explicit",
                            "Route_B" if r1 == "Route_A" else "Route_A")]
        for pos, (cond, route) in enumerate(order, start=1):
            x = cond == "Explicit"
            active = (rng.normal(200, 22) if x else rng.normal(305, 38)) + skill
            active *= 0.96 if pos == 2 else 1.0        # small learning effect
            active += 6 if route == "Route_B" else 0
            pause = int(rng.integers(15, 41))
            err = int(np.clip(rng.poisson(0.7 if x else 2.2), 0, 4))
            lost = int(np.clip(round(err * (0.0 if x else 0.65) + rng.normal(0, 0.4)), 0, 3))
            maps = int(np.clip(round(rng.normal(2 if x else 5, 1.0) + 0.4 * err), 1, 9))
            mdur = int(maps * rng.uniform(14, 22) + (0 if x else 15))
            board = int(np.clip(round(rng.normal(1.3 if x else 3.4, 0.7)), 1, 6))
            done = True
            if pid == "P14" and cond == "Implicit":    # second timeout in the dummy set
                active, err, lost, maps, mdur, board, done = 480, 4, 3, 8, 185, 6, False
            note = rng.choice(compass_x if x else compass_i)
            new_t.append(dict(participant_id=pid, trial_number=pos, condition=cond, route_id=route,
                              trial_position=pos, elapsed_time_sec=int(round(active)) + pause,
                              combat_cutscene_pause_sec=pause, completed=done, wayfinding_errors=err,
                              lost_episodes=lost, map_openings=maps,
                              total_map_open_duration_sec=mdur, objective_board_openings=board,
                              compass_use_notes=note))
            lat = 0.0 if x else 1.0
            tlx = [rng.normal(33 + 33 * lat, 8), rng.normal(20 + 12 * lat, 5), rng.normal(28 + 30 * lat, 8),
                   rng.normal(28 + 22 * lat, 7), rng.normal(33 + 35 * lat, 8), rng.normal(15 + 30 * lat, 8)]
            ue = [int(np.clip(round(rng.normal(5.6 if x else 3.5, 0.7)), 1, 7)) for _ in range(8)]
            row = dict(participant_id=pid, condition=cond)
            for n, v in zip(["mental_demand", "physical_demand", "temporal_demand", "performance",
                             "effort", "frustration"], tlx):
                row[n] = _round5(v)
            for i, v in enumerate(ue, start=1):
                row[f"ueqs_item{i}_raw_1_to_7"] = v
            new_q.append(row)

    parts = pd.concat([parts, pd.DataFrame(new_p).astype(str)], ignore_index=True)
    trials = pd.concat([trials, pd.DataFrame(new_t).astype(str)], ignore_index=True)
    quest = pd.concat([quest, pd.DataFrame(new_q).astype(str)], ignore_index=True)

    # ---- second coder: random quarter of participants (n/4 = 4) -------------
    pids = list(parts["participant_id"])
    sample = sorted(rng.choice(pids, size=len(pids) // 4, replace=False))
    rel_rows = []
    for pid in sample:
        for _, t in trials[trials["participant_id"] == pid].iterrows():
            e, d = int(t["wayfinding_errors"]), 4
            hit = set(rng.choice(d, size=e, replace=False)) if e else set()
            for dp in range(1, d + 1):
                c1 = int((dp - 1) in hit)
                c2 = 1 - c1 if rng.random() < 0.06 else c1
                rel_rows.append((pid, t["condition"], "error_code", dp, c1, c2))
            for m in ["map_openings", "objective_board_openings", "lost_episodes"]:
                c1 = int(t[m])
                c2 = c1 + int(rng.choice([0, 0, 0, 1, -1])) if c1 > 0 else c1
                rel_rows.append((pid, t["condition"], m, "trial", c1, max(c2, 0)))
    rel = pd.DataFrame(rel_rows, columns=["participant_id", "condition", "measure", "item", "coder1", "coder2"])

    # ---- coded interviews (reconciled codes; excerpts are placeholders) -----
    int_rows = []
    for pid in pids:
        for cond in ("Explicit", "Implicit"):
            for code, p in zip(CODES_SOURCE, P_SOURCE[cond]):
                if rng.random() < p:
                    for _ in range(int(rng.integers(1, 3))):
                        int_rows.append((pid, cond, "trial", "source", code, EXCERPTS[code]))
            t = trials[(trials.participant_id == pid) & (trials.condition == cond)].iloc[0]
            if int(t["lost_episodes"]) > 0:
                for code in rng.choice(["ambiguous_clue", "missed_branch", "lost_orientation", "map_hard_to_relate"],
                                       size=min(2, int(t["lost_episodes"]) + 1), replace=False):
                    int_rows.append((pid, cond, "trial", "difficulty", code, EXCERPTS[code]))
            xs = cond == "Explicit"
            for code, p in [("more_confident", 0.8 if xs else 0.25), ("more_effortful", 0.2 if xs else 0.8),
                            ("more_immersive", 0.35 if xs else 0.7), ("frustrating", 0.1 if xs else 0.45)]:
                if rng.random() < p:
                    int_rows.append((pid, cond, "final", "experience", code, EXCERPTS[code]))
    inter = pd.DataFrame(int_rows, columns=["participant_id", "condition", "interview_stage",
                                            "code_family", "code", "excerpt"])
    inter.insert(0, "segment_id", [f"S{i + 1:03d}" for i in range(len(inter))])

    if not clean:
        _inject_faults(parts, trials, quest)
        quest = pd.concat([quest, pd.DataFrame([{**quest[quest.participant_id == "P01"].iloc[0].to_dict(),
                                                 "participant_id": "P17"}])], ignore_index=True)
        trials = _dup_row(trials, "P11")
    return parts, trials, quest, rel, inter


def _dup_row(trials, pid):
    i = trials.index[(trials.participant_id == pid) & (trials.trial_position == "1")][0]
    return pd.concat([trials.iloc[: i + 1], trials.iloc[[i]], trials.iloc[i + 1:]], ignore_index=True)


def _inject_faults(parts, trials, quest):
    trials.loc[(trials.participant_id == "P09") & (trials.trial_position == "1"), "participant_id"] = "p9"
    quest.loc[(quest.participant_id == "P10") & (quest.condition == "Explicit"), "condition"] = " explicit "
    quest.loc[(quest.participant_id == "P12") & (quest.condition == "Explicit"), "effort"] = "105"
    trials.loc[(trials.participant_id == "P13") & (trials.condition == "Implicit"),
               "total_map_open_duration_sec"] = "n/a"
    m = trials.participant_id == "P14"
    trials.loc[m & (trials.condition == "Implicit"), "completed"] = "No"
    trials.loc[m & (trials.condition == "Explicit"), "completed"] = "TRUE"
    quest.loc[(quest.participant_id == "P15") & (quest.condition == "Implicit"), "ueqs_item3_raw_1_to_7"] = "-"
    i = trials.index[(trials.participant_id == "P16") & (trials.trial_position == "2")][0]
    trials.loc[i, "combat_cutscene_pause_sec"] = str(int(trials.loc[i, "elapsed_time_sec"]) + 20)
    parts.loc[parts.participant_id == "P16", "gender"] = "woman"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="write fault-free files")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts, trials, quest, rel, inter = build(a.clean)
    parts.to_csv(out / "participants_dummy.csv", index=False)
    trials.to_csv(out / "trials_dummy.csv", index=False)
    quest.to_csv(out / "questionnaires_dummy.csv", index=False)
    rel.to_csv(out / "reliability_dummy.csv", index=False)
    inter.to_csv(out / "interviews_coded_dummy.csv", index=False)
    print(f"Wrote 5 dummy CSVs to {out.resolve()} ({'clean' if a.clean else 'with injected faults'}).")
    if not a.clean:
        for t, p, r, d in INJECTED_FAULTS:
            print(f"  injected: {t:<15} {p}  {d}")


if __name__ == "__main__":
    main()
