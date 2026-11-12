"""mutation_check.py - breaks pipeline.py on purpose and confirms test_pipeline.py FAILS each time.
A surviving mutant = a behaviour the tests do not protect. Usage: python mutation_check.py"""
import shutil, subprocess, sys
from pathlib import Path

SRC = Path("pipeline.py")
MUTANTS = [
    ("cap 480 -> 500", "TIME_CAP_SEC = 480", "TIME_CAP_SEC = 500"),
    ("age boundary < 18 -> <= 18", 'df["age"] < 18', 'df["age"] <= 18'),
    ("pause >= elapsed -> >", '>= df["elapsed_time_sec"]', '> df["elapsed_time_sec"]'),
    ("UEQ-S offset 4 -> 3", 'astype(float) - 4.0', 'astype(float) - 3.0'),
    ("UEQ-S reversal removed", "sc[f\"i{i}\"] = -v if i in reversed_items else v", "sc[f\"i{i}\"] = v"),
    ("TLX skipna True", "df[TLX_COLS].mean(axis=1, skipna=False)", "df[TLX_COLS].mean(axis=1, skipna=True)"),
    ("difference sign flipped", 'wide[f"{m}_diff"] = wide[f"{m}_I"] - wide[f"{m}_X"]',
     'wide[f"{m}_diff"] = wide[f"{m}_X"] - wide[f"{m}_I"]'),
    ("Holm multiplier m-rank -> m", "(m - rank) * p[i]", "m * p[i]"),
    ("Holm monotonicity dropped", "running = max(running, (m - rank) * p[i])", "running = (m - rank) * p[i]"),
    ("MAD z constant changed", "0.6745 *", "0.5 *"),
    ("UEQ-S overall skipna True", 'sc.mean(axis=1, skipna=False)', 'sc.mean(axis=1, skipna=True)'),
    ("pragmatic skipna True", 'UEQS_PRAGMATIC_ITEMS]].mean(axis=1, skipna=False)', 'UEQS_PRAGMATIC_ITEMS]].mean(axis=1, skipna=True)'),
]


def main():
    orig = SRC.read_text()
    backup = Path("pipeline.py.bak"); shutil.copy(SRC, backup)
    survived = []
    try:
        for name, a, b in MUTANTS:
            if a not in orig:
                print(f"SKIP (pattern not found): {name}"); continue
            SRC.write_text(orig.replace(a, b, 1))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "test_pipeline.py"],
                               capture_output=True, text=True)
            killed = r.returncode != 0
            print(f"{'killed  ' if killed else 'SURVIVED'}  {name}")
            if not killed: survived.append(name)
    finally:
        SRC.write_text(orig); backup.unlink(missing_ok=True)
    print(f"\n{len(MUTANTS) - len(survived)}/{len(MUTANTS)} mutants killed")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
