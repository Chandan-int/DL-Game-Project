# retrain_trigger.py
"""
Checks if enough new sessions exist to justify retraining.
Run this manually or via a scheduled task after playing.
"""

import csv
import subprocess
from pathlib import Path

MIN_SESSIONS_TO_RETRAIN = 20   # retrain after every 20 new sessions
SESSIONS_CSV            = Path("data/sessions.csv")
COUNTER_FILE            = Path("data/.last_retrain_count")


def count_sessions() -> int:
    if not SESSIONS_CSV.exists():
        return 0
    with open(SESSIONS_CSV) as f:
        return sum(1 for row in csv.DictReader(f))


def last_retrain_count() -> int:
    if not COUNTER_FILE.exists():
        return 0
    return int(COUNTER_FILE.read_text().strip())


def save_retrain_count(n: int) -> None:
    COUNTER_FILE.write_text(str(n))


def maybe_retrain() -> None:
    total    = count_sessions()
    last     = last_retrain_count()
    new_rows = total - last

    print(f"[retrain] Total sessions : {total}")
    print(f"[retrain] Since last run : {new_rows}")
    print(f"[retrain] Threshold      : {MIN_SESSIONS_TO_RETRAIN}")

    if new_rows < MIN_SESSIONS_TO_RETRAIN:
        remaining = MIN_SESSIONS_TO_RETRAIN - new_rows
        print(f"[retrain] ⏳ Need {remaining} more sessions before retraining.")
        return

    print(f"[retrain] 🚀 Triggering retrain...")
    result = subprocess.run(["python", "train.py"], capture_output=False)

    if result.returncode == 0:
        save_retrain_count(total)
        print(f"[retrain] ✅ Retrain complete — model updated in MLflow")
    else:
        print(f"[retrain] ❌ Retrain failed — check train.py output")


if __name__ == "__main__":
    maybe_retrain()