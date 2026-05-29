# push_new_data.py
"""
Run this after playing sessions.
Checks new data exists, then pushes to GitHub to trigger CI/CD retraining.
"""

import subprocess
import pandas as pd
from pathlib import Path


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}")
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {result.stderr}")
    return result.stdout.strip()


def main() -> None:
    # 1 — Check sessions.csv exists
    csv_path = Path("data/sessions.csv")
    if not csv_path.exists():
        print("❌ data/sessions.csv not found — play some sessions first!")
        return

    # 2 — Show current data summary
    df = pd.read_csv(csv_path)
    print(f"── Session data summary ─────────────────────")
    print(f"   Total sessions : {len(df)}")
    print(f"   Difficulty range: {df['difficulty_level'].min():.2f} → {df['difficulty_level'].max():.2f}")
    print(f"   Score range     : {df['score'].min()} → {df['score'].max()}")
    print(f"\n── Last 3 sessions ──────────────────────────")
    print(df[["level","deaths","accuracy","score","difficulty_level"]].tail(3).to_string())
    print()

    # 3 — Check if there's anything new to push
    status = subprocess.run(
        "git status --porcelain data/sessions.csv",
        shell=True, capture_output=True, text=True
    ).stdout.strip()

    if not status:
        print("⚠️  sessions.csv hasn't changed since last push.")
        print("   Play more sessions first, then run this script again.")
        return

    print(f"✅ New session data detected — pushing to GitHub...")

    # 4 — Git add, commit, push
    run("git add data/sessions.csv")
    run(f'git commit -m "Add {len(df)} sessions — trigger retraining"')
    run("git push origin main")

    print("\n✅ Pushed to GitHub!")
    print("   → Watch retraining at:")
    print("     https://github.com/Chandan-int/DL-Game-Project/actions")


if __name__ == "__main__":
    main()