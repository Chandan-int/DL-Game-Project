# debug_data.py
"""Show the actual range of values in your training data."""

import sqlite3
import pandas as pd

# Try SQLite first, fall back to CSV
try:
    conn = sqlite3.connect("data/game_telemetry.db")
    df = pd.read_sql("SELECT * FROM sessions", conn)
    conn.close()
    print("Loaded from SQLite")
except Exception:
    import glob
    files = glob.glob("data/*.csv")
    df = pd.concat([pd.read_csv(f) for f in files])
    print(f"Loaded from CSV: {files}")

FEATURE_NAMES = [
    "level", "deaths", "accuracy",
    "avg_reaction_time_ms", "completion_time_sec",
    "score", "difficulty_level",
]

print("\n── Training data ranges ─────────────────────────────────")
print(f"{'Feature':<25} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print("─" * 62)
for col in FEATURE_NAMES:
    if col in df.columns:
        print(f"{col:<25} {df[col].min():>8.2f} {df[col].max():>8.2f} "
              f"{df[col].mean():>8.2f} {df[col].std():>8.2f}")