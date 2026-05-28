"""
features.py
───────────
Reads the SQLite DB produced by ingest.py and outputs a feature-engineered
CSV (data/features.csv) ready for the PyTorch trainer.

Dead columns removed vs v1
──────────────────────────
  ✗ level_* one-hot   — only level 1 exists in current data; all zeros except level_1
  ✗ deaths_norm       — deaths=3 always (deterministic with stub difficulty)
  ✗ accuracy          — accuracy=0.0 always (no shooting mechanic yet)

Live features (all that actually vary)
───────────────────────────────────────
  survival_norm           min-max normalised completion_time_sec  (longer = better)
  reaction_norm           min-max normalised avg_reaction_time_ms (lower = better)
  score_norm              min-max normalised score
  roll5_survival_norm     rolling-5 avg of survival_norm
  roll5_reaction_norm     rolling-5 avg of reaction_norm
  roll5_score_norm        rolling-5 avg of score_norm
  sessions_log            log1p(total_sessions) — experience curve
  target                  difficulty_level (regression target, 0–1)

Normalisation: min-max per column, clipped at p99 for reaction time.

Usage:
    python features.py
    python features.py --db data/game.db --out data/features.csv
"""

import argparse
import csv
import math
import os
import sqlite3

DEFAULT_DB  = "data/game.db"
DEFAULT_OUT = "data/features.csv"


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _minmax(val: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return round(max(0.0, min(1.0, (val - lo) / (hi - lo))), 5)


def build_features(db_path: str = DEFAULT_DB, out_path: str = DEFAULT_OUT) -> None:
    if not os.path.isfile(db_path):
        print(f"[features] DB not found: {db_path}. Run ingest.py first.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            s.player_id,
            s.completion_time_sec,
            s.avg_reaction_time_ms,
            s.score,
            s.difficulty_level,
            p.total_sessions,
            p.roll5_avg_reaction_ms,
            p.roll5_avg_score
        FROM sessions s
        JOIN players  p ON s.player_id = p.player_id
        ORDER BY s.id
    """).fetchall()

    # also pull roll5 of completion_time — add it to players if missing
    # (ingest.py v1 didn't track this; we compute it here from sessions directly)
    roll5_survival = {}
    for pid in {r["player_id"] for r in rows}:
        times = conn.execute("""
            SELECT completion_time_sec FROM sessions
            WHERE player_id = ? ORDER BY timestamp DESC LIMIT 5
        """, (pid,)).fetchall()
        avg = sum(t[0] for t in times) / len(times) if times else 0.0
        roll5_survival[pid] = avg

    conn.close()

    if not rows:
        print("[features] No data. Run ingest.py first.")
        return

    print(f"[features] {len(rows)} sessions loaded")

    # ── Normalisation bounds (global across all sessions) ──────────────────
    survival_sorted = sorted(float(r["completion_time_sec"])  for r in rows)
    reaction_sorted = sorted(float(r["avg_reaction_time_ms"]) for r in rows)
    score_sorted    = sorted(float(r["score"])                for r in rows)

    surv_lo,  surv_hi  = survival_sorted[0], survival_sorted[-1]
    react_lo, react_hi = reaction_sorted[0], _percentile(reaction_sorted, 99)
    score_lo, score_hi = score_sorted[0],    score_sorted[-1]

    # rolling-5 bounds
    rs_roll5 = sorted(roll5_survival.values())
    rr_roll5 = sorted(float(r["roll5_avg_reaction_ms"] or 0) for r in rows)
    rsc_roll5 = sorted(float(r["roll5_avg_score"] or 0)      for r in rows)

    print(f"[features] bounds → survival [{surv_lo:.1f}, {surv_hi:.1f}]s  "
          f"reaction [{react_lo:.0f}, {react_hi:.0f}]ms  "
          f"score [{score_lo:.0f}, {score_hi:.0f}]")

    # ── Build rows ─────────────────────────────────────────────────────────
    feature_rows = []
    for r in rows:
        pid = r["player_id"]
        feature_rows.append({
            "player_id":          pid,
            # per-session
            "survival_norm":      _minmax(float(r["completion_time_sec"]),  surv_lo,  surv_hi),
            "reaction_norm":      _minmax(float(r["avg_reaction_time_ms"]), react_lo, react_hi),
            "score_norm":         _minmax(float(r["score"]),                score_lo, score_hi),
            # rolling-5 player averages
            "roll5_survival_norm": _minmax(roll5_survival[pid],
                                           min(rs_roll5), max(rs_roll5)),
            "roll5_reaction_norm": _minmax(float(r["roll5_avg_reaction_ms"] or 0),
                                           min(rr_roll5) if rr_roll5 else 0,
                                           max(rr_roll5) if rr_roll5 else 1),
            "roll5_score_norm":    _minmax(float(r["roll5_avg_score"] or 0),
                                           min(rsc_roll5) if rsc_roll5 else 0,
                                           max(rsc_roll5) if rsc_roll5 else 1),
            # experience
            "sessions_log":       round(math.log1p(int(r["total_sessions"])), 5),
            # target
            "target":             round(float(r["difficulty_level"]), 5),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(feature_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"[features] ✓ {len(feature_rows)} rows → {out_path}")
    print(f"[features]   columns ({len(fieldnames)}): {fieldnames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",  default=DEFAULT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    build_features(db_path=args.db, out_path=args.out)