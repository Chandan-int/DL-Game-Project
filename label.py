"""
label.py
────────
Retroactively assigns a meaningful difficulty target (0.0–1.0) to every
session in data/sessions.csv, replacing the stub 0.5.

Why this is needed
──────────────────
The game's get_difficulty() returns 0.5 for everyone → difficulty_level=0.5
always → constant target the MLP cannot learn from.

Why NOT score/time as efficiency
─────────────────────────────────
Score increments every frame at 60 fps, so score ≈ completion_time × 60.
They are perfectly correlated — dividing one by the other gives a near-
constant (~60) for every session, carrying zero skill signal.

Labelling formula (3 independent signals)
──────────────────────────────────────────
  1. completion_time_sec  → longer survival = more skilled
  2. avg_reaction_time_ms → lower = faster reactions = more skilled
  3. deaths               → fewer deaths = more skilled (inverted)

Each is min-max normalised per-player (personal baseline, not global),
then combined:

  raw_skill = 0.5 * survival_norm
            + 0.3 * (1 - reaction_norm)   ← invert: lower ms = better
            + 0.2 * (1 - deaths_norm)     ← invert: fewer deaths = better

A rolling-5 average smooths session-to-session noise.
Final target is clamped to [0.15, 0.85].

Output
──────
Rewrites data/sessions.csv with updated difficulty_level column.

Usage:
    python label.py
    python label.py --csv path/to/sessions.csv
"""

import argparse
import csv
import os
from collections import defaultdict

DEFAULT_CSV = "data/sessions.csv"

W_SURVIVAL  = 0.5
W_REACTION  = 0.3
W_DEATHS    = 0.2
ROLL_WINDOW = 5
CLAMP_LO    = 0.15
CLAMP_HI    = 0.85


def _minmax_norm(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def _rolling_avg(vals: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(vals)):
        chunk = vals[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def label(csv_path: str = DEFAULT_CSV) -> None:
    if not os.path.isfile(csv_path):
        print(f"[label] CSV not found: {csv_path}")
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[label] CSV is empty.")
        return

    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)
    for pid in by_player:
        by_player[pid].sort(key=lambda r: r["timestamp"])

    print(f"\n{'player':<8} {'#':<4} {'time_s':>7} {'react_ms':>9} {'deaths':>7} "
          f"{'raw_skill':>10} {'roll5':>7} {'old':>6} {'new':>7}")
    print("─" * 70)

    updated: list[dict] = []

    for pid, sessions in by_player.items():
        survivals = [float(s["completion_time_sec"])  for s in sessions]
        reactions = [float(s["avg_reaction_time_ms"]) for s in sessions]
        deaths    = [float(s["deaths"])               for s in sessions]

        surv_norm  = _minmax_norm(survivals)
        react_norm = _minmax_norm(reactions)
        death_norm = _minmax_norm(deaths)

        raw_skills = [
            W_SURVIVAL * s + W_REACTION * (1 - r) + W_DEATHS * (1 - d)
            for s, r, d in zip(surv_norm, react_norm, death_norm)
        ]

        smoothed = _rolling_avg(raw_skills, ROLL_WINDOW)
        targets  = [round(_clamp(v, CLAMP_LO, CLAMP_HI), 4) for v in smoothed]

        for i, (s, rs, sm, tgt) in enumerate(zip(sessions, raw_skills, smoothed, targets)):
            old = s["difficulty_level"]
            print(f"{pid:<8} {i+1:<4} {float(s['completion_time_sec']):>7.1f} "
                  f"{float(s['avg_reaction_time_ms']):>9.1f} "
                  f"{int(float(s['deaths'])):>7} "
                  f"{rs:>10.4f} {sm:>7.4f} {old:>6} {tgt:>7}")
            s["difficulty_level"] = tgt
            updated.append(s)

    lookup = {(r["player_id"], r["timestamp"]): r["difficulty_level"] for r in updated}
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r["difficulty_level"] = lookup[(r["player_id"], r["timestamp"])]
            writer.writerow(r)

    targets_written = [r["difficulty_level"] for r in updated]
    print(f"\n[label] ✓ {len(rows)} rows re-labelled → {csv_path}")
    print(f"[label]   target range: [{min(targets_written)}, {max(targets_written)}]")
    print(f"[label]   signals used: survival_time (0.5) + reaction (0.3) + deaths (0.2)")
    print(f"[label]   Next → python ingest.py && python features.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    label(csv_path=args.csv)