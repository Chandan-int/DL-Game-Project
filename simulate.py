"""
simulate.py
───────────
Generates realistic fake game sessions and appends them to data/sessions.csv.
Simulates a player who gradually improves over time — the learning curve the
MLP needs to see in training data.

What it simulates
─────────────────
  • Multiple player profiles (beginner / average / expert)
  • Survival time grows as sessions accumulate (player learns)
  • Reaction time slowly decreases (player gets faster)
  • Gaussian noise on every metric (real humans aren't consistent)
  • Deaths stay at 3 (matches current deterministic stub behaviour)
  • Score derived from survival time × 60fps (matches your game loop)

Usage:
    python simulate.py                        # 3 players × 30 sessions = 90 rows
    python simulate.py --players 5 --sessions 40
    python simulate.py --players 1 --sessions 20 --player-id p_001
                                              # add sessions for a specific player
"""

import argparse
import csv
import math
import os
import random
from datetime import datetime, timedelta

DEFAULT_CSV     = "data/sessions.csv"
DEFAULT_PLAYERS = 3
DEFAULT_SESSIONS = 30

CSV_FIELDS = [
    "player_id", "level", "deaths", "accuracy",
    "avg_reaction_time_ms", "completion_time_sec",
    "score", "difficulty_level", "timestamp",
]

# ── Player archetypes ─────────────────────────────────────────────────────────
# Each archetype defines starting skill and improvement rate.
# survival_start/end: seconds alive at session 1 vs session N
# reaction_start/end: ms reaction time at session 1 vs session N
# noise_*: std-dev of Gaussian noise applied each session

ARCHETYPES = {
    "beginner": dict(
        survival_start=8.0,   survival_end=40.0,
        reaction_start=1800,  reaction_end=1500,
        noise_survival=4.0,   noise_reaction=60.0,
    ),
    "average": dict(
        survival_start=20.0,  survival_end=70.0,
        reaction_start=1600,  reaction_end=1300,
        noise_survival=6.0,   noise_reaction=50.0,
    ),
    "expert": dict(
        survival_start=50.0,  survival_end=110.0,
        reaction_start=1300,  reaction_end=950,
        noise_survival=8.0,   noise_reaction=40.0,
    ),
}


def _lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation. t in [0, 1]."""
    return start + (end - start) * t


def _progress(session_idx: int, total: int) -> float:
    """
    Non-linear progress curve — improvement is faster early, plateaus later.
    Uses sqrt so early sessions show big gains, later ones are marginal.
    """
    return math.sqrt(session_idx / max(total - 1, 1))


def simulate_player(
    player_id: str,
    archetype: str,
    n_sessions: int,
    start_time: datetime,
) -> list[dict]:
    """Generate n_sessions rows for one player."""
    cfg = ARCHETYPES[archetype]
    rows = []
    ts = start_time

    for i in range(n_sessions):
        t = _progress(i, n_sessions)

        # survival time with noise (minimum 5s)
        survival = max(5.0, _lerp(cfg["survival_start"], cfg["survival_end"], t)
                       + random.gauss(0, cfg["noise_survival"]))

        # reaction time with noise (minimum 600ms)
        reaction = max(600.0, _lerp(cfg["reaction_start"], cfg["reaction_end"], t)
                       + random.gauss(0, cfg["noise_reaction"]))

        # score = frames survived × 60fps (matches game loop exactly)
        score = int(survival * 60)

        # timestamp: space sessions ~5–20 min apart with some randomness
        gap_minutes = random.uniform(5, 20)
        ts += timedelta(minutes=gap_minutes)

        rows.append({
            "player_id":            player_id,
            "level":                1,
            "deaths":               3,          # deterministic with stub
            "accuracy":             0.0,        # no shooting mechanic yet
            "avg_reaction_time_ms": round(reaction, 1),
            "completion_time_sec":  round(survival, 2),
            "score":                score,
            "difficulty_level":     0.5,        # stub value; label.py will replace
            "timestamp":            ts.isoformat(),
        })

    return rows


def generate(
    csv_path: str = DEFAULT_CSV,
    n_players: int = DEFAULT_PLAYERS,
    n_sessions: int = DEFAULT_SESSIONS,
    player_id: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    all_rows: list[dict] = []

    if player_id:
        # single player mode — always beginner archetype so growth is visible
        archetype = random.choice(list(ARCHETYPES.keys()))
        print(f"[simulate] player {player_id} ({archetype}) × {n_sessions} sessions")
        rows = simulate_player(player_id, archetype, n_sessions,
                               start_time=datetime(2026, 5, 26, 9, 0, 0))
        all_rows.extend(rows)
    else:
        archetypes = list(ARCHETYPES.keys())
        start = datetime(2026, 5, 26, 8, 0, 0)
        for i in range(n_players):
            pid       = f"p_{100 + i:03d}"
            archetype = archetypes[i % len(archetypes)]
            print(f"[simulate] player {pid} ({archetype}) × {n_sessions} sessions")
            rows = simulate_player(pid, archetype, n_sessions,
                                   start_time=start + timedelta(hours=i * 2))
            all_rows.extend(rows)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[simulate] ✓ {len(all_rows)} rows appended → {csv_path}")
    print(f"[simulate]   survival range: "
          f"[{min(r['completion_time_sec'] for r in all_rows):.1f}s, "
          f"{max(r['completion_time_sec'] for r in all_rows):.1f}s]")
    print(f"[simulate]   reaction range: "
          f"[{min(r['avg_reaction_time_ms'] for r in all_rows):.0f}ms, "
          f"{max(r['avg_reaction_time_ms'] for r in all_rows):.0f}ms]")
    print(f"\n[simulate]   Next →")
    print(f"              python label.py")
    print(f"              python ingest.py")
    print(f"              python features.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake game sessions for training.")
    parser.add_argument("--csv",       default=DEFAULT_CSV)
    parser.add_argument("--players",   type=int, default=DEFAULT_PLAYERS,
                        help="Number of players to simulate (ignored if --player-id set)")
    parser.add_argument("--sessions",  type=int, default=DEFAULT_SESSIONS,
                        help="Sessions per player")
    parser.add_argument("--player-id", default=None,
                        help="Simulate sessions for a specific existing player ID")
    args = parser.parse_args()

    generate(
        csv_path=args.csv,
        n_players=args.players,
        n_sessions=args.sessions,
        player_id=args.player_id,
    )