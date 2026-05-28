"""
ingest.py
─────────
Reads data/sessions.csv and upserts every row into SQLite.

Tables created/updated:
  • sessions  — one row per game session (mirrors CSV schema)
  • players   — one row per player, with rolling-5 aggregates recomputed on each run

Usage:
    python ingest.py                  # default: data/sessions.csv → data/game.db
    python ingest.py --csv path/to/sessions.csv --db path/to/game.db
"""

import argparse
import csv
import os
import sqlite3
from datetime import datetime

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CSV = "data/sessions.csv"
DEFAULT_DB  = "data/game.db"

# ── DDL ───────────────────────────────────────────────────────────────────────
DDL_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id            TEXT    NOT NULL,
    level                INTEGER NOT NULL,
    deaths               INTEGER NOT NULL,
    accuracy             REAL    NOT NULL,
    avg_reaction_time_ms REAL    NOT NULL,
    completion_time_sec  REAL    NOT NULL,
    score                INTEGER NOT NULL,
    difficulty_level     REAL    NOT NULL,
    timestamp            TEXT    NOT NULL,
    UNIQUE(player_id, timestamp)          -- idempotent upsert key
);
"""

DDL_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    player_id              TEXT PRIMARY KEY,
    total_sessions         INTEGER NOT NULL DEFAULT 0,
    -- rolling-5 averages (last 5 sessions)
    roll5_avg_deaths       REAL,
    roll5_avg_accuracy     REAL,
    roll5_avg_reaction_ms  REAL,
    roll5_avg_score        REAL,
    roll5_avg_difficulty   REAL,
    -- all-time averages (handy reference)
    all_avg_deaths         REAL,
    all_avg_accuracy       REAL,
    all_avg_reaction_ms    REAL,
    all_avg_score          REAL,
    last_seen              TEXT
);
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(DDL_SESSIONS)
    conn.execute(DDL_PLAYERS)
    conn.commit()


def _upsert_sessions(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert rows, skipping duplicates (player_id + timestamp). Returns count inserted."""
    sql = """
        INSERT OR IGNORE INTO sessions
            (player_id, level, deaths, accuracy, avg_reaction_time_ms,
             completion_time_sec, score, difficulty_level, timestamp)
        VALUES
            (:player_id, :level, :deaths, :accuracy, :avg_reaction_time_ms,
             :completion_time_sec, :score, :difficulty_level, :timestamp)
    """
    # cast types from CSV strings
    typed = []
    for r in rows:
        typed.append({
            "player_id":            r["player_id"],
            "level":                int(r["level"]),
            "deaths":               int(r["deaths"]),
            "accuracy":             float(r["accuracy"]),
            "avg_reaction_time_ms": float(r["avg_reaction_time_ms"]),
            "completion_time_sec":  float(r["completion_time_sec"]),
            "score":                int(r["score"]),
            "difficulty_level":     float(r["difficulty_level"]),
            "timestamp":            r["timestamp"],
        })
    cur = conn.executemany(sql, typed)
    conn.commit()
    return cur.rowcount


def _refresh_players(conn: sqlite3.Connection, player_ids: list[str]) -> None:
    """Recompute player aggregates for every affected player_id."""
    for pid in player_ids:
        # all sessions for this player, newest first
        rows = conn.execute("""
            SELECT deaths, accuracy, avg_reaction_time_ms, score, difficulty_level, timestamp
            FROM   sessions
            WHERE  player_id = ?
            ORDER  BY timestamp DESC
        """, (pid,)).fetchall()

        if not rows:
            continue

        total = len(rows)
        last5 = rows[:5]   # most recent 5

        def avg(col, src):
            vals = [r[col] for r in src]
            return round(sum(vals) / len(vals), 4) if vals else None

        conn.execute("""
            INSERT INTO players
                (player_id, total_sessions,
                 roll5_avg_deaths, roll5_avg_accuracy, roll5_avg_reaction_ms,
                 roll5_avg_score, roll5_avg_difficulty,
                 all_avg_deaths, all_avg_accuracy, all_avg_reaction_ms, all_avg_score,
                 last_seen)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(player_id) DO UPDATE SET
                total_sessions        = excluded.total_sessions,
                roll5_avg_deaths      = excluded.roll5_avg_deaths,
                roll5_avg_accuracy    = excluded.roll5_avg_accuracy,
                roll5_avg_reaction_ms = excluded.roll5_avg_reaction_ms,
                roll5_avg_score       = excluded.roll5_avg_score,
                roll5_avg_difficulty  = excluded.roll5_avg_difficulty,
                all_avg_deaths        = excluded.all_avg_deaths,
                all_avg_accuracy      = excluded.all_avg_accuracy,
                all_avg_reaction_ms   = excluded.all_avg_reaction_ms,
                all_avg_score         = excluded.all_avg_score,
                last_seen             = excluded.last_seen
        """, (
            pid, total,
            avg("deaths",               last5),
            avg("accuracy",             last5),
            avg("avg_reaction_time_ms", last5),
            avg("score",                last5),
            avg("difficulty_level",     last5),
            avg("deaths",               rows),
            avg("accuracy",             rows),
            avg("avg_reaction_time_ms", rows),
            avg("score",                rows),
            rows[0]["timestamp"],       # last_seen = most recent session
        ))
    conn.commit()


# ── Public entry point ────────────────────────────────────────────────────────

def ingest(csv_path: str = DEFAULT_CSV, db_path: str = DEFAULT_DB) -> None:
    if not os.path.isfile(csv_path):
        print(f"[ingest] CSV not found: {csv_path}")
        return

    # read CSV
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[ingest] CSV is empty — nothing to do.")
        return

    print(f"[ingest] {len(rows)} rows read from {csv_path}")

    conn = _connect(db_path)
    _ensure_schema(conn)

    inserted = _upsert_sessions(conn, rows)
    print(f"[ingest] {inserted} new session(s) inserted (duplicates skipped)")

    affected_players = list({r["player_id"] for r in rows})
    _refresh_players(conn, affected_players)
    print(f"[ingest] players table refreshed for: {affected_players}")

    # summary
    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    total_players  = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    print(f"[ingest] DB totals → sessions: {total_sessions}, players: {total_players}")
    conn.close()
    print(f"[ingest] done → {db_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest game CSV into SQLite.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to sessions CSV")
    parser.add_argument("--db",  default=DEFAULT_DB,  help="Path to SQLite DB")
    args = parser.parse_args()
    ingest(csv_path=args.csv, db_path=args.db)