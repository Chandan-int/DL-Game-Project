# debug_scaler.py
"""Show scaler mean/scale and what test inputs look like after scaling."""

import json
import numpy as np

FEATURE_NAMES = [
    "level", "deaths", "accuracy",
    "avg_reaction_time_ms", "completion_time_sec",
    "score", "difficulty_level",
]

with open("models/scaler.json") as f:
    data = json.load(f)

mean  = np.array(data["mean"],  dtype=np.float32)
scale = np.array(data["scale"], dtype=np.float32)

print("── Scaler internals ─────────────────────────────────────")
print(f"{'Feature':<25} {'Mean':>10} {'Scale':>10}")
print("─" * 47)
for name, m, s in zip(FEATURE_NAMES, mean, scale):
    print(f"{name:<25} {m:>10.3f} {s:>10.3f}")

# Test inputs
weak_player = {
    "level": 1, "deaths": 8, "accuracy": 0.25,
    "avg_reaction_time_ms": 600, "completion_time_sec": 180,
    "score": 400, "difficulty_level": 0.3,
}
strong_player = {
    "level": 5, "deaths": 0, "accuracy": 0.92,
    "avg_reaction_time_ms": 180, "completion_time_sec": 45,
    "score": 3500, "difficulty_level": 0.8,
}

print("\n── Scaled features ──────────────────────────────────────")
print(f"{'Feature':<25} {'Weak (raw→scaled)':>20} {'Strong (raw→scaled)':>22}")
print("─" * 70)

for name, m, s in zip(FEATURE_NAMES, mean, scale):
    w_raw = weak_player[name]
    s_raw = strong_player[name]
    w_scaled = (w_raw - m) / (s + 1e-8)
    s_scaled = (s_raw - m) / (s + 1e-8)
    print(f"{name:<25} {w_raw:>6} → {w_scaled:>+7.3f}      {s_raw:>6} → {s_scaled:>+7.3f}")