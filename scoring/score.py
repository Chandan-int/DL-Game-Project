# scoring/score.py
"""
Lightweight Flask scoring server.
Runs inside Azure Container Instance.
POST /score  → {"difficulty_score": 0.644}
"""

import json
import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify


class DifficultyMLP(nn.Module):
    """Must match train.py architecture exactly."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),  nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


FEATURE_NAMES = [
    "level", "deaths", "accuracy",
    "avg_reaction_time_ms", "completion_time_sec",
    "score", "difficulty_level",
]
FEATURE_RANGES = {
    "level":                (1,    10),
    "deaths":               (0,    10),
    "accuracy":             (0.0,  1.0),
    "avg_reaction_time_ms": (100,  2000),
    "completion_time_sec":  (10,   180),
    "score":                (0,    10000),
    "difficulty_level":     (0.0,  1.0),
}

# Load model once at startup
model = DifficultyMLP()
model.load_state_dict(
    torch.load("difficulty_model.pt", map_location="cpu", weights_only=True)
)
model.eval()
print("[score] ✅ Model loaded")

app = Flask(__name__)


@app.route("/score", methods=["POST"])
def score():
    """Accept raw telemetry dict, return difficulty score."""
    telemetry = request.get_json()

    features = []
    for name in FEATURE_NAMES:
        val      = float(telemetry.get(name, 0))
        lo, hi   = FEATURE_RANGES[name]
        norm     = max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-8)))
        features.append(norm)

    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        difficulty = float(model(x).item())

    difficulty = round(max(0.15, min(0.85, difficulty)), 4)
    print(f"[score] predicted: {difficulty:.4f}")
    return jsonify({"difficulty_score": difficulty})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)