# difficulty_predictor.py  (v2 — fixed scaling)
"""
Loads the best registered MLflow model and predicts difficulty
from a player telemetry dict. Uses manual min-max normalization
to match the pre-normalized training data format.
"""

import json
import numpy as np
import torch
import mlflow.pytorch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DifficultyParams:
    """All game parameters controlled by the ML model."""
    score:            float
    enemy_speed:      float
    spawn_rate:       float
    damage:           int
    obstacle_density: float

    def __repr__(self) -> str:
        return (
            f"DifficultyParams(score={self.score:.2f} | "
            f"speed={self.enemy_speed:.1f} | "
            f"spawn={self.spawn_rate:.2f}/s | "
            f"dmg={self.damage} | "
            f"density={self.obstacle_density:.2f})"
        )


# ── Feature order must match train.py exactly ─────────────────────────────────
FEATURE_NAMES = [
    "level",
    "deaths",
    "accuracy",
    "avg_reaction_time_ms",
    "completion_time_sec",
    "score",
    "difficulty_level",
]

# ── Raw game value ranges for manual 0→1 normalization ───────────────────────
# These define what the game can produce — adjust if your game changes
FEATURE_RANGES = {
    "level":                (1,    10),      # game levels 1-10
    "deaths":               (0,    10),      # deaths per session
    "accuracy":             (0.0,  1.0),     # already 0-1
    "avg_reaction_time_ms": (100,  2000),    # ms — slow players ~2000
    "completion_time_sec":  (10,   180),     # seconds per level
    "score":                (0,    10000),   # raw game score
    "difficulty_level":     (0.0,  1.0),     # already 0-1
}


def normalize_telemetry(telemetry: dict) -> np.ndarray:
    """
    Min-max normalize raw telemetry values to [0, 1] using known game ranges.
    This matches how features.csv was prepared during training.
    """
    normalized = []
    for feature in FEATURE_NAMES:
        raw_val = float(telemetry[feature])
        lo, hi  = FEATURE_RANGES[feature]
        # Clip to range, then normalize
        clipped = max(lo, min(hi, raw_val))
        norm    = (clipped - lo) / (hi - lo + 1e-8)
        normalized.append(norm)
    return np.array(normalized, dtype=np.float32)


class DifficultyPredictor:
    """
    Loads the best MLflow model and predicts difficulty
    from raw player telemetry using min-max normalization.
    """

    SPEED_MIN,   SPEED_MAX   = 2.0,  8.0
    SPAWN_MIN,   SPAWN_MAX   = 0.5,  3.0
    DAMAGE_MIN,  DAMAGE_MAX  = 5,    25
    DENSITY_MIN, DENSITY_MAX = 0.1,  0.9

    def __init__(
        self,
        model_name:     str   = "DifficultyMLP",
        tracking_uri:   str   = "http://127.0.0.1:5000",
        fallback_score: float = 0.5,
    ) -> None:
        self.fallback_score = fallback_score
        self.model = None

        mlflow.set_tracking_uri(tracking_uri)
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load latest model version from MLflow registry."""
        try:
            uri = f"models:/{model_name}/latest"
            self.model = mlflow.pytorch.load_model(uri)
            self.model.eval()
            print(f"[predictor] ✅ Model loaded: {uri}")
        except Exception as e:
            print(f"[predictor] ⚠️  Model load failed: {e}")
            print(f"[predictor]    Using fallback score={self.fallback_score}")

    def predict(self, telemetry: dict) -> DifficultyParams:
        """
        Takes raw telemetry dict → returns DifficultyParams.
        Normalizes inputs to [0,1] before inference.
        """
        score = self._predict_score(telemetry)
        return self._score_to_params(score)

    def _predict_score(self, telemetry: dict) -> float:
        """Normalize → inference → return score in [0, 1]."""
        if self.model is None:
            return self.fallback_score

        try:
            features = normalize_telemetry(telemetry)
            print(f"[predictor] normalized: {np.round(features, 3)}")

            with torch.no_grad():
                x   = torch.tensor(features).unsqueeze(0)  # [1, 7]
                raw = self.model(x).item()

            print(f"[predictor] raw output : {raw:.6f}")
            score = float(np.clip(raw, 0.0, 1.0))
            print(f"[predictor] 🎯 score   : {score:.3f}")
            return score

        except Exception as e:
            print(f"[predictor] ⚠️  Prediction error: {e}")
            return self.fallback_score

    def _score_to_params(self, score: float) -> DifficultyParams:
        """Linearly map score (0→1) to game parameter ranges."""
        def lerp(lo: float, hi: float) -> float:
            return lo + score * (hi - lo)

        return DifficultyParams(
            score            = score,
            enemy_speed      = lerp(self.SPEED_MIN,   self.SPEED_MAX),
            spawn_rate       = lerp(self.SPAWN_MIN,   self.SPAWN_MAX),
            damage           = int(lerp(self.DAMAGE_MIN, self.DAMAGE_MAX)),
            obstacle_density = lerp(self.DENSITY_MIN, self.DENSITY_MAX),
        )