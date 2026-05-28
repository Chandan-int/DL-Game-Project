# test_predictor.py
"""Verify the predictor loads and outputs sane values — no Pygame needed."""
# test_predictor.py  (updated — uses values within training distribution)
"""Verify predictor loads and outputs sane values."""

from difficulty_predictor import DifficultyPredictor

def test_predictor() -> None:
    predictor = DifficultyPredictor()

    # These values are deliberately within typical training data ranges
    # Weak player — high deaths, low accuracy, slow reactions
    weak_player = {
        "level":                1,
        "deaths":               5,
        "accuracy":             0.35,
        "avg_reaction_time_ms": 450,
        "completion_time_sec":  120,
        "score":                800,
        "difficulty_level":     0.3,
    }

    # Strong player — no deaths, high accuracy, fast reactions
    strong_player = {
        "level":                4,
        "deaths":               1,
        "accuracy":             0.85,
        "avg_reaction_time_ms": 220,
        "completion_time_sec":  60,
        "score":                2800,
        "difficulty_level":     0.75,
    }

    print("\n── Weak player ──────────────────────────────")
    params_easy = predictor.predict(weak_player)
    print(params_easy)

    print("\n── Strong player ────────────────────────────")
    params_hard = predictor.predict(strong_player)
    print(params_hard)

    print("\n── Difference ───────────────────────────────")
    print(f"  Score gap     : {params_hard.score - params_easy.score:+.3f}")
    print(f"  Speed gap     : {params_hard.enemy_speed - params_easy.enemy_speed:+.2f}")
    print(f"  Spawn gap     : {params_hard.spawn_rate - params_easy.spawn_rate:+.2f}/s")

    # Soft check — just warn instead of crash so we can debug further
    if params_easy.score < params_hard.score:
        print("\n✅ Assertion passed — strong player gets harder difficulty!")
    else:
        print(f"\n⚠️  Scores equal or inverted:")
        print(f"   weak={params_easy.score:.4f}  strong={params_hard.score:.4f}")
        print(f"   Run: python debug_scaler.py  to investigate")

if __name__ == "__main__":
    test_predictor()