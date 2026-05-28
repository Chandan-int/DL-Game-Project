"""
Space Dodge — Pygame game with ML-driven difficulty adaptation.
After each session the DifficultyPredictor reads player telemetry
and adjusts enemy speed, spawn rate, damage, and obstacle density.
"""

import pygame
import random
import time
import csv
import os
import math
import torch
import urllib.request
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from difficulty_predictor import DifficultyPredictor, DifficultyParams

# Make sure torch is imported for the local model fallback inside get_difficulty
try:
    import torch
except ImportError:
    torch = None

# ── Constants ──────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 800, 600
FPS = 60
PLAYER_SPEED = 5
PLAYER_SIZE = 30
ENEMY_SIZE = 25
BULLET_SPEED = 10
BULLET_SIZE = 6
HIT_FLASH_FRAMES = 8

CSV_PATH = "data/sessions.csv"

# ── Colours ────────────────────────────────────────────────────────────────────
BLACK = (10, 10, 20)
WHITE = (255, 255, 255)
CYAN = (0, 220, 255)
RED = (255, 60, 60)
ORANGE = (255, 165, 0)
GREEN = (80, 220, 80)
GREY = (120, 120, 140)
YELLOW = (255, 230, 50)


# ── Azure endpoint config ─────────────────────────────────────────────────────

def _load_endpoint_config() -> dict | None:
    """Load ACI endpoint URL from saved config file."""
    config_path = Path("models/endpoint_config.json")
    if not config_path.exists():
        print("[difficulty] endpoint_config.json not found — using local model")
        return None
    try:
        config = json.loads(config_path.read_text())
        if not config.get("endpoint_url"):
            return None
        print(f"[difficulty] ✅ Azure endpoint loaded: {config['endpoint_url']}")
        return config
    except Exception:
        return None


_ENDPOINT_CONFIG = _load_endpoint_config()


def get_difficulty(player_id: str, predictor_instance: DifficultyPredictor = None) -> float:
    """
    Returns difficulty 0.0–1.0.
    Priority: Azure ACI endpoint → local model → fallback 0.5
    """
    # ── Try Azure endpoint first ──────────────────────────────────────────────
    if _ENDPOINT_CONFIG:
        last = _get_last_session(player_id)
        if last:
            try:
                score = _call_azure_endpoint(last)
                print(f"[difficulty] ☁️  Azure  player={player_id} score={score:.3f}")
                return score
            except Exception as e:
                print(f"[difficulty] ⚠️  Azure failed: {e} — falling back to local")

    # ── Fall back to local model ───────────────────────────────────────────────
    # We attempt to find the _PREDICTOR array properties attached to our predictor instance
    _PREDICTOR = getattr(predictor_instance, "_PREDICTOR", None) if predictor_instance else None

    if _PREDICTOR is not None and torch is not None:
        # Simple extraction if helper function isn't exposed globally
        last_session = _get_last_session(player_id)
        if last_session:
            FEATURE_COLS = [
                "level", "deaths", "accuracy",
                "avg_reaction_time_ms", "completion_time_sec",
                "score", "difficulty_level",
            ]
            model, mean, scale, _ = _PREDICTOR
            try:
                row = [
                    (float(last_session.get(c, 0)) - m) / (s + 1e-8)
                    for c, m, s in zip(FEATURE_COLS, mean, scale)
                ]
                x = torch.tensor([row], dtype=torch.float32)
                with torch.no_grad():
                    score = float(model(x).item())
                score = round(max(0.15, min(0.85, score)), 3)
                print(f"[difficulty] 💻 Local  player={player_id} score={score:.3f}")
                return score
            except Exception as e:
                print(f"[difficulty] ⚠️ Local pipeline failed: {e}")

    print(f"[difficulty] ⚠️  No history or model assets available for {player_id} — using 0.5")
    return 0.5


def _get_last_session(player_id: str) -> dict | None:
    """Read the most recent session row for this player from CSV."""
    if not os.path.isfile(CSV_PATH):
        return None
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["player_id"] == player_id]
    return rows[-1] if rows else None


def _call_azure_endpoint(session: dict) -> float:
    """POST session telemetry to ACI endpoint, return difficulty score."""
    url = _ENDPOINT_CONFIG["endpoint_url"]
    payload = {
        "level": float(session.get("level", 1)),
        "deaths": float(session.get("deaths", 0)),
        "accuracy": float(session.get("accuracy", 0)),
        "avg_reaction_time_ms": float(session.get("avg_reaction_time_ms", 500)),
        "completion_time_sec": float(session.get("completion_time_sec", 60)),
        "score": float(session.get("score", 0)),
        "difficulty_level": float(session.get("difficulty_level", 0.5)),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=5)
    result = json.loads(resp.read())
    score = float(result["difficulty_score"])
    return round(max(0.15, min(0.85, score)), 3)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Telemetry:
    """One session's worth of player data."""
    player_id: str = "p_001"
    level: int = 1
    deaths: int = 0
    shots_fired: int = 0
    shots_hit: int = 0
    reaction_times_ms: list = field(default_factory=list)
    completion_time_sec: float = 0.0
    score: int = 0
    difficulty_level: float = 0.5

    @property
    def accuracy(self) -> float:
        if self.shots_fired == 0:
            return 0.0
        return round(self.shots_hit / self.shots_fired, 3)

    @property
    def avg_reaction_time_ms(self) -> float:
        if not self.reaction_times_ms:
            return 500.0
        return round(sum(self.reaction_times_ms) / len(self.reaction_times_ms), 1)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "level": self.level,
            "deaths": self.deaths,
            "accuracy": self.accuracy,
            "avg_reaction_time_ms": self.avg_reaction_time_ms,
            "completion_time_sec": round(self.completion_time_sec, 2),
            "score": self.score,
            "difficulty_level": self.difficulty_level,
        }


# ── Sprites ────────────────────────────────────────────────────────────────────

class Player(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()
        self.image = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, CYAN, [
            (PLAYER_SIZE // 2, 0),
            (0, PLAYER_SIZE),
            (PLAYER_SIZE, PLAYER_SIZE),
        ])
        self.rect = self.image.get_rect(center=(SCREEN_W // 2, SCREEN_H - 80))
        self.hp = 100
        self.max_hp = 100
        self.flash_timer = 0
        self.last_enemy_seen_time: float | None = None

    def update(self, keys: pygame.key.ScancodeWrapper) -> None:
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= PLAYER_SPEED
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_W:
            self.rect.x += PLAYER_SPEED
        if keys[pygame.K_UP] and self.rect.top > 0:
            self.rect.y -= PLAYER_SPEED
        if keys[pygame.K_DOWN] and self.rect.bottom < SCREEN_H:
            self.rect.y += PLAYER_SPEED
        if self.flash_timer > 0:
            self.flash_timer -= 1

    def take_damage(self, amount: int) -> None:
        self.hp = max(0, self.hp - amount)
        self.flash_timer = HIT_FLASH_FRAMES

    def draw(self, surface: pygame.Surface) -> None:
        if self.flash_timer % 2 == 0:
            surface.blit(self.image, self.rect)
        bar_w = 80
        bar_x = self.rect.centerx - bar_w // 2
        bar_y = self.rect.top - 12
        pygame.draw.rect(surface, GREY, (bar_x, bar_y, bar_w, 6))
        hp_w = int(bar_w * self.hp / self.max_hp)
        color = GREEN if self.hp > 50 else ORANGE if self.hp > 25 else RED
        pygame.draw.rect(surface, color, (bar_x, bar_y, hp_w, 6))


class Enemy(pygame.sprite.Sprite):
    def __init__(self, speed: float) -> None:
        super().__init__()
        self.image = pygame.Surface((ENEMY_SIZE, ENEMY_SIZE), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, RED, [
            (ENEMY_SIZE // 2, ENEMY_SIZE),
            (0, 0),
            (ENEMY_SIZE, 0),
        ])
        self.rect = self.image.get_rect(
            x=random.randint(0, SCREEN_W - ENEMY_SIZE), y=-ENEMY_SIZE
        )
        self.speed = speed
        self.spawn_time = time.time()

    def update(self) -> None:
        self.rect.y += self.speed
        if self.rect.top > SCREEN_H:
            self.kill()


class Bullet(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int) -> None:
        super().__init__()
        self.image = pygame.Surface((BULLET_SIZE, BULLET_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(self.image, YELLOW,
                           (BULLET_SIZE // 2, BULLET_SIZE // 2), BULLET_SIZE // 2)
        self.rect = self.image.get_rect(center=(x, y))

    def update(self) -> None:
        self.rect.y -= BULLET_SPEED
        if self.rect.bottom < 0:
            self.kill()


# ── Telemetry logger ───────────────────────────────────────────────────────────

def log_session(telemetry: Telemetry) -> None:
    """Append one session row to data/sessions.csv."""
    Path("data").mkdir(exist_ok=True)
    log_path = Path(CSV_PATH)
    row = telemetry.to_dict()
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[telemetry] session logged → {log_path}")
    print(f"[telemetry] {row}")


# ── HUD helpers ────────────────────────────────────────────────────────────────

def draw_hud(
        surface: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        score: int,
        hp: int,
        level: int,
        params: DifficultyParams,
        elapsed: float,
        kills: int,
        kill_target: int,
) -> None:
    pygame.draw.rect(surface, (20, 20, 35), (0, 0, SCREEN_W, 40))
    surface.blit(font.render(f"Score: {score}", True, WHITE), (10, 8))
    surface.blit(font.render(f"Level: {level}", True, CYAN), (200, 8))
    surface.blit(font.render(f"Kills: {kills}/{kill_target}", True, GREEN), (350, 8))
    surface.blit(font.render(f"Time:  {elapsed:.0f}s", True, GREY), (520, 8))
    surface.blit(font.render(f"HP: {hp}", True, RED), (680, 8))

    diff_text = f"Difficulty: {params.score:.2f}"
    diff_color = GREEN if params.score < 0.4 else ORANGE if params.score < 0.7 else RED
    surface.blit(small_font.render(diff_text, True, diff_color), (10, SCREEN_H - 55))
    surface.blit(small_font.render(f"Speed:  {params.enemy_speed:.1f}", True, GREY), (10, SCREEN_H - 38))
    surface.blit(small_font.render(f"Spawn:  {params.spawn_rate:.2f}/s", True, GREY), (10, SCREEN_H - 22))


def draw_overlay(
        surface: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        title: str,
        lines: list[str],
        color: tuple,
) -> None:
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))
    title_surf = font.render(title, True, color)
    surface.blit(title_surf, title_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 60)))
    for i, line in enumerate(lines):
        s = small_font.render(line, True, WHITE)
        surface.blit(s, s.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + i * 28)))


# ── Difficulty scaling ─────────────────────────────────────────────────────────

# ── Difficulty scaling ─────────────────────────────────────────────────────────

def difficulty_to_params(score: float) -> DifficultyParams:
    """
    Convert ML difficulty score (0–1) into gameplay parameters.
    """

    score = max(0.15, min(0.85, score))

    return DifficultyParams(

        # Main difficulty score
        score=score,

        # Enemy movement speed
        enemy_speed=2.0 + score * 5.0,

        # Enemies spawned per second
        spawn_rate=0.5 + score * 2.5,

        # Damage player takes
        damage=int(5 + score * 20),

        # Extra gameplay density scaling
        obstacle_density=0.1 + score * 0.5,
    )
#── Main game ──────────────────────────────────────────────────────────────────

def run_session(
        level: int,
        params: DifficultyParams,
        predictor: DifficultyPredictor,
) -> Telemetry:
    """Run one game session. Returns telemetry for the session."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Space Dodge — Level {level}")
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 22)
    fps_clock = pygame.time.Clock()

    all_sprites = pygame.sprite.Group()
    enemies = pygame.sprite.Group()
    bullets = pygame.sprite.Group()

    player = Player()
    all_sprites.add(player)

    telemetry = Telemetry(level=level, difficulty_level=params.score)
    kill_target = 20 + level * 5
    kills = 0
    start_time = time.time()
    last_spawn = time.time()
    spawn_interval = 1.0 / params.spawn_rate

    state = "playing"

    while True:
        elapsed = time.time() - start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.KEYDOWN:
                if state == "playing":
                    if event.key == pygame.K_SPACE:
                        b = Bullet(player.rect.centerx, player.rect.top)
                        all_sprites.add(b)
                        bullets.add(b)
                        telemetry.shots_fired += 1

                        if player.last_enemy_seen_time:
                            rt = (time.time() - player.last_enemy_seen_time) * 1000
                            if rt < 3000:
                                telemetry.reaction_times_ms.append(rt)
                            player.last_enemy_seen_time = None

                elif state in ("dead", "win"):
                    if event.key == pygame.K_r:
                        telemetry.completion_time_sec = elapsed
                        return telemetry
                    if event.key == pygame.K_q:
                        pygame.quit()
                        raise SystemExit

        if state == "playing":
            keys = pygame.key.get_pressed()
            player.update(keys)
            enemies.update()
            bullets.update()

            if time.time() - last_spawn > spawn_interval:
                e = Enemy(speed=params.enemy_speed)
                all_sprites.add(e)
                enemies.add(e)
                last_spawn = time.time()
                if player.last_enemy_seen_time is None:
                    player.last_enemy_seen_time = time.time()

            hits = pygame.sprite.groupcollide(bullets, enemies, True, True)
            for _ in hits:
                kills += 1
                telemetry.shots_hit += 1
                telemetry.score += int(100 * params.score + 50)

            enemy_hits = pygame.sprite.spritecollide(player, enemies, True)
            for _ in enemy_hits:
                player.take_damage(params.damage)
                telemetry.score = max(0, telemetry.score - 20)

            if player.hp <= 0:
                telemetry.deaths += 1
                telemetry.completion_time_sec = elapsed
                state = "dead"

            if kills >= kill_target:
                telemetry.completion_time_sec = elapsed
                state = "win"

        screen.fill(BLACK)

        random.seed(42)
        for _ in range(80):
            sx = random.randint(0, SCREEN_W)
            sy = random.randint(0, SCREEN_H)
            pygame.draw.circle(screen, (40, 40, 60), (sx, sy), 1)
        random.seed()

        all_sprites.draw(screen)
        player.draw(screen)

        draw_hud(screen, font, small_font,
                 telemetry.score, player.hp, level,
                 params, elapsed, kills, kill_target)

        if state == "dead":
            draw_overlay(screen, font, small_font,
                         "YOU DIED", [
                             f"Score: {telemetry.score}  |  Accuracy: {telemetry.accuracy:.0%}",
                             f"Difficulty was: {params.score:.2f}",
                             "Press R to retry  |  Q to quit",
                         ], RED)

        if state == "win":
            draw_overlay(screen, font, small_font,
                         "LEVEL CLEAR!", [
                             f"Score: {telemetry.score}  |  Time: {elapsed:.1f}s",
                             f"Accuracy: {telemetry.accuracy:.0%}  |  Difficulty: {params.score:.2f}",
                             "Press R for next level  |  Q to quit",
                         ], GREEN)

        pygame.display.flip()
        fps_clock.tick(FPS)


def main() -> None:
    """
    Main loop — Azure difficulty → gameplay params → session loop.
    """

    player_id = "p_001"
    level = 1

    predictor = DifficultyPredictor()

    print(f"[game] Starting player={player_id}")

    while True:

        # Get ML difficulty score
        difficulty = get_difficulty(player_id, predictor)

        # Convert score into gameplay parameters
        params = difficulty_to_params(difficulty)

        print(f"\n[game] ── Level {level} ──────────────────────────")
        print(f"[game] Difficulty : {difficulty:.3f}")
        print(f"[game] Params     : {params}")

        # Run actual gameplay session
        telemetry = run_session(
            level=level,
            params=params,
            predictor=predictor,
        )

        # Save telemetry
        telemetry.player_id = player_id
        log_session(telemetry)

        print(f"[game] Session complete — fetching new difficulty...")

        level += 1

if __name__ == "__main__":
    main()