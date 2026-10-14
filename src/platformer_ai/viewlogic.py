"""Pure-Python logic of the viewer (no pygame, no jax): coordinates, animation, camera, controls.

Kept separate so it can be unit-tested without a window.
"""
from __future__ import annotations

import math

from platformer_ai import level as L
from platformer_ai import terrain

# ---- screen / world -----------------------------------------------------------------------
SCREEN_W, SCREEN_H = 480, 270        # the Godot viewport (1920x1080 at zoom 4)
TILE = 16                            # pixels per tile in the art
ORIGIN_TX, ORIGIN_TY = 10, 5         # our grid cell (0, 0) is tile (10, 5) of the original tilemaps
CAM_OFFSET_Y = -4                    # Godot Camera2D offset
CAM_LIMIT_BOTTOM = (L.H + ORIGIN_TY) * TILE + 30   # Godot limit_bottom (270 for the 10-row level): a little below the map
CAM_SMOOTH = 5.0                     # Godot position_smoothing_speed (default)
SPRITE_OFFSET = {"player": (1, -12), "enemy": (0, -12)}      # sprite centre relative to the feet
PLATFORM_SPRITE = (16, 0, 32, 9)     # region in platforms.png (x, y, w, h)
PLATFORM_COLLIDER_H = 8


def to_px(x: float, y: float) -> tuple[float, float]:
    """Env coordinates (tiles, y down) -> world pixels of the art."""
    return (x + ORIGIN_TX) * TILE, (y + ORIGIN_TY) * TILE


# ---- animations (frames are (x, y, w, h) rectangles in the sprite sheet) -------------------
def _row(y, w, h, n, x0=0):
    return [(x0 + i * w, y, w, h) for i in range(n)]


ANIMS = {
    "player": {
        "idle": dict(fps=10.0, frames=_row(0, 32, 32, 4)),
        "run": dict(fps=10.0, frames=_row(64, 32, 32, 8) + _row(96, 32, 32, 8)),
        "jump": dict(fps=5.0, frames=[(64, 160, 32, 32)]),
    },
    "enemy": {"walk": dict(fps=10.0, frames=_row(24, 24, 24, 4))},
    "coin": {"spin": dict(fps=10.0, frames=_row(0, 16, 16, 12))},
}


def frame_index(t: float, fps: float, n: int) -> int:
    """Looping animation: frame shown t seconds after it started."""
    return int(math.floor(max(t, 0.0) * fps + 1e-9)) % n


class Animator:
    """Like AnimatedSprite2D.play(): switching animation restarts it, same animation keeps running."""

    def __init__(self, table: dict):
        self.table, self.name, self.t0 = table, None, 0.0

    def frame(self, name: str, now: float) -> tuple[int, int, int, int]:
        if name != self.name:
            self.name, self.t0 = name, now
        a = self.table[name]
        return a["frames"][frame_index(now - self.t0, a["fps"], len(a["frames"]))]


def player_anim(vx: float, vy: float) -> str:
    """The original script: in the air -> jump; on the ground -> idle or run.
    On the ground vy is exactly 0 in this env (gravity then landing resets it every step)."""
    if vy != 0.0:
        return "jump"
    return "run" if vx != 0.0 else "idle"


def facing_left(vx: float, was_left: bool) -> bool:
    """Sprite flips when moving left, keeps its side when standing still."""
    if vx < 0:
        return True
    if vx > 0:
        return False
    return was_left


# ---- camera ---------------------------------------------------------------------------------
class Camera:
    """Follows the player like the Godot Camera2D (offset, bottom limit, position smoothing)."""

    def __init__(self):
        self.x = self.y = 0.0

    @staticmethod
    def target(px: float, py: float) -> tuple[float, float]:
        return px, min(py + CAM_OFFSET_Y, CAM_LIMIT_BOTTOM - SCREEN_H / 2)

    def snap(self, px: float, py: float) -> None:
        self.x, self.y = self.target(px, py)

    def update(self, px: float, py: float, dt: float) -> None:
        tx, ty = self.target(px, py)
        k = 1.0 - math.exp(-CAM_SMOOTH * dt)
        self.x += (tx - self.x) * k
        self.y += (ty - self.y) * k

    def top_left(self) -> tuple[int, int]:
        """Integer pixel of the view's top-left corner (rounded so the pixel art does not shimmer)."""
        return int(round(self.x - SCREEN_W / 2)), int(round(self.y - SCREEN_H / 2))


# ---- controls -------------------------------------------------------------------------------
def keys_to_action(left: bool, right: bool, jump: bool) -> int:
    """Same action coding as the env: horizontal = a % 3 (none, left, right), jump if a >= 3."""
    h = 0 if left == right else (1 if left else 2)
    return h + (3 if jump else 0)


# ---- what to draw ----------------------------------------------------------------------------
def platform_runs() -> list[tuple[float, float]]:
    """Static one-way tiles grouped into runs -> (centre x, top y) in env tiles; one sprite per 2 tiles."""
    out = []
    for r in range(L.H):
        c = 0
        while c < L.W:
            if L.ONEWAY[r, c]:
                c0 = c
                while c < L.W and L.ONEWAY[r, c]:
                    c += 1
                n = c - c0
                for i in range(0, n, 2):
                    w = min(2, n - i)
                    out.append((c0 + i + w / 2.0, float(r)))
            else:
                c += 1
    return out


def coin_positions(plat_x: float) -> list[tuple[float, float]]:
    """Centre of every coin in env tiles (the last one rides on the moving platform)."""
    out = []
    for i in range(L.N_COINS):
        if L.COIN_RIDES[i]:
            out.append((plat_x + L.RIDER_COIN_OFFSET[0], L.PLAT_TOP + L.RIDER_COIN_OFFSET[1]))
        else:
            out.append((float(L.COIN_X[i]), float(L.COIN_Y[i])))
    return out


def world_labels() -> list[tuple[str, float, float]]:
    """Signs of the level (level.LABELS) as (text, left, top) in world pixels."""
    return [(t, *to_px(x, y)) for t, x, y in L.LABELS]


SCORE_LABEL_WIDTH = 102                # the "You collected N coins." sign is centred above the goal flag, wrapped to this


def score_label_anchor():
    """(centre x, bottom y) in world pixels of the coin-count sign: just above the first goal flag; None without a goal."""
    runs = goal_runs()
    if not runs:
        return None
    c, r, n = runs[0]
    x, y = to_px(c + 0.5, r)
    return x, y - 4


def goal_runs():
    return terrain.goal_runs(L.GOAL)


def score_text(n: int) -> str:
    return f"You collected {n} coin{'' if n == 1 else 's'}."


def wrap_words(text: str, width: float, measure) -> list[str]:
    """Greedy word wrap; `measure(s)` returns the pixel width of s."""
    lines, cur = [], ""
    for w in text.split():
        trial = w if not cur else cur + " " + w
        if cur and measure(trial) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# ---- smoothing between 30 Hz sim steps and 60 Hz drawing ---------------------------------------
def lerp_pos(a: float, b: float, alpha: float, max_jump: float = 2.0) -> float:
    """Position between the previous (a) and current (b) sim state; no smoothing across a reset/teleport."""
    if abs(b - a) > max_jump:
        return b
    return a + (b - a) * min(max(alpha, 0.0), 1.0)
