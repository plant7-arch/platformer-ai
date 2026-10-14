"""Draws one 480x270 frame of the level with numpy (no pygame): terrain, platforms, coins, enemy, knight.

Assets: Brackeys platformer bundle (CC0) in <assets>/sprites.  The terrain, backdrop, decorations and goal flag are
generated from the ASCII level (terrain.py), so changing level.py changes the picture too.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from platformer_ai import level as L
from platformer_ai import terrain
from platformer_ai import viewlogic as V

CLEAR = (77, 77, 77)                 # Godot's default clear colour, shown where no tile is drawn


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def blit(dst: np.ndarray, src: np.ndarray, x: int, y: int, flip: bool = False) -> None:
    """Alpha-blend an RGBA image onto an RGB array at integer (x, y), clipped to dst."""
    if flip:
        src = src[:, ::-1]
    h, w = src.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.uint16)
    d = dst[y0:y1, x0:x1].astype(np.uint16)
    dst[y0:y1, x0:x1] = ((s[..., :3].astype(np.uint16) * a + d * (255 - a)) // 255).astype(np.uint8)


def crop(world: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """world[y:y+h, x:x+w] as a copy; anything outside the world is the clear colour."""
    out = np.empty((h, w, 3), np.uint8)
    out[:] = CLEAR
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, world.shape[1]), min(y + h, world.shape[0])
    if x0 < x1 and y0 < y1:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = world[y0:y1, x0:x1]
    return out


@dataclass
class Assets:
    world: np.ndarray            # pre-drawn tilemaps (RGB)
    world_origin: tuple          # world pixel of world[0, 0]
    knight: np.ndarray
    slime: np.ndarray
    coin: np.ndarray
    platform: np.ndarray         # the 32x9 platform sprite


def build_world(tileset: np.ndarray, layers, sprites=()) -> tuple[np.ndarray, tuple[int, int]]:
    """Draw tile layers (each an (n, 4) array: tx, ty, ax, ay; the first one is the bottom layer; its extent sets the
    size), then loose sprites given as (rgba image, world x, world y of the top-left corner)."""
    allt = layers[0]
    tx0, ty0 = int(allt[:, 0].min()), int(allt[:, 1].min())
    tx1, ty1 = int(allt[:, 0].max()), int(allt[:, 1].max())
    world = np.empty(((ty1 - ty0 + 1) * V.TILE, (tx1 - tx0 + 1) * V.TILE, 3), np.uint8)
    world[:] = CLEAR
    for layer in layers:
        for tx, ty, ax, ay in layer.tolist():
            tile = tileset[ay * V.TILE:(ay + 1) * V.TILE, ax * V.TILE:(ax + 1) * V.TILE]
            blit(world, tile, (tx - tx0) * V.TILE, (ty - ty0) * V.TILE)
    for img, wx, wy in sprites:
        blit(world, img, int(wx) - tx0 * V.TILE, int(wy) - ty0 * V.TILE)
    return world, (tx0 * V.TILE, ty0 * V.TILE)


def keep_clear() -> np.ndarray:
    """Cells of the level where decorations must not stand (coins, starts, goal, the enemy's path)."""
    m = np.zeros((L.H, L.W), bool)

    def mark(x, y):
        r, c = int(np.floor(y)), int(np.floor(x))
        if 0 <= r < L.H and 0 <= c < L.W:
            m[r, c] = True

    for x, y in V.coin_positions(L.PLAT_START_X)[:L.N_STATIC_COINS]:
        mark(x, y)
    for x, y in [(L.SPAWN_X, L.SPAWN_Y), *zip(L.START_X, L.START_Y)]:
        mark(x, y - 0.01)
    for lo, hi, ey, ride in zip(L.ENEMY_MIN, L.ENEMY_MAX, L.ENEMY_Y, L.ENEMY_RIDE):
        if ride >= 0:                                       # riders walk on a platform, not on the ground
            continue
        for c in range(int(np.floor(lo)), int(np.floor(hi)) + 1):
            mark(c + 0.5, float(ey) - 0.01)
    return m | np.asarray(L.GOAL)


def make_world(tileset: np.ndarray, solid: np.ndarray, goal: np.ndarray, clear: np.ndarray | None = None,
               theme: str = "grass", decorate: bool = True):
    """Backdrop + terrain (+ decorations) + goal flags for any level grid."""
    h, w = solid.shape
    origin = (V.ORIGIN_TX, V.ORIGIN_TY)
    tx0, tx1 = V.ORIGIN_TX - 20, V.ORIGIN_TX + w + 20                 # margin: the camera sees past the level ends
    ty0, ty1 = terrain.BACKDROP_TOP, V.cam_limit_bottom(h) // V.TILE + 2
    layers = [terrain.background_tiles(tx0, tx1, ty0, ty1), terrain.terrain_tiles(solid, origin, theme)]
    if decorate:
        layers.insert(1, terrain.decor_tiles(solid, clear, origin))     # behind the ground, in front of the backdrop
    flags = []
    for c, r, n in terrain.goal_runs(goal):
        wx, wy = V.to_px(c, r)
        flags.append((terrain.flag_sprite(n), wx, wy))
    return build_world(tileset, layers, flags)


def load_assets(assets_dir: str | Path = "assets", theme: str = "grass", decorate: bool = True) -> Assets:
    d = Path(assets_dir) / "sprites"
    need = ["knight.png", "slime_green.png", "coin.png", "platforms.png", "world_tileset.png"]
    missing = [n for n in need if not (d / n).is_file()]
    if missing:
        raise FileNotFoundError(f"missing in {d}: {', '.join(missing)}  (copy the Brackeys sprites there, see README)")
    world, origin = make_world(load_rgba(d / "world_tileset.png"), L.SOLID, L.GOAL, keep_clear(), theme, decorate)
    x, y, w, h = V.PLATFORM_SPRITE
    plat = load_rgba(d / "platforms.png")[y:y + h, x:x + w]
    return Assets(world, origin, load_rgba(d / "knight.png"), load_rgba(d / "slime_green.png"),
                  load_rgba(d / "coin.png"), plat)


def platform_block(sprite: np.ndarray, n: int) -> np.ndarray:
    """The platform sprite as one seamless block n tiles wide: left cap, middle tiles, right cap (V.platform_pieces)."""
    return np.concatenate([sprite[:, x0:x1] for x0, x1 in V.platform_pieces(n)], axis=1)


def _sheet(sheet: np.ndarray, rect) -> np.ndarray:
    x, y, w, h = rect
    return sheet[y:y + h, x:x + w]


def compose(a: Assets, cam_tl: tuple[int, int], *, player_xy, player_frame, player_flip,
            enemy_x, enemy_flip, enemy_frame, coin_frame, coins_alive, plat_x) -> np.ndarray:
    """One frame, (270, 480, 3) uint8.  Positions are in env tiles; cam_tl is the view's top-left in world px.
    enemy_x / enemy_flip: one entry per enemy;  plat_x: one entry per moving platform (both may be empty)."""
    cx, cy = cam_tl
    frame = crop(a.world, cx - a.world_origin[0], cy - a.world_origin[1], V.SCREEN_W, V.SCREEN_H)

    def put(img, wx, wy, flip=False):                      # world pixel -> screen
        blit(frame, img, int(round(wx)) - cx, int(round(wy)) - cy, flip)

    ph = a.platform.shape[0]
    def block(cx_tiles, top, n):                           # one continuous platform block, n tiles wide
        img = platform_block(a.platform, n)
        px, py = V.to_px(cx_tiles, top)
        put(img, px - img.shape[1] / 2, py + V.PLATFORM_COLLIDER_H / 2 - ph / 2)
    for sx, sy, n in V.platform_runs():                    # static one-way platforms
        block(sx, sy, n)
    for x, top, half in zip(plat_x, L.PLAT_TOP, L.PLAT_HALF):      # moving platforms, any width
        block(float(x), float(top), int(round(2 * float(half))))

    coin_img = _sheet(a.coin, coin_frame)
    for alive, (x, y) in zip(coins_alive, V.coin_positions(plat_x)):
        if alive:
            wx, wy = V.to_px(x, y)
            put(coin_img, wx - 8, wy - 8)

    ox, oy = V.SPRITE_OFFSET["enemy"]
    for x, y, flip in zip(enemy_x, L.ENEMY_Y, enemy_flip):
        ex, ey = V.to_px(float(x), float(y))
        put(_sheet(a.slime, enemy_frame), ex + ox - 12, ey + oy - 12, flip)

    px, py = V.to_px(*player_xy)
    ox, oy = V.SPRITE_OFFSET["player"]
    put(_sheet(a.knight, player_frame), px + ox - 16, py + oy - 16, player_flip)
    return frame
