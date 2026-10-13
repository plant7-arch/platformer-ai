"""Draws one 480x270 frame of the level with numpy (no pygame): tilemaps, platforms, coins, enemy, knight.

Assets: Brackeys platformer bundle (CC0) in <assets>/sprites.  Tile layout: world_tiles.npz, decoded from the
original scene (columns: tile x, tile y, atlas x, atlas y).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from platformer_ai import level as L
from platformer_ai import viewlogic as V

CLEAR = (77, 77, 77)                 # Godot's default clear colour, shown where no tile is drawn
TILES_FILE = Path(__file__).with_name("world_tiles.npz")


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


def build_world(tileset: np.ndarray, layers) -> tuple[np.ndarray, tuple[int, int]]:
    """Draw tilemap layers (each an (n, 4) array: tx, ty, ax, ay), first one at the bottom."""
    allt = np.concatenate(layers)
    tx0, ty0 = int(allt[:, 0].min()), int(allt[:, 1].min())
    tx1, ty1 = int(allt[:, 0].max()), int(allt[:, 1].max())
    world = np.empty(((ty1 - ty0 + 1) * V.TILE, (tx1 - tx0 + 1) * V.TILE, 3), np.uint8)
    world[:] = CLEAR
    for layer in layers:
        for tx, ty, ax, ay in layer.tolist():
            tile = tileset[ay * V.TILE:(ay + 1) * V.TILE, ax * V.TILE:(ax + 1) * V.TILE]
            blit(world, tile, (tx - tx0) * V.TILE, (ty - ty0) * V.TILE)
    return world, (tx0 * V.TILE, ty0 * V.TILE)


def load_assets(assets_dir: str | Path = "assets") -> Assets:
    d = Path(assets_dir) / "sprites"
    need = ["knight.png", "slime_green.png", "coin.png", "platforms.png", "world_tileset.png"]
    missing = [n for n in need if not (d / n).is_file()]
    if missing:
        raise FileNotFoundError(f"missing in {d}: {', '.join(missing)}  (copy the Brackeys sprites there, see README)")
    tiles = np.load(TILES_FILE)
    world, origin = build_world(load_rgba(d / "world_tileset.png"), [tiles["background"], tiles["mid"]])
    x, y, w, h = V.PLATFORM_SPRITE
    plat = load_rgba(d / "platforms.png")[y:y + h, x:x + w]
    return Assets(world, origin, load_rgba(d / "knight.png"), load_rgba(d / "slime_green.png"),
                  load_rgba(d / "coin.png"), plat)


def _sheet(sheet: np.ndarray, rect) -> np.ndarray:
    x, y, w, h = rect
    return sheet[y:y + h, x:x + w]


def compose(a: Assets, cam_tl: tuple[int, int], *, player_xy, player_frame, player_flip,
            enemy_x, enemy_flip, enemy_frame, coin_frame, coins_alive, plat_x) -> np.ndarray:
    """One frame, (270, 480, 3) uint8.  Positions are in env tiles; cam_tl is the view's top-left in world px."""
    cx, cy = cam_tl
    frame = crop(a.world, cx - a.world_origin[0], cy - a.world_origin[1], V.SCREEN_W, V.SCREEN_H)

    def put(img, wx, wy, flip=False):                      # world pixel -> screen
        blit(frame, img, int(round(wx)) - cx, int(round(wy)) - cy, flip)

    ph, pw = a.platform.shape[:2]
    for sx, sy in V.platform_runs():                       # static one-way platforms
        px, py = V.to_px(sx, sy)
        put(a.platform, px - pw / 2, py + V.PLATFORM_COLLIDER_H / 2 - ph / 2)
    px, py = V.to_px(plat_x, L.PLAT_TOP)                   # moving platform
    put(a.platform, px - pw / 2, py + V.PLATFORM_COLLIDER_H / 2 - ph / 2)

    coin_img = _sheet(a.coin, coin_frame)
    for alive, (x, y) in zip(coins_alive, V.coin_positions(plat_x)):
        if alive:
            wx, wy = V.to_px(x, y)
            put(coin_img, wx - 8, wy - 8)

    ex, ey = V.to_px(enemy_x, L.ENEMY_Y)
    ox, oy = V.SPRITE_OFFSET["enemy"]
    put(_sheet(a.slime, enemy_frame), ex + ox - 12, ey + oy - 12, enemy_flip)

    px, py = V.to_px(*player_xy)
    ox, oy = V.SPRITE_OFFSET["player"]
    put(_sheet(a.knight, player_frame), px + ox - 16, py + oy - 16, player_flip)
    return frame
