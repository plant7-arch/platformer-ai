"""Terrain art generated from the level itself (numpy only, no pygame / PIL).

Editing the ASCII level in level.py is enough: every solid cell gets a tile of the tileset, the backdrop is
generated for any width, and the goal gets a flag.  The tileset has two tiles per material (a top and a body),
so the rule is simple: a solid cell with an empty cell above it is a *top* tile (grass), any other solid cell is a
*body* tile (dirt).  A cheap hash of the cell position picks between body variants, so it is not a wall of
identical tiles, and it is the same every run.

Tiles are (tx, ty, ax, ay): where to draw (in tiles) and which tile of world_tileset.png (atlas column, row).
"""
from __future__ import annotations

import numpy as np

# atlas coordinates in world_tileset.png (16 x 16 px tiles)
THEMES = {
    "grass": dict(top=[(0, 0)], body=[(0, 1), (0, 1), (0, 1), (1, 1)]),
    "sand": dict(top=[(2, 0)], body=[(2, 1), (2, 1), (2, 1), (3, 0)]),
    "ice": dict(top=[(6, 0)], body=[(6, 1), (6, 1), (8, 1)]),
}
DECOR = [(1, 4), (1, 5), (1, 6), (8, 5), (8, 6)]        # small bushes and mushrooms on top of the ground
DECOR_SHARE = 0.12                                       # share of free surface cells that get one

# backdrop: horizontal stripes (sky -> deep water), one atlas row per tile row, as in the original scene
BACKDROP_TOP, BACKDROP_BOTTOM = -9, 27
_STRIPES = [(2, 9), (3, 10), (8, 11), (9, 12), (15, 13), (16, 14), (27, 15)]   # (last tile row, atlas row)


def backdrop_row(ty: int) -> int:
    """Atlas row of the backdrop at tile row ty (rows above / below the stripes repeat the first / last)."""
    ty = min(max(ty, BACKDROP_TOP), BACKDROP_BOTTOM)
    for last, ay in _STRIPES:
        if ty <= last:
            return ay
    return _STRIPES[-1][1]


def background_tiles(tx0: int, tx1: int, ty0: int, ty1: int) -> np.ndarray:
    """Backdrop for tile columns tx0..tx1-1 and rows ty0..ty1-1, as an (n, 4) int array."""
    xs, ys = np.meshgrid(np.arange(tx0, tx1), np.arange(ty0, ty1))
    ay = np.vectorize(backdrop_row)(ys)
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size, int), ay.ravel()], axis=1).astype(np.int32)


def _hash(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random non-negative ints per cell."""
    return ((c.astype(np.int64) * 73856093) ^ (r.astype(np.int64) * 19349663)) & 0x7FFFFFFF


def top_mask(solid: np.ndarray) -> np.ndarray:
    """Solid cells with an empty cell (or the map edge) above them."""
    above = np.zeros_like(solid)
    above[1:] = solid[:-1]
    return solid & ~above


def terrain_tiles(solid: np.ndarray, origin=(0, 0), theme: str = "grass") -> np.ndarray:
    """One tile per solid cell.  origin = tile position of cell (0, 0) in the world."""
    th = THEMES[theme]
    rows, cols = np.nonzero(solid)
    is_top = top_mask(solid)[rows, cols]
    h = _hash(cols, rows)
    top = np.asarray(th["top"])[h % len(th["top"])]
    body = np.asarray(th["body"])[h % len(th["body"])]
    atlas = np.where(is_top[:, None], top, body)
    return np.concatenate([np.stack([cols + origin[0], rows + origin[1]], axis=1), atlas], axis=1).astype(np.int32)


def decor_tiles(solid: np.ndarray, clear: np.ndarray | None = None, origin=(0, 0)) -> np.ndarray:
    """Small bushes / mushrooms standing on top tiles.  `clear` marks cells that must stay empty (coins, spawn...)."""
    H, W = solid.shape
    rows, cols = np.nonzero(top_mask(solid))
    keep = rows > 0                                       # needs a cell above to stand in
    rows, cols = rows[keep], cols[keep]
    free = ~solid[rows - 1, cols]
    if clear is not None:
        free &= ~clear[rows - 1, cols]
    h = _hash(cols * 7 + 3, rows * 5 + 1)
    pick = free & ((h % 1000) < DECOR_SHARE * 1000)
    rows, cols, h = rows[pick], cols[pick], h[pick]
    atlas = np.asarray(DECOR)[(h // 1000) % len(DECOR)] if len(rows) else np.zeros((0, 2), int)
    return np.concatenate([np.stack([cols + origin[0], rows - 1 + origin[1]], axis=1), atlas], axis=1).astype(np.int32)


def goal_runs(goal: np.ndarray) -> list[tuple[int, int, int]]:
    """Vertical runs of goal cells -> (column, top row, number of rows); one flag is drawn per run."""
    out = []
    for c in range(goal.shape[1]):
        r = 0
        while r < goal.shape[0]:
            if goal[r, c]:
                r0 = r
                while r < goal.shape[0] and goal[r, c]:
                    r += 1
                out.append((c, r0, r - r0))
            else:
                r += 1
    return out


def flag_sprite(n_tiles: int) -> np.ndarray:
    """A pixel-art flag, 16 px wide and n_tiles * 16 px tall, as RGBA (pole standing on the bottom edge)."""
    h = max(1, n_tiles) * 16
    img = np.zeros((h, 16, 4), np.uint8)
    pole, pole_dark = (176, 132, 82, 255), (112, 76, 44, 255)
    img[2:h, 4] = pole_dark
    img[2:h, 5] = pole
    img[1, 4:6] = pole                                    # pole top
    cloth, cloth_dark, outline = (228, 64, 64, 255), (176, 36, 52, 255), (74, 20, 36, 255)
    fh = min(9, max(5, h // 3))                           # pennant height
    for i in range(fh):                                   # pennant pointing right, narrowing to the tip
        w = max(2, 10 - (abs(2 * i - (fh - 1)) * 10) // fh)
        y = 2 + i
        img[y, 6:6 + w] = cloth if i < (fh + 1) // 2 else cloth_dark
        img[y, 6 + w - 1] = outline
    img[1, 6:12] = outline                                # outline on top and bottom
    img[2 + fh, 6:11] = outline
    img[h - 1, 3:7] = outline                             # base
    return img
