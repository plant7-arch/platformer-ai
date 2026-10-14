"""Terminal rendering of a single environment state (no dependencies).
Legend: # solid  - one-way platform  M moving platform  o coin  X enemy  G goal  @ player"""
import numpy as np

from platformer_ai import env as E
from platformer_ai import level as L


def frame(state) -> str:
    grid = np.where(L.SOLID, "#", ".")
    grid[L.ONEWAY] = "-"
    grid[L.GOAL] = "G"
    for px, top, half in zip(np.asarray(state.plat_x), L.PLAT_TOP, L.PLAT_HALF):
        left = int(np.floor(float(px) - half + 0.5))
        for c in range(left, left + int(2 * half)):
            if 0 <= c < L.W:
                grid[int(top), c] = "M"
    cx, cy = (np.asarray(a) for a in E._coin_pos(state.plat_x))
    for i in np.flatnonzero(np.asarray(state.coins_alive)):
        r, c = int(np.floor(cy[i])), int(np.floor(cx[i]))
        if 0 <= r < L.H and 0 <= c < L.W:
            grid[r, c] = "o"
    for ex, ey in zip(np.asarray(state.enemy_x), L.ENEMY_Y):
        er, ec = int(np.floor(float(ey) - 0.4)), int(np.floor(float(ex)))
        if 0 <= ec < L.W:
            grid[er, ec] = "X"
    r = int(np.floor(float(state.y) - E.HEIGHT / 2))
    c = int(np.floor(float(state.x)))
    if 0 <= r < L.H and 0 <= c < L.W:
        grid[r, c] = "@"
    return "\n".join("".join(row) for row in grid)
