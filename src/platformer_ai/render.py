"""Terminal rendering of a single environment state (no dependencies).
Legend: # solid  - one-way platform  M moving platform  o coin  X enemy  G goal  @ player"""
import numpy as np

from platformer_ai import env as E
from platformer_ai import level as L


def frame(state) -> str:
    grid = np.where(L.SOLID, "#", ".")
    grid[L.ONEWAY] = "-"
    grid[L.GOAL] = "G"
    left = int(np.floor(float(state.plat_x) - L.PLAT_HALF + 0.5))
    for c in range(left, left + int(2 * L.PLAT_HALF)):
        if 0 <= c < L.W:
            grid[int(L.PLAT_TOP), c] = "M"
    cx, cy = (np.asarray(a) for a in E._coin_pos(state.plat_x))
    for i in np.flatnonzero(np.asarray(state.coins_alive)):
        r, c = int(np.floor(cy[i])), int(np.floor(cx[i]))
        if 0 <= r < L.H and 0 <= c < L.W:
            grid[r, c] = "o"
    er, ec = int(np.floor(L.ENEMY_Y - 0.4)), int(np.floor(float(state.enemy_x)))
    if 0 <= ec < L.W:
        grid[er, ec] = "X"
    r = int(np.floor(float(state.y) - E.HEIGHT / 2))
    c = int(np.floor(float(state.x)))
    if 0 <= r < L.H and 0 <= c < L.W:
        grid[r, c] = "@"
    return "\n".join("".join(row) for row in grid)
