"""The level, as plain ASCII: the same layout as the original Godot level (70 x 10 tiles).

    #  solid tile                 -  one-way platform tile (stand on it; pass through from below/sides)
    o  coin                       G  goal (touch = win)
    P  player spawn               E  enemy spawn (patrols left/right, touch = death)
    S  extra start point (only used by random-start training; same as P otherwise)
    M  moving platform (a one-way platform that sweeps right/left, carrying the player)
    .  empty

Columns outside the map are walls; rows above/below the map are open (falling out = death).
Units are tiles (1 tile = 1.0), x grows right, y grows DOWN (row index).
P / E stand on the bottom edge of their tile.  There is exactly one P; any number of E (0, 1, 2, ...), and
every E is its own enemy.  Neighbouring M tiles in a row form ONE moving platform (so "MM.MM" is two platforms of
width 2); any number of platforms is fine, also none.  One extra coin rides on every moving platform.
Order inside the arrays below: enemies and platforms are numbered row by row, left to right.

Paths: after the map and a blank line, one optional line per platform / enemy says where it moves (x in tiles, the
centre of the thing; row and column count from 1 like the error messages):

    platform ROW COL to STOP_X          starts where its M tiles are, moves to STOP_X (left or right) and back
    enemy ROW COL from MIN_X to MAX_X   starts at its E and patrols between MIN_X and MAX_X (a wall inside is an error)

Without a line: a platform moves DEFAULT_PLATFORM_TRAVEL tiles to the right, an enemy DEFAULT_ENEMY_REACH tiles each way.
levelcheck.py makes sure every path is inside the map and free of tiles, coins, goal, spawn and other movers.

Not in the original Godot level: the goal flag G (the original had no way to win).
The viewer draws terrain, backdrop and goal flag from this ASCII (terrain.py), so editing the level is enough.
"""
import os

import numpy as np

from platformer_ai import levelcheck

LEVEL = """\
......................................................................
......................................................................
.....................................................................G
.................P.......................................o...........G
.......###.......o.......................ooo..........S.###...S...o.##
......#####.....--....................S.####.........#################
.....######..............S........o.S.######S...E..S.#################
########...........So.#####MM......###################################
#######....################........###################################
...........####....###....#...........................................

# one line per mover: where it goes (x in tiles).  The enemy turns at these x, the platform reverses at its stop x.
platform 8 28 to 34
enemy 7 49 from 44.5625 to 52.4375
"""

# Try another level without editing this file: PLATFORMER_LEVEL=my_level.txt uv run python -m platformer_ai.viewer
# (a text file with the map rows and optional path lines; works for every command, including train and levelcheck).
LEVEL_FILE = os.environ.get("PLATFORMER_LEVEL")
LEVEL_SOURCE = LEVEL_FILE or "the LEVEL in " + os.path.relpath(__file__)
CHECK_HINT = "Check it without starting anything:  uv run python -m platformer_ai.levelcheck" + (f" {LEVEL_FILE}" if LEVEL_FILE else "")
levelcheck.SOURCE = LEVEL_SOURCE                    # named in every level message
if LEVEL_FILE:
    try:
        with open(LEVEL_FILE) as _f:
            LEVEL = _f.read()
    except OSError as _e:
        levelcheck.short_errors_on_exit("PLATFORMER_LEVEL must be the path of a text file with the level.")
        raise levelcheck.LevelError(f"cannot read the level file PLATFORMER_LEVEL={LEVEL_FILE}: {_e.strerror}") from None

# Signs drawn in the world by the viewer (no effect on the game): (text, x, y) = top-left corner in tiles.
LABELS = [("Falling hurts...", 28.9, 7.9), ("Great Job!", 62.3, 0.3)]

# --- moving platform / enemy / riding coin parameters (tiles, tiles per second) ---
DEFAULT_PLATFORM_TRAVEL = 6.0   # a platform without a path line moves this far to the right of its start
DEFAULT_ENEMY_REACH = 3.0       # an enemy without a path line patrols this far either way of its start
PLATFORM_SPEED = 3.2         # Godot: 96 px/s = 6 tiles/s, scaled by our player speed (6) / Godot's (11.25)
ENEMY_SPEED = 2.0            # Godot: 60 px/s = 3.75 tiles/s, same scaling
ENEMY_HALF_W, ENEMY_H = 0.375, 0.875      # hit box 12 x 14 px
COIN_RADIUS = 0.3125                      # 5 px
RIDER_COIN_OFFSET = (-0.0625, -0.5)       # relative to (platform centre x, platform top y)


def parse(text: str):
    """ASCII -> arrays and positions.  Raises levelcheck.LevelError (all problems, with row/col) for an unusable level
    and logs warnings for a suspicious one; see levelcheck.py."""
    levelcheck.validate(text, DEFAULT_PLATFORM_TRAVEL, DEFAULT_ENEMY_REACH, ENEMY_HALF_W, ENEMY_H)
    rows, plats, enemies = levelcheck.layout(text, DEFAULT_PLATFORM_TRAVEL, DEFAULT_ENEMY_REACH)
    grid = np.array([list(r) for r in rows])

    sr, sc = np.argwhere(grid == "P")[0]
    enemies = [(e.x, e.y, e.lo, e.hi) for e in enemies]                       # start x, feet y, patrol range
    plats = [(float(p.row), p.start, p.stop, 2.0 * p.half) for p in plats]   # top y, start x, stop x, width
    coins = np.argwhere(grid == "o")
    starts = np.argwhere(grid == "S")
    static_coins = np.stack([coins[:, 1] + 0.5, coins[:, 0] + 0.5], axis=1)      # (x, y) cell centres
    return dict(
        solid=grid == "#", oneway=grid == "-", goal=grid == "G",
        spawn=(float(sc) + 0.5, float(sr) + 1.0), enemies=enemies, platforms=plats,
        coins=static_coins,
        starts=[(float(c) + 0.5, float(r) + 1.0) for r, c in starts],
    )


f32 = np.float32
try:
    _p = parse(LEVEL)
except levelcheck.LevelError:
    levelcheck.short_errors_on_exit(CHECK_HINT)       # a short message instead of a traceback when it is not caught
    raise
SOLID: np.ndarray = _p["solid"]        # (H, W) bool
ONEWAY: np.ndarray = _p["oneway"]      # (H, W) bool, static one-way tiles
GOAL: np.ndarray = _p["goal"]          # (H, W) bool
H, W = SOLID.shape
SPAWN_X, SPAWN_Y = _p["spawn"]

# Enemies: N_ENEMIES (may be 0).  All arrays have shape (N_ENEMIES,).
N_ENEMIES = len(_p["enemies"])
ENEMY_X = np.array([e[0] for e in _p["enemies"]], dtype=f32)       # start x (centre)
ENEMY_Y = np.array([e[1] for e in _p["enemies"]], dtype=f32)       # constant y of the feet
ENEMY_MIN = np.array([e[2] for e in _p["enemies"]], dtype=f32)     # the enemy turns round at these centre x
ENEMY_MAX = np.array([e[3] for e in _p["enemies"]], dtype=f32)

# Moving platforms: N_PLATS (may be 0).  All arrays have shape (N_PLATS,).
N_PLATS = len(_p["platforms"])
PLAT_TOP = np.array([m[0] for m in _p["platforms"]], dtype=f32)    # y of the top surface
PLAT_START_X = np.array([m[1] for m in _p["platforms"]], dtype=f32)   # centre x at the start (where the M tiles are)
PLAT_STOP_X = np.array([m[2] for m in _p["platforms"]], dtype=f32)    # centre x at the other end
PLAT_HALF = np.array([m[3] / 2.0 for m in _p["platforms"]], dtype=f32)
PLAT_X_MIN = np.minimum(PLAT_START_X, PLAT_STOP_X)                 # the platform reverses at these centre x
PLAT_X_MAX = np.maximum(PLAT_START_X, PLAT_STOP_X)
PLAT_DIR0 = np.where(PLAT_STOP_X >= PLAT_START_X, 1.0, -1.0).astype(f32)   # first direction of travel

# Coins: the static ones from the map, then one riding on each moving platform (in platform order).
N_STATIC_COINS = len(_p["coins"])
N_COINS = N_STATIC_COINS + N_PLATS
COIN_X = np.concatenate([_p["coins"][:, 0], np.zeros(N_PLATS)]).astype(f32)   # riders' x/y come from the platform
COIN_Y = np.concatenate([_p["coins"][:, 1], np.zeros(N_PLATS)]).astype(f32)
COIN_RIDES = np.arange(N_COINS) >= N_STATIC_COINS

# Extra start points ('S'); stand on the bottom edge of their tile like P.
_starts = _p["starts"] or [(SPAWN_X, SPAWN_Y)]
START_X = np.array([p[0] for p in _starts], dtype=f32)
START_Y = np.array([p[1] for p in _starts], dtype=f32)
N_STARTS = len(_starts)
