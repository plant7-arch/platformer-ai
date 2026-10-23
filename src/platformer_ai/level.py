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
width 2); any number of platforms is fine, also none.
Riders: an E standing right on an M tile walks on that moving platform (bounded by its ends, carried along by it); an E on
a run of '-' tiles patrols only that run.  An 'o' right above an M tile sticks to the platform and moves with it (a platform has a coin only if you draw one).
Order inside the arrays below: enemies and platforms are numbered row by row, left to right.

Paths: after the map and a blank line, one optional line per platform / enemy says where it moves (x in tiles, the
centre of the thing; row and column count from 1 like the error messages):

    platform ROW COL [to STOP_X]   starts where its M tiles are, moves to STOP_X (left or right) and back
    enemy ROW COL from MIN_X to MAX_X        starts at its E and patrols between MIN_X and MAX_X (a wall inside is an
                                             error).  A rider's MIN_X / MAX_X are x on its platform at the platform's start.

Without a line: a platform moves DEFAULT_PLATFORM_TRAVEL tiles to the right, an enemy DEFAULT_ENEMY_REACH tiles each way.
levelcheck.py makes sure every path is inside the map and free of tiles, coins, goal, spawn and other movers.

Not in the original Godot level: the goal flag G (the original had no way to win).
The viewer draws terrain, backdrop and goal flag from this ASCII (terrain.py), so editing the level is enough.
"""
import os
from dataclasses import dataclass

import numpy as np

from platformer_ai import levelcheck

LEVEL = """\
......................................................................
......................................................................
.....................................................................G
.................P.......................................o...........G
.......###.......o.......................ooo..........S.###...S...o.##
......#####.....--....................S.####.........#################
.....######..............S.o......o.S.######S...E..S.#################
########...........So.#####MM......###################################
#######....################........###################################
...........####....###....#...........................................

# one line per mover: where it goes (x in tiles).  The enemy turns at these x, the platform reverses at its stop x.
platform 8 28 to 34
enemy 7 49 from 44.5625 to 52.4375
"""

BUILTIN_LEVEL = LEVEL      # the level above, also when PLATFORMER_LEVEL picks another one

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

# Signs drawn in the world by the viewer (no effect on the game): (text, x, y) = top-left corner in tiles.  They belong to
# the map of the built-in level, so the viewer only draws them when the level has that map (IS_BUILTIN, below).
LABELS = [("Falling hurts...", 28.9, 7.9), ("Great Job!", 62.3, 0.3)]

# --- moving platform / enemy / riding coin parameters (tiles, tiles per second) ---
DEFAULT_PLATFORM_TRAVEL = 6.0   # a platform without a path line moves this far to the right of its start
DEFAULT_ENEMY_REACH = 3.0       # an enemy without a path line patrols this far either way of its start
PLATFORM_SPEED = 3.2         # Godot: 96 px/s = 6 tiles/s, scaled by our player speed (6) / Godot's (11.25)
ENEMY_SPEED = 2.0            # Godot: 60 px/s = 3.75 tiles/s, same scaling
ENEMY_HALF_W, ENEMY_H = 0.375, 0.875      # hit box 12 x 14 px
COIN_RADIUS = 0.3125                      # 5 px


def parse(text: str, source: str | None = None):
    """ASCII -> arrays and positions.  Raises levelcheck.LevelError (all problems, with row/col) for an unusable level
    and logs warnings for a suspicious one; see levelcheck.py.  source: file name for the messages."""
    levelcheck.validate(text, DEFAULT_PLATFORM_TRAVEL, DEFAULT_ENEMY_REACH, ENEMY_HALF_W, ENEMY_H, source=source)
    rows, plats, enemies, sticky = levelcheck.layout(text, DEFAULT_PLATFORM_TRAVEL, DEFAULT_ENEMY_REACH, ENEMY_HALF_W)
    grid = np.array([list(r) for r in rows])

    sr, sc = np.argwhere(grid == "P")[0]
    # start x, feet y, patrol limits (a rider's limits are relative to the centre of its platform), platform index or -1
    enemy_rows = [(e.x, e.y, e.lo - (plats[e.ride].start if e.ride >= 0 else 0.0),
                   e.hi - (plats[e.ride].start if e.ride >= 0 else 0.0), e.ride) for e in enemies]
    sticky_cells = {(r, c) for r, c, _ in sticky}
    coins = [(int(r), int(c)) for r, c in np.argwhere(grid == "o") if (int(r), int(c)) not in sticky_cells]
    stuck = [(c + 0.5 - plats[i].start, r + 0.5, i) for r, c, i in sticky]    # x relative to the platform centre, y, platform
    platform_rows = [(float(p.row), p.start, p.stop, 2.0 * p.half) for p in plats]
    starts = np.argwhere(grid == "S")
    static_coins = np.array([(c + 0.5, r + 0.5) for r, c in coins], dtype=float).reshape(-1, 2)   # (x, y) cell centres
    return dict(
        solid=grid == "#", oneway=grid == "-", goal=grid == "G",
        spawn=(float(sc) + 0.5, float(sr) + 1.0), enemies=enemy_rows, platforms=platform_rows,
        coins=static_coins, stuck_coins=stuck,
        starts=[(float(c) + 0.5, float(r) + 1.0) for r, c in starts],
    )


f32 = np.float32


@dataclass(frozen=True)
class Arrays:
    """One level as numpy arrays (its own sizes).  env.make_levels() pads several of them to one shape for the GPU."""
    name: str
    SOLID: np.ndarray        # (H, W) bool
    ONEWAY: np.ndarray       # (H, W) bool, static one-way tiles
    GOAL: np.ndarray         # (H, W) bool
    SPAWN_X: float
    SPAWN_Y: float
    # enemies, shape (N_ENEMIES,)
    ENEMY_X: np.ndarray      # start x (centre)
    ENEMY_Y: np.ndarray      # constant y of the feet
    ENEMY_MIN: np.ndarray    # the enemy turns round at these centre x ...
    ENEMY_MAX: np.ndarray    # ... (for a rider: relative to its platform's centre)
    ENEMY_RIDE: np.ndarray   # index of the moving platform it walks on, or -1
    # moving platforms, shape (N_PLATS,)
    PLAT_TOP: np.ndarray     # y of the top surface
    PLAT_START_X: np.ndarray     # centre x at the start (where the M tiles are)
    PLAT_STOP_X: np.ndarray      # centre x at the other end
    PLAT_HALF: np.ndarray
    # coins, shape (N_COINS,): the static ones, then the ones stuck to a moving platform
    N_STATIC_COINS: int
    COIN_X: np.ndarray       # a stuck coin's x is relative to its platform's centre
    COIN_Y: np.ndarray
    COIN_PLAT: np.ndarray    # platform index or -1
    # start points ('S'; the spawn if there are none)
    START_X: np.ndarray
    START_Y: np.ndarray

    H = property(lambda s: s.SOLID.shape[0])
    W = property(lambda s: s.SOLID.shape[1])
    N_ENEMIES = property(lambda s: len(s.ENEMY_X))
    N_PLATS = property(lambda s: len(s.PLAT_TOP))
    N_COINS = property(lambda s: len(s.COIN_X))
    N_STICKY_COINS = property(lambda s: len(s.COIN_X) - s.N_STATIC_COINS)
    N_STARTS = property(lambda s: len(s.START_X))
    PLAT_X_MIN = property(lambda s: np.minimum(s.PLAT_START_X, s.PLAT_STOP_X))   # the platform reverses at these x
    PLAT_X_MAX = property(lambda s: np.maximum(s.PLAT_START_X, s.PLAT_STOP_X))
    PLAT_DIR0 = property(lambda s: np.where(s.PLAT_STOP_X >= s.PLAT_START_X, 1.0, -1.0).astype(f32))  # first direction
    COIN_RIDES = property(lambda s: s.COIN_PLAT >= 0)


def build(text: str, name: str = "level") -> Arrays:
    """Level text -> Arrays.  Raises levelcheck.LevelError for an unusable level (name = file name in the messages)."""
    p = parse(text, source=name)
    stuck = p["stuck_coins"]
    n_static = len(p["coins"])
    starts = p["starts"] or [p["spawn"]]
    col = lambda rows, i, dtype=f32: np.array([r[i] for r in rows], dtype=dtype)     # noqa: E731
    return Arrays(
        name=name, SOLID=p["solid"], ONEWAY=p["oneway"], GOAL=p["goal"],
        SPAWN_X=p["spawn"][0], SPAWN_Y=p["spawn"][1],
        ENEMY_X=col(p["enemies"], 0), ENEMY_Y=col(p["enemies"], 1), ENEMY_MIN=col(p["enemies"], 2),
        ENEMY_MAX=col(p["enemies"], 3), ENEMY_RIDE=col(p["enemies"], 4, np.int32),
        PLAT_TOP=col(p["platforms"], 0), PLAT_START_X=col(p["platforms"], 1), PLAT_STOP_X=col(p["platforms"], 2),
        PLAT_HALF=np.array([m[3] / 2.0 for m in p["platforms"]], dtype=f32),
        N_STATIC_COINS=n_static,
        COIN_X=np.concatenate([p["coins"][:, 0], [c[0] for c in stuck]]).astype(f32),
        COIN_Y=np.concatenate([p["coins"][:, 1], [c[1] for c in stuck]]).astype(f32),
        COIN_PLAT=np.concatenate([-np.ones(n_static), [c[2] for c in stuck]]).astype(np.int32),
        START_X=col(starts, 0), START_Y=col(starts, 1),
    )


def load(path: str) -> Arrays:
    """A level file -> Arrays (LevelError with the file name if it cannot be read or is not a valid level)."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        what = "no such level file or folder" if not os.path.exists(path) else e.strerror
        raise levelcheck.LevelError(f"cannot read the level file {path}: {what}") from None
    return build(text, name=path)


def load_all(paths) -> list[Arrays]:
    """Level files and folders (a folder = all its *.txt files) -> [Arrays].  Every problem of every file at once."""
    out, problems = [], []
    for path in levelcheck.level_files(paths):
        try:
            out.append(load(path))
        except levelcheck.LevelError as e:
            problems.append(str(e))
    if problems:
        raise levelcheck.LevelError("\n".join(problems))
    return out


# ---- THE level (built-in, or PLATFORMER_LEVEL): the one the viewer, play and the env defaults use -------------------
try:
    CURRENT = build(LEVEL, name=LEVEL_SOURCE)
except levelcheck.LevelError:
    levelcheck.short_errors_on_exit(CHECK_HINT)       # a short message instead of a traceback when it is not caught
    raise


def same_map(text_a: str, text_b: str) -> bool:
    """Do two level texts have the same map rows (path lines and comments aside)?"""
    return levelcheck.split_text(text_a)[0] == levelcheck.split_text(text_b)[0]


IS_BUILTIN = same_map(LEVEL, BUILTIN_LEVEL)        # the built-in map (level.py, or a copy such as docs/builtin.txt)
SOLID: np.ndarray = CURRENT.SOLID        # (H, W) bool
ONEWAY: np.ndarray = CURRENT.ONEWAY      # (H, W) bool, static one-way tiles
GOAL: np.ndarray = CURRENT.GOAL          # (H, W) bool
H, W = SOLID.shape
SPAWN_X, SPAWN_Y = CURRENT.SPAWN_X, CURRENT.SPAWN_Y

# Enemies: N_ENEMIES (may be 0).  All arrays have shape (N_ENEMIES,).
N_ENEMIES = CURRENT.N_ENEMIES
ENEMY_X, ENEMY_Y = CURRENT.ENEMY_X, CURRENT.ENEMY_Y
ENEMY_MIN, ENEMY_MAX = CURRENT.ENEMY_MIN, CURRENT.ENEMY_MAX      # riders: relative to their platform's centre
ENEMY_RIDE = CURRENT.ENEMY_RIDE

# Moving platforms: N_PLATS (may be 0).  All arrays have shape (N_PLATS,).
N_PLATS = CURRENT.N_PLATS
PLAT_TOP, PLAT_START_X, PLAT_STOP_X, PLAT_HALF = CURRENT.PLAT_TOP, CURRENT.PLAT_START_X, CURRENT.PLAT_STOP_X, CURRENT.PLAT_HALF
PLAT_X_MIN, PLAT_X_MAX, PLAT_DIR0 = CURRENT.PLAT_X_MIN, CURRENT.PLAT_X_MAX, CURRENT.PLAT_DIR0

# Coins: the static ones from the map, then the ones stuck on a moving platform (an 'o' right above an 'M' tile)
# (only coins drawn in the map exist; a platform has none of its own).
# COIN_X / COIN_Y: position of a static coin; for one riding a platform COIN_X is relative to the platform centre.
N_STATIC_COINS, N_STICKY_COINS, N_COINS = CURRENT.N_STATIC_COINS, CURRENT.N_STICKY_COINS, CURRENT.N_COINS
COIN_X, COIN_Y, COIN_PLAT, COIN_RIDES = CURRENT.COIN_X, CURRENT.COIN_Y, CURRENT.COIN_PLAT, CURRENT.COIN_RIDES

# Extra start points ('S'); stand on the bottom edge of their tile like P.  Without any: the spawn.
START_X, START_Y, N_STARTS = CURRENT.START_X, CURRENT.START_Y, CURRENT.N_STARTS
