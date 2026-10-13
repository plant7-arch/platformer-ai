"""The level, as plain ASCII: the same layout as the original Godot level (70 x 10 tiles).

    #  solid tile                 -  one-way platform tile (stand on it; pass through from below/sides)
    o  coin                       G  goal (touch = win)
    P  player spawn               E  enemy spawn (patrols left/right, touch = death)
    S  extra start point (only used by random-start training; same as P otherwise)
    M  moving platform (a one-way platform 2 tiles wide that sweeps right/left, carrying the player)
    .  empty

Columns outside the map are walls; rows above/below the map are open (falling out = death).
Units are tiles (1 tile = 1.0), x grows right, y grows DOWN (row index).
P / E stand on the bottom edge of their tile.  The moving platform starts at the M tiles and
sweeps PLATFORM_TRAVEL tiles to the right, then back.  One extra coin rides on the moving platform.

Not in the original Godot level: the goal flag G (the original had no way to win).
"""
import numpy as np

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
"""

# --- moving platform / enemy / riding coin parameters (tiles, tiles per second) ---
PLATFORM_TRAVEL = 6.0
PLATFORM_SPEED = 3.2         # Godot: 96 px/s = 6 tiles/s, scaled by our player speed (6) / Godot's (11.25)
ENEMY_SPEED = 2.0            # Godot: 60 px/s = 3.75 tiles/s, same scaling
ENEMY_HALF_W, ENEMY_H = 0.375, 0.875      # hit box 12 x 14 px
ENEMY_RAY_LEN, ENEMY_RAY_H = 0.5625, 0.4375   # wall sensors: 9 px sideways, at 7 px height
COIN_RADIUS = 0.3125                      # 5 px
RIDER_COIN_OFFSET = (-0.0625, -0.5)       # relative to (platform centre x, platform top y)


def parse(text: str):
    rows = text.strip("\n").split("\n")
    if len({len(r) for r in rows}) != 1:
        raise ValueError("all level rows must have the same length")
    grid = np.array([list(r) for r in rows])

    def one(ch):
        cells = np.argwhere(grid == ch)
        if len(cells) != 1:
            raise ValueError(f"level needs exactly one '{ch}'")
        return cells[0]

    sr, sc = one("P")
    er, ec = one("E")
    m_cells = np.argwhere(grid == "M")
    if len(m_cells) == 0 or len(set(m_cells[:, 0])) != 1:
        raise ValueError("'M' tiles must be in a single row")
    m_row, m_c0, m_c1 = int(m_cells[0, 0]), int(m_cells[:, 1].min()), int(m_cells[:, 1].max())
    coins = np.argwhere(grid == "o")
    starts = np.argwhere(grid == "S")
    static_coins = np.stack([coins[:, 1] + 0.5, coins[:, 0] + 0.5], axis=1)      # (x, y) cell centres
    return dict(
        solid=grid == "#", oneway=grid == "-", goal=grid == "G",
        spawn=(float(sc) + 0.5, float(sr) + 1.0), enemy=(float(ec) + 0.5, float(er) + 1.0),
        plat_top=float(m_row), plat_width=float(m_c1 - m_c0 + 1), plat_x0=(m_c0 + m_c1 + 1) / 2.0,
        coins=static_coins,
        starts=[(float(c) + 0.5, float(r) + 1.0) for r, c in starts],
    )


_p = parse(LEVEL)
SOLID: np.ndarray = _p["solid"]        # (H, W) bool
ONEWAY: np.ndarray = _p["oneway"]      # (H, W) bool, static one-way tiles
GOAL: np.ndarray = _p["goal"]          # (H, W) bool
H, W = SOLID.shape
SPAWN_X, SPAWN_Y = _p["spawn"]
ENEMY_X, ENEMY_Y = _p["enemy"]
PLAT_TOP: float = _p["plat_top"]       # y of the moving platform's top surface
PLAT_HALF: float = _p["plat_width"] / 2.0
PLAT_X_MIN: float = _p["plat_x0"]      # centre x at the left end of the sweep (= start)
PLAT_X_MAX: float = PLAT_X_MIN + PLATFORM_TRAVEL

# Coins: the static ones from the map, plus one riding the moving platform (last index).
N_COINS = len(_p["coins"]) + 1
COIN_X = np.append(_p["coins"][:, 0], 0.0).astype(np.float32)      # rider's x/y are computed from the platform
COIN_Y = np.append(_p["coins"][:, 1], 0.0).astype(np.float32)
COIN_RIDES = np.zeros(N_COINS, dtype=bool)
COIN_RIDES[-1] = True

# Extra start points ('S'); stand on the bottom edge of their tile like P.
_starts = _p["starts"] or [(SPAWN_X, SPAWN_Y)]
START_X = np.array([p[0] for p in _starts], dtype=np.float32)
START_Y = np.array([p[1] for p in _starts], dtype=np.float32)
N_STARTS = len(_starts)


def _patrol_limits():
    """Walls left/right of the enemy at sensor height -> the x range it patrols."""
    row = int(np.floor(ENEMY_Y - ENEMY_RAY_H))
    c = int(ENEMY_X)
    while c >= 0 and not SOLID[row, c]:
        c -= 1
    left_face = c + 1
    c = int(ENEMY_X)
    while c < W and not SOLID[row, c]:
        c += 1
    return left_face + ENEMY_RAY_LEN, c - ENEMY_RAY_LEN


ENEMY_MIN, ENEMY_MAX = _patrol_limits()
