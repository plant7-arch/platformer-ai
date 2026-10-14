"""Minimal pure-JAX platformer.  One environment = pure functions on a small pytree;
use jax.vmap for thousands of them.  Nothing here touches the CPU.

    state = reset(key)                                        # the real spawn
    state, obs, reward, done, info = step(state, action)     # auto-resets when done

The level is an INPUT (data, not code): every function takes `levels=` (from make_levels; default: the level of
level.py / PLATFORMER_LEVEL).  Several levels of different sizes are padded to one shape, every environment carries the
index of the level it plays (state.level), and an auto-reset picks a random level, so one batch trains on all of them:

    levels = make_levels(level.load_all(["levels/"]))
    state = reset(key, levels=levels)                         # a random level (or level=i)
    state, obs, reward, done, info = step(state, action, levels=levels)

Training helpers (all off by default, switched on in ppo.Config):
    random_phases=True    platform and enemy start at random phases, spawn position is jittered
    random_starts=True    some episodes restart at random points along the level (see reset)
    progress_coef=0.1     small bonus per tile of NEW furthest-right progress (not farmable)

Actions (6): horizontal = [none, left, right][a % 3],  jump = a >= 3.
Rules: solid tiles, one-way platforms (static tiles + any number of moving platforms that carry the player),
any number of patrolling enemies (touch = death), coins (also ones stuck to a moving platform), a goal, pits.
A level may have 0, 1 or many enemies and moving platforms; the state arrays have the length of the largest count
among the levels (unused slots are switched off by masks in the level data).
Reward: +1 per coin, +5 goal (terminates), -1 death (pit or enemy, terminates).
Episode is truncated after MAX_STEPS (info["truncated"]); `done` = terminated | truncated.
Observation: egocentric 9 x 17 tile window with 5 channels + 13 scalars (see observe()).  The window shows every
enemy and platform; the scalars describe the NEAREST enemy and the NEAREST platform (fixed size for any level).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax

from platformer_ai import level as L

# ---- physics (tiles, seconds) -------------------------------------------------------------
DT = 1.0 / 30.0          # one env step = 1/30 s
GRAVITY = 40.0
JUMP_SPEED = 14.0        # apex ~2.7 tiles: the static one-way platform (3 tiles up) is out of reach
RUN_SPEED = 6.0
MAX_FALL = 18.0          # must stay < 1 tile per step (18 * DT = 0.6) so nothing tunnels
HW = 0.3                 # player half width   (AABB: width 0.6)
HEIGHT = 0.8             # player height, < 1 tile so at most 2 tile rows/cols are touched
EPS = 1e-4

MAX_STEPS = 900
N_ACTIONS = 6
R_COIN, R_GOAL, R_DEATH = 1.0, 5.0, -1.0
SPAWN_PROB = 0.5          # random_starts: share of episodes that start at the (jittered) real spawn
SPAWN_JITTER_X, SPAWN_JITTER_Y = 0.5, 0.6     # random_phases: spawn x +-0.5, dropped from up to 0.6 higher

# ---- observation window: rows -PR..+PR and cols -PC..+PC around the player's centre tile ---
PR, PC = 4, 8
PATCH = (2 * PR + 1, 2 * PC + 1)
N_CHANNELS = 5           # solid, one-way (static + moving platform), coin, goal, enemy
N_SCALARS = 13
OBS_DIM = N_CHANNELS * PATCH[0] * PATCH[1] + N_SCALARS      # the same for every level
PLAT_THICK = 0.5         # only used to draw the moving platform into the observation
ABOVE = 4                # extra empty rows above the padded grids: a jump (apex 2.7 tiles) above the map's top row
                         # still sees sky there (a negative slice start would wrap round to the bottom in JAX)
FAR = -1.0e4             # where unused (padding) enemies, platforms and coins sit: far above every map


# ---- the level as data ----------------------------------------------------------------------
@struct.dataclass
class Level:
    """One or more levels as arrays, padded to the largest size / count.  make_levels() builds it; every array has a
    leading level axis (n_levels, ...).  *_on masks say which enemy / platform / coin slots exist in each level."""
    h: jax.Array             # map height / width (int32)
    w: jax.Array
    solid: jax.Array         # (Hmax, Wmax) bool, False outside the map
    oneway: jax.Array
    goal: jax.Array
    solid_pad: jax.Array     # the three grids padded for the observation window (side walls are solid)
    oneway_pad: jax.Array
    goal_pad: jax.Array
    spawn_x: jax.Array
    spawn_y: jax.Array
    enemy_on: jax.Array      # (Nmax,) bool
    enemy_x0: jax.Array      # start x
    enemy_y: jax.Array       # feet y (constant)
    enemy_min: jax.Array     # turning points (a rider's are relative to its platform's centre)
    enemy_max: jax.Array
    enemy_ride: jax.Array    # platform index or -1
    plat_on: jax.Array       # (Kmax,) bool
    plat_top: jax.Array
    plat_half: jax.Array
    plat_x_min: jax.Array    # reverses at these centre x
    plat_x_max: jax.Array
    plat_start_x: jax.Array
    plat_dir0: jax.Array
    n_plats: jax.Array       # number of real platforms (int32)
    coin_on: jax.Array       # (Cmax,) bool
    coin_x: jax.Array        # a coin stuck to a platform: x relative to the platform's centre
    coin_y: jax.Array
    coin_plat: jax.Array     # platform index or -1
    start_x: jax.Array       # (Smax,) start points ('S', or the spawn if a level has none)
    start_y: jax.Array
    n_starts: jax.Array


def _pad(a, n, fill, dtype):
    a = np.asarray(a, dtype=dtype)
    return np.concatenate([a, np.full(n - len(a), fill, dtype=dtype)])


def make_levels(levels) -> Level:
    """[level.Arrays] -> Level: all of them padded to the largest map and the largest numbers of enemies, platforms,
    coins and start points (masks mark the real ones).  The observation is the same size for every level."""
    levels = list(levels)
    if not levels:
        raise ValueError("make_levels needs at least one level")
    hm, wm = max(a.H for a in levels), max(a.W for a in levels)
    nm, km = max(a.N_ENEMIES for a in levels), max(a.N_PLATS for a in levels)
    cm, sm = max(a.N_COINS for a in levels), max(a.N_STARTS for a in levels)
    f32, i32 = np.float32, np.int32

    def one(a):
        def grid(g, wall=False):
            out = np.zeros((hm, wm), bool)
            out[:a.H, :a.W] = g
            p = np.pad(out, ((PR + ABOVE, PR), (PC, PC)))
            if wall:                    # columns outside the map are solid walls, at every height
                p[:, :PC] = True
                p[:, PC + a.W:] = True
            return out, p
        solid, solid_pad = grid(a.SOLID, wall=True)
        oneway, oneway_pad = grid(a.ONEWAY)
        goal, goal_pad = grid(a.GOAL)
        on = lambda n, m: np.arange(m) < n          # noqa: E731
        return dict(
            h=i32(a.H), w=i32(a.W), solid=solid, oneway=oneway, goal=goal,
            solid_pad=solid_pad, oneway_pad=oneway_pad, goal_pad=goal_pad,
            spawn_x=f32(a.SPAWN_X), spawn_y=f32(a.SPAWN_Y),
            enemy_on=on(a.N_ENEMIES, nm), enemy_x0=_pad(a.ENEMY_X, nm, 0.0, f32), enemy_y=_pad(a.ENEMY_Y, nm, FAR, f32),
            enemy_min=_pad(a.ENEMY_MIN, nm, -1.0, f32), enemy_max=_pad(a.ENEMY_MAX, nm, 1.0, f32),
            enemy_ride=_pad(a.ENEMY_RIDE, nm, -1, i32),
            plat_on=on(a.N_PLATS, km), plat_top=_pad(a.PLAT_TOP, km, FAR, f32), plat_half=_pad(a.PLAT_HALF, km, 0.0, f32),
            plat_x_min=_pad(a.PLAT_X_MIN, km, 0.0, f32), plat_x_max=_pad(a.PLAT_X_MAX, km, 0.0, f32),
            plat_start_x=_pad(a.PLAT_START_X, km, 0.0, f32), plat_dir0=_pad(a.PLAT_DIR0, km, 1.0, f32),
            n_plats=i32(a.N_PLATS),
            coin_on=on(a.N_COINS, cm), coin_x=_pad(a.COIN_X, cm, FAR, f32), coin_y=_pad(a.COIN_Y, cm, FAR, f32),
            coin_plat=_pad(a.COIN_PLAT, cm, -1, i32),
            start_x=_pad(a.START_X, sm, a.START_X[0], f32), start_y=_pad(a.START_Y, sm, a.START_Y[0], f32),
            n_starts=i32(a.N_STARTS))

    rows = [one(a) for a in levels]
    return Level(**{k: jnp.asarray(np.stack([r[k] for r in rows])) for k in rows[0]})


def n_levels(levels: Level) -> int:
    return int(levels.h.shape[0])


def _pick(levels: Level, i) -> Level:
    """The arrays of level i (i may be traced: every environment of a vmapped batch plays its own level)."""
    return jax.tree_util.tree_map(lambda a: a[i], levels)


# The level of level.py (or PLATFORMER_LEVEL): the default everywhere, so code that never mentions levels still works.
DEFAULT_LEVELS = make_levels([L.CURRENT])
H, W, N_COINS, N_ENEMIES, N_PLATS = L.H, L.W, L.N_COINS, L.N_ENEMIES, L.N_PLATS      # of that level


def _levels(levels):
    return DEFAULT_LEVELS if levels is None else levels


def _default_lv(lv):
    return _pick(DEFAULT_LEVELS, 0) if lv is None else lv


@struct.dataclass
class EnvState:
    x: jax.Array            # centre x of the player
    y: jax.Array            # y of the player's FEET (bottom edge)
    vx: jax.Array
    vy: jax.Array
    coins_alive: jax.Array  # (Cmax,) bool; padding slots start dead
    plat_x: jax.Array       # (Kmax,) moving platform centre x
    plat_dir: jax.Array     # (Kmax,) +1 / -1
    enemy_x: jax.Array      # (Nmax,) enemy centre x (their y is constant)
    enemy_dir: jax.Array    # (Nmax,) +1 / -1
    t: jax.Array            # steps since reset
    furthest: jax.Array     # furthest x reached this episode (progress bonus)
    from_spawn: jax.Array   # bool: episode started at the real spawn (logging)
    rng: jax.Array          # PRNG key stream for random restarts
    ep_return: jax.Array    # task return (coins + goal + death), without shaping; logging only
    ep_coins: jax.Array     # logging only
    level: jax.Array = 0    # index of the level this environment plays (into `levels`)


def initial_state(rng=None, levels=None, level=0) -> EnvState:
    f32 = jnp.float32
    lv = _pick(_levels(levels), level)
    return EnvState(
        x=jnp.asarray(lv.spawn_x, f32), y=jnp.asarray(lv.spawn_y, f32), vx=f32(0), vy=f32(0),
        coins_alive=lv.coin_on,
        plat_x=lv.plat_start_x, plat_dir=lv.plat_dir0, enemy_x=lv.enemy_x0,
        enemy_dir=jnp.ones(lv.enemy_x0.shape[0], f32),
        t=jnp.int32(0), furthest=jnp.asarray(lv.spawn_x, f32), from_spawn=jnp.bool_(True),
        rng=jax.random.PRNGKey(0) if rng is None else rng,
        ep_return=f32(0), ep_coins=f32(0), level=jnp.asarray(level, jnp.int32))


def _sample_start(key, lv: Level, level, use_s_points: bool, random_phases: bool, spawn_prob: float) -> EnvState:
    """Random episode start (lv = the arrays of `level`).
    * random_phases: platforms and enemies begin at random phases and the spawn position is jittered, so
      the agent has to react to what it sees instead of replaying one timed route.
    * use_s_points: with probability 1 - spawn_prob start at a random 'S' point or on a moving platform
      instead of the spawn (always with random phases) to practise the hard parts directly."""
    f32, i32 = jnp.float32, jnp.int32
    km, nm = lv.plat_top.shape[0], lv.enemy_x0.shape[0]
    k = jax.random.split(key, 10)
    plat_x = jax.random.uniform(k[2], (km,), minval=lv.plat_x_min, maxval=lv.plat_x_max)
    plat_dir = jnp.where(jax.random.uniform(k[3], (km,)) < 0.5, 1.0, -1.0)
    base = _enemy_base(plat_x, lv)                     # riders: measured from their platform at its random position
    ex = base + jax.random.uniform(k[4], (nm,), minval=lv.enemy_min + 0.1, maxval=lv.enemy_max - 0.1)
    e_dir = jnp.where(jax.random.uniform(k[5], (nm,)) < 0.5, 1.0, -1.0)
    jx = jax.random.uniform(k[7], minval=-SPAWN_JITTER_X, maxval=SPAWN_JITTER_X)
    jy = jax.random.uniform(k[8], minval=-SPAWN_JITTER_Y, maxval=0.0)
    spawn_x = lv.spawn_x + (jx if random_phases else 0.0)
    spawn_y = lv.spawn_y + (jy if random_phases else 0.0)

    if use_s_points:
        use_spawn = jax.random.uniform(k[0]) < spawn_prob
        # options: the S points, plus (if the level has moving platforms) riding one of them, picked at random
        idx = jax.random.randint(k[1], (), 0, lv.n_starts + (lv.n_plats > 0).astype(i32))
        i = jnp.minimum(idx, lv.n_starts - 1)
        sx = jnp.where(use_spawn, spawn_x, lv.start_x[i])
        sy = jnp.where(use_spawn, spawn_y, lv.start_y[i])
        if km:
            on_ferry = idx == lv.n_starts
            p = jax.random.randint(k[9], (), 0, jnp.maximum(lv.n_plats, 1))
            sx = jnp.where(use_spawn, spawn_x, jnp.where(on_ferry, plat_x[p], lv.start_x[i]))
            sy = jnp.where(use_spawn, spawn_y, jnp.where(on_ferry, lv.plat_top[p], lv.start_y[i]))
        phases = jnp.logical_or(~use_spawn, random_phases)
    else:
        use_spawn = jnp.bool_(True)
        sx, sy = jnp.asarray(spawn_x, f32), jnp.asarray(spawn_y, f32)
        phases = jnp.bool_(random_phases)

    # never start on top of an enemy: push it 3 tiles away (to whichever side still fits its patrol range)
    away = jnp.where(sx + 3.0 <= base + lv.enemy_max, sx + 3.0, sx - 3.0)
    away = jnp.clip(away, base + lv.enemy_min + 0.1, base + lv.enemy_max - 0.1)
    ex = jnp.where(jnp.abs(ex - sx) < 1.5, away, ex)
    init = EnvState(
        x=jnp.asarray(lv.spawn_x, f32), y=jnp.asarray(lv.spawn_y, f32), vx=f32(0), vy=f32(0),
        coins_alive=lv.coin_on, plat_x=lv.plat_start_x, plat_dir=lv.plat_dir0, enemy_x=lv.enemy_x0,
        enemy_dir=jnp.ones(nm, f32), t=jnp.int32(0), furthest=jnp.asarray(lv.spawn_x, f32),
        from_spawn=jnp.bool_(True), rng=k[6], ep_return=f32(0), ep_coins=f32(0), level=jnp.asarray(level, i32))
    return init.replace(
        x=jnp.asarray(sx, f32), y=jnp.asarray(sy, f32), furthest=jnp.asarray(sx, f32),
        plat_x=jnp.where(phases, plat_x, init.plat_x).astype(f32),
        plat_dir=jnp.where(phases, plat_dir, init.plat_dir).astype(f32),
        enemy_x=jnp.where(phases, ex, init.enemy_x).astype(f32),
        enemy_dir=jnp.where(phases, e_dir, init.enemy_dir).astype(f32),
        from_spawn=use_spawn)


def reset(key=None, random_start: bool = False, random_phases: bool = False,
          spawn_prob: float = SPAWN_PROB, levels=None, level=None) -> EnvState:
    """The exact default spawn, unless a randomisation option is on (see _sample_start).
    level: which level (an index into `levels`); None = a random one (with a single level: that one)."""
    if key is None:
        key = jax.random.PRNGKey(0)
    levels = _levels(levels)
    if level is None:
        if n_levels(levels) == 1:
            level = 0
        else:
            key, k_level = jax.random.split(key)
            level = jax.random.randint(k_level, (), 0, n_levels(levels))
    if random_start or random_phases:
        return _sample_start(key, _pick(levels, level), level, random_start, random_phases, spawn_prob)
    return initial_state(rng=key, levels=levels, level=level)


# ---- tile queries (lv = the arrays of one level) --------------------------------------------
def _in_map(r, c, lv):
    return (r >= 0) & (r < lv.h) & (c >= 0) & (c < lv.w)


def _grid_at(grid, r, c, lv):
    return grid[jnp.clip(r, 0, grid.shape[0] - 1), jnp.clip(c, 0, grid.shape[1] - 1)] & _in_map(r, c, lv)


def _solid(r, c, lv):
    """Solid tile?  Side walls outside the map; open sky above and open pit below."""
    inside_c = (c >= 0) & (c < lv.w)
    return jnp.where(inside_c, _grid_at(lv.solid, r, c, lv), True)


def _oneway(r, c, lv):
    return _grid_at(lv.oneway, r, c, lv)


def _tile_support(x, y, lv=None):
    """Standing on a solid tile or on top of a one-way tile?  (lv: default = the level of level.py)"""
    lv = _default_lv(lv)
    row = jnp.floor(y + 0.01).astype(jnp.int32)
    c_l = jnp.floor(x - HW + EPS).astype(jnp.int32)
    c_r = jnp.floor(x + HW - EPS).astype(jnp.int32)
    solid = _solid(row, c_l, lv) | _solid(row, c_r, lv)
    oneway = (_oneway(row, c_l, lv) | _oneway(row, c_r, lv)) & (y - row < 0.02)
    return solid | oneway


def _on_mplat(x, y, plat_x, lv=None):
    """(Kmax,) bool: standing on each moving platform?"""
    lv = _default_lv(lv)
    return ((jnp.abs(y - lv.plat_top) <= 0.02) &
            (x + HW > plat_x - lv.plat_half) & (x - HW < plat_x + lv.plat_half) & lv.plat_on)


def _overlap(x, y, shape=None):
    """(Hmax, Wmax) bool: tiles overlapped by the player's box (default shape: the level of level.py)."""
    shape = (H, W) if shape is None else shape
    rows, cols = jnp.arange(shape[0])[:, None], jnp.arange(shape[1])[None, :]
    return ((cols + 1 > x - HW) & (cols < x + HW) &
            (rows + 1 > y - HEIGHT) & (rows < y))


def _coin_pos(plat_x, lv=None):
    """(Cmax,) x and y of every coin: static ones where they are, riding ones moving with their platform."""
    lv = _default_lv(lv)
    if lv.plat_top.shape[0] == 0:
        return lv.coin_x, lv.coin_y
    return lv.coin_x + jnp.where(lv.coin_plat >= 0, plat_x[jnp.maximum(lv.coin_plat, 0)], 0.0), lv.coin_y


def _enemy_base(plat_x, lv):
    """(Nmax,) x the enemy limits are measured from: the centre of the platform it walks on, else 0."""
    if lv.plat_top.shape[0] == 0:
        return jnp.zeros(lv.enemy_x0.shape[0], jnp.float32)
    return jnp.where(lv.enemy_ride >= 0, plat_x[jnp.maximum(lv.enemy_ride, 0)], 0.0)


def _first(mask, values):
    """values[i] for the first True in mask, else 0 (empty-safe: no argmax on a zero-length array)."""
    first = mask & (jnp.cumsum(mask.astype(jnp.int32)) == 1)
    return jnp.sum(jnp.where(first, values, 0.0))


NONE_NEAR = (2.0, 0.0, 0.0)      # the nearest-thing scalars when there is none: far away to the right, standing still


def _nearest(px, py, xs, ys, dirs, on, x_scale, y_scale):
    """The 3 observation scalars for the nearest of several things (only those switched `on`): dx, dy (scaled,
    clipped) and its direction."""
    none = jnp.asarray(NONE_NEAR, jnp.float32)
    if xs.shape[0] == 0:
        return none
    dx, dy = xs - px, ys - py
    i = jnp.argmin(jnp.where(on, dx * dx + dy * dy, jnp.inf))
    near = jnp.stack([jnp.clip(dx[i] / x_scale, -2.0, 2.0), jnp.clip(dy[i] / y_scale, -2.0, 2.0), dirs[i]])
    return jnp.where(on.any(), near, none)


# ---- observation --------------------------------------------------------------------------
def observe(s: EnvState, levels=None):
    f32, i32 = jnp.float32, jnp.int32
    lv = _pick(_levels(levels), s.level)
    r = jnp.floor(s.y - HEIGHT / 2).astype(i32)
    c = jnp.floor(s.x).astype(i32)
    # window start in the padded grids; never negative: lax.dynamic_slice would wrap a negative start (like
    # Python's a[-1]) to the bottom of the map and then clamp it, showing ground where there is sky
    start = (jnp.maximum(r + ABOVE, 0), jnp.maximum(c, 0))
    solid = lax.dynamic_slice(lv.solid_pad, start, PATCH)
    oneway = lax.dynamic_slice(lv.oneway_pad, start, PATCH)
    goal = lax.dynamic_slice(lv.goal_pad, start, PATCH)

    wr = (r - PR + jnp.arange(PATCH[0]))[:, None]       # tile row of every window cell
    wc = (c - PC + jnp.arange(PATCH[1]))[None, :]       # tile col of every window cell

    def box(left, right, top, bottom):                   # window cells touched by a box
        return (wc + 1 > left) & (wc < right) & (wr + 1 > top) & (wr < bottom)

    def boxes(left, right, top, bottom, on):            # several boxes (leading axis) -> cells touched by any
        hit = box(left[:, None, None], right[:, None, None], top[:, None, None], bottom[:, None, None])
        return (hit & on[:, None, None]).any(axis=0)

    plat = boxes(s.plat_x - lv.plat_half, s.plat_x + lv.plat_half, lv.plat_top, lv.plat_top + PLAT_THICK, lv.plat_on)
    enemy = boxes(s.enemy_x - L.ENEMY_HALF_W, s.enemy_x + L.ENEMY_HALF_W, lv.enemy_y - L.ENEMY_H, lv.enemy_y,
                  lv.enemy_on)
    cx, cy = _coin_pos(s.plat_x, lv)
    coins = ((jnp.floor(cx)[:, None, None] == wc) & (jnp.floor(cy)[:, None, None] == wr)
             & s.coins_alive[:, None, None]).any(axis=0)

    patches = jnp.concatenate([solid.ravel(), (oneway | plat).ravel(), coins.ravel(),
                               goal.ravel(), enemy.ravel()]).astype(f32)

    scalars = jnp.concatenate([jnp.stack([
        s.vx / RUN_SPEED,
        s.vy / JUMP_SPEED,
        (_tile_support(s.x, s.y, lv) | _on_mplat(s.x, s.y, s.plat_x, lv).any()).astype(f32),
        s.x - jnp.floor(s.x) - 0.5,                  # sub-tile position
        s.y - jnp.floor(s.y) - 0.5,
        s.x / lv.w.astype(f32),                      # progress through the level
        s.t / MAX_STEPS]),                           # time used
        _nearest(s.x, s.y, s.enemy_x, lv.enemy_y, s.enemy_dir, lv.enemy_on, 8.0, 4.0),   # nearest enemy: dx, dy, dir
        _nearest(s.x, s.y, s.plat_x, lv.plat_top, s.plat_dir, lv.plat_on, 8.0, 4.0),     # nearest moving platform
    ]).astype(f32)
    return jnp.concatenate([patches, scalars])


# ---- dynamics -----------------------------------------------------------------------------
def step(state: EnvState, action, random_starts: bool = False, random_phases: bool = False,
         spawn_prob: float = SPAWN_PROB, progress_coef: float = 0.0, levels=None, keep_level: bool = False):
    """One tick.  The options are static training settings (bind them with functools.partial).
    keep_level: the auto-reset stays on the same level (evaluation); otherwise it picks a random one of `levels`."""
    f32, i32 = jnp.float32, jnp.int32
    levels = _levels(levels)
    lv = _pick(levels, state.level)
    action = jnp.asarray(action, i32)
    next_key, reset_key = jax.random.split(state.rng)
    hdir = jnp.asarray([0.0, -1.0, 1.0], f32)[action % 3]
    want_jump = action >= 3

    # moving platforms: move, then flip at the ends (like the original script)
    new_px = state.plat_x + state.plat_dir * L.PLATFORM_SPEED * DT
    new_pd = jnp.where(new_px >= lv.plat_x_max, -1.0,
                       jnp.where(new_px <= lv.plat_x_min, 1.0, state.plat_dir))
    plat_dx = new_px - state.plat_x

    tile_sup = _tile_support(state.x, state.y, lv)
    on_mp = _on_mplat(state.x, state.y, state.plat_x, lv)    # (Kmax,)
    on_floor = tile_sup | on_mp.any()
    carry = jnp.where(tile_sup, 0.0, _first(on_mp, plat_dx))    # the platform carries whoever stands on it

    vx = hdir * RUN_SPEED
    vy = jnp.minimum(state.vy + GRAVITY * DT, MAX_FALL)
    vy = jnp.where(want_jump & on_floor, -JUMP_SPEED, vy)

    # --- move in x, stop at solid walls ---
    dx = vx * DT + carry
    nx = state.x + dx
    col = jnp.floor(nx + jnp.sign(dx) * HW).astype(i32)
    r_top = jnp.floor(state.y - HEIGHT + EPS).astype(i32)
    r_bot = jnp.floor(state.y - EPS).astype(i32)
    blocked = (dx != 0) & (_solid(r_top, col, lv) | _solid(r_bot, col, lv))
    nx = jnp.where(blocked, jnp.where(dx > 0, col - HW - EPS, col + 1 + HW + EPS), nx)

    # --- move in y: land on solid tiles / one-way tiles / a moving platform, bump ceilings ---
    ny = state.y + vy * DT
    c_l = jnp.floor(nx - HW + EPS).astype(i32)
    c_r = jnp.floor(nx + HW - EPS).astype(i32)
    row_feet = jnp.floor(ny).astype(i32)
    row_head = jnp.floor(ny - HEIGHT).astype(i32)
    solid_hit = _solid(row_feet, c_l, lv) | _solid(row_feet, c_r, lv)
    oneway_hit = (_oneway(row_feet, c_l, lv) | _oneway(row_feet, c_r, lv)) & (state.y <= row_feet + 0.01)
    landed_tile = (vy > 0) & (solid_hit | oneway_hit)
    over_mp = (nx + HW > new_px - lv.plat_half) & (nx - HW < new_px + lv.plat_half)
    landed_mp = (vy > 0) & over_mp & (state.y <= lv.plat_top + 0.01) & (ny >= lv.plat_top) & lv.plat_on   # (Kmax,)
    mp_top = jnp.min(jnp.concatenate([jnp.where(landed_mp, lv.plat_top, 1e9), jnp.asarray([1e9], f32)]))  # highest
    bonk = (vy < 0) & (_solid(row_head, c_l, lv) | _solid(row_head, c_r, lv))
    ny = jnp.where(landed_tile, row_feet.astype(f32), ny)
    landed_mp = landed_mp.any()
    ny = jnp.where(landed_mp, mp_top, ny)
    ny = jnp.where(bonk, row_head + 1.0 + HEIGHT + EPS, ny)
    vy = jnp.where(landed_tile | landed_mp | bonk, 0.0, vy)

    # --- enemies: walk back and forth between their own x limits (walls play no part) ---
    # (riders: limits are relative to their platform, and the platform's own movement is added to the walk)
    base = _enemy_base(state.plat_x, lv)
    e_dir = jnp.where(state.enemy_x >= base + lv.enemy_max, -1.0, state.enemy_dir)
    e_dir = jnp.where(state.enemy_x < base + lv.enemy_min, 1.0, e_dir)
    carried = (jnp.where(lv.enemy_ride >= 0, plat_dx[jnp.maximum(lv.enemy_ride, 0)], 0.0)
               if lv.plat_top.shape[0] else 0.0)
    new_ex = state.enemy_x + e_dir * L.ENEMY_SPEED * DT + carried
    hit_enemy = ((nx + HW > new_ex - L.ENEMY_HALF_W) & (nx - HW < new_ex + L.ENEMY_HALF_W) &
                 (ny > lv.enemy_y - L.ENEMY_H) & (ny - HEIGHT < lv.enemy_y) & lv.enemy_on).any()

    # --- coins (circles), goal, death ---
    cx, cy = _coin_pos(new_px, lv)
    qx = jnp.clip(cx, nx - HW, nx + HW)
    qy = jnp.clip(cy, ny - HEIGHT, ny)
    touch = (cx - qx) ** 2 + (cy - qy) ** 2 < L.COIN_RADIUS ** 2
    got = state.coins_alive & touch
    n_got = got.sum().astype(f32)
    coins_alive = state.coins_alive & ~touch
    goal = jnp.any(lv.goal & _overlap(nx, ny, lv.goal.shape))
    dead = (ny > lv.h + 1.0) | hit_enemy
    task_reward = n_got * R_COIN + goal.astype(f32) * R_GOAL + dead.astype(f32) * R_DEATH
    furthest = jnp.maximum(state.furthest, nx)                           # progress bonus: new ground only
    reward = task_reward + progress_coef * (furthest - state.furthest)
    terminated = goal | dead
    t = state.t + 1
    truncated = (t >= MAX_STEPS) & ~terminated
    done = terminated | truncated

    nxt = EnvState(x=nx.astype(f32), y=ny.astype(f32), vx=vx.astype(f32), vy=vy.astype(f32),
                   coins_alive=coins_alive, plat_x=new_px.astype(f32), plat_dir=new_pd.astype(f32),
                   enemy_x=new_ex.astype(f32), enemy_dir=e_dir.astype(f32), t=t.astype(i32),
                   furthest=furthest.astype(f32), from_spawn=state.from_spawn,
                   rng=next_key, ep_return=state.ep_return + task_reward,
                   ep_coins=state.ep_coins + n_got, level=state.level)
    final_obs = observe(nxt, levels)
    info = dict(terminated=terminated, truncated=truncated, final_obs=final_obs,
                ep_return=nxt.ep_return, ep_length=t, ep_coins=nxt.ep_coins, goal=goal,
                max_x=nxt.furthest, from_spawn=state.from_spawn, level=state.level)

    fresh = reset(reset_key, random_start=random_starts, random_phases=random_phases, spawn_prob=spawn_prob,
                  levels=levels, level=state.level if keep_level else None)     # auto-reset, still no CPU involved
    out = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), fresh, nxt)
    if random_starts or random_phases or n_levels(levels) > 1:
        fresh_obs = observe(fresh, levels)
    else:                                                       # always the same start: a constant
        fresh_obs = RESET_OBS if levels is DEFAULT_LEVELS else observe(initial_state(levels=levels), levels)
    obs = jnp.where(done, fresh_obs, final_obs)
    return out, obs, reward.astype(f32), done, info


RESET_OBS = observe(initial_state())
