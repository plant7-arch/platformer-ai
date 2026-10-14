"""Minimal pure-JAX platformer.  One environment = pure functions on a small pytree;
use jax.vmap for thousands of them.  Nothing here touches the CPU.

    state = reset(key)                                        # the real spawn
    state, obs, reward, done, info = step(state, action)     # auto-resets when done

Training helpers (all off by default, switched on in ppo.Config):
    random_phases=True    platform and enemy start at random phases, spawn position is jittered
    random_starts=True    some episodes restart at random points along the level (see reset)
    progress_coef=0.1     small bonus per tile of NEW furthest-right progress (not farmable)

Actions (6): horizontal = [none, left, right][a % 3],  jump = a >= 3.
Rules: solid tiles, one-way platforms (static tiles + any number of moving platforms that carry the player),
any number of patrolling enemies (touch = death), coins (one rides on every moving platform), a goal, pits.
The level may have 0, 1 or many enemies and moving platforms; all state arrays have a fixed length (N_ENEMIES, N_PLATS).
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
OBS_DIM = N_CHANNELS * PATCH[0] * PATCH[1] + N_SCALARS
PLAT_THICK = 0.5         # only used to draw the moving platform into the observation

H, W, N_COINS, N_ENEMIES, N_PLATS = L.H, L.W, L.N_COINS, L.N_ENEMIES, L.N_PLATS
_SOLID = jnp.asarray(L.SOLID)
_ONEWAY = jnp.asarray(L.ONEWAY)
_GOAL = jnp.asarray(L.GOAL)
_ROWS = jnp.arange(H)[:, None]
_COLS = jnp.arange(W)[None, :]
_STATIC_COIN_X, _STATIC_COIN_Y = jnp.asarray(L.COIN_X[:L.N_STATIC_COINS]), jnp.asarray(L.COIN_Y[:L.N_STATIC_COINS])
_ENEMY_X0, _ENEMY_Y = jnp.asarray(L.ENEMY_X), jnp.asarray(L.ENEMY_Y)                 # (N_ENEMIES,)
_ENEMY_MIN, _ENEMY_MAX = jnp.asarray(L.ENEMY_MIN), jnp.asarray(L.ENEMY_MAX)
_PLAT_TOP, _PLAT_HALF = jnp.asarray(L.PLAT_TOP), jnp.asarray(L.PLAT_HALF)           # (N_PLATS,)
_PLAT_X_MIN, _PLAT_X_MAX = jnp.asarray(L.PLAT_X_MIN), jnp.asarray(L.PLAT_X_MAX)      # reverse at these
_PLAT_START_X, _PLAT_DIR0 = jnp.asarray(L.PLAT_START_X), jnp.asarray(L.PLAT_DIR0)
_START_X, _START_Y = jnp.asarray(L.START_X), jnp.asarray(L.START_Y)


def _padded(grid, wall=False):
    p = np.pad(grid, ((PR, PR), (PC, PC)))
    if wall:                       # columns outside the map are solid walls
        p[:, :PC] = True
        p[:, -PC:] = True
    return jnp.asarray(p)


_SOLID_PAD = _padded(L.SOLID, wall=True)
_ONEWAY_PAD = _padded(L.ONEWAY)
_GOAL_PAD = _padded(L.GOAL)


@struct.dataclass
class EnvState:
    x: jax.Array            # centre x of the player
    y: jax.Array            # y of the player's FEET (bottom edge)
    vx: jax.Array
    vy: jax.Array
    coins_alive: jax.Array  # (N_COINS,) bool; the last N_PLATS coins ride on the moving platforms
    plat_x: jax.Array       # (N_PLATS,) moving platform centre x
    plat_dir: jax.Array     # (N_PLATS,) +1 / -1
    enemy_x: jax.Array      # (N_ENEMIES,) enemy centre x (their y is constant)
    enemy_dir: jax.Array    # (N_ENEMIES,) +1 / -1
    t: jax.Array            # steps since reset
    furthest: jax.Array     # furthest x reached this episode (progress bonus)
    from_spawn: jax.Array   # bool: episode started at the real spawn (logging)
    rng: jax.Array          # PRNG key stream for random restarts
    ep_return: jax.Array    # task return (coins + goal + death), without shaping; logging only
    ep_coins: jax.Array     # logging only


def initial_state(rng=None) -> EnvState:
    f32 = jnp.float32
    return EnvState(
        x=f32(L.SPAWN_X), y=f32(L.SPAWN_Y), vx=f32(0), vy=f32(0),
        coins_alive=jnp.ones(N_COINS, dtype=bool),
        plat_x=_PLAT_START_X, plat_dir=_PLAT_DIR0, enemy_x=_ENEMY_X0, enemy_dir=jnp.ones(N_ENEMIES, f32),
        t=jnp.int32(0), furthest=f32(L.SPAWN_X), from_spawn=jnp.bool_(True),
        rng=jax.random.PRNGKey(0) if rng is None else rng,
        ep_return=f32(0), ep_coins=f32(0))


def _sample_start(key, use_s_points: bool, random_phases: bool, spawn_prob: float) -> EnvState:
    """Random episode start.
    * random_phases: platform and enemy begin at random phases and the spawn position is jittered, so
      the agent has to react to what it sees instead of replaying one timed route.
    * use_s_points: with probability 1 - spawn_prob start at a random 'S' point or on the ferry
      instead of the spawn (always with random phases) to practise the hard parts directly."""
    f32 = jnp.float32
    k = jax.random.split(key, 10)
    plat_x = jax.random.uniform(k[2], (N_PLATS,), minval=_PLAT_X_MIN, maxval=_PLAT_X_MAX)
    plat_dir = jnp.where(jax.random.uniform(k[3], (N_PLATS,)) < 0.5, 1.0, -1.0)
    ex = jax.random.uniform(k[4], (N_ENEMIES,), minval=_ENEMY_MIN + 0.1, maxval=_ENEMY_MAX - 0.1)
    e_dir = jnp.where(jax.random.uniform(k[5], (N_ENEMIES,)) < 0.5, 1.0, -1.0)
    jx = jax.random.uniform(k[7], minval=-SPAWN_JITTER_X, maxval=SPAWN_JITTER_X)
    jy = jax.random.uniform(k[8], minval=-SPAWN_JITTER_Y, maxval=0.0)
    spawn_x = L.SPAWN_X + (jx if random_phases else 0.0)
    spawn_y = L.SPAWN_Y + (jy if random_phases else 0.0)

    if use_s_points:
        use_spawn = jax.random.uniform(k[0]) < spawn_prob
        # options: the S points, plus (if the level has moving platforms) riding one of them, picked at random
        idx = jax.random.randint(k[1], (), 0, L.N_STARTS + (1 if N_PLATS else 0))
        i = jnp.minimum(idx, L.N_STARTS - 1)
        sx = jnp.where(use_spawn, spawn_x, _START_X[i])
        sy = jnp.where(use_spawn, spawn_y, _START_Y[i])
        if N_PLATS:
            on_ferry = idx == L.N_STARTS
            p = jax.random.randint(k[9], (), 0, N_PLATS)
            sx = jnp.where(use_spawn, spawn_x, jnp.where(on_ferry, plat_x[p], _START_X[i]))
            sy = jnp.where(use_spawn, spawn_y, jnp.where(on_ferry, _PLAT_TOP[p], _START_Y[i]))
        phases = jnp.logical_or(~use_spawn, random_phases)
    else:
        use_spawn = jnp.bool_(True)
        sx, sy = jnp.asarray(spawn_x, f32), jnp.asarray(spawn_y, f32)
        phases = jnp.bool_(random_phases)

    # never start on top of an enemy: push it 3 tiles away (to whichever side still fits its patrol range)
    away = jnp.where(sx + 3.0 <= _ENEMY_MAX, sx + 3.0, sx - 3.0)
    away = jnp.clip(away, _ENEMY_MIN + 0.1, _ENEMY_MAX - 0.1)
    ex = jnp.where(jnp.abs(ex - sx) < 1.5, away, ex)
    base = initial_state(rng=k[6])
    return base.replace(
        x=jnp.asarray(sx, f32), y=jnp.asarray(sy, f32), furthest=jnp.asarray(sx, f32),
        plat_x=jnp.where(phases, plat_x, base.plat_x).astype(f32),
        plat_dir=jnp.where(phases, plat_dir, base.plat_dir).astype(f32),
        enemy_x=jnp.where(phases, ex, base.enemy_x).astype(f32),
        enemy_dir=jnp.where(phases, e_dir, base.enemy_dir).astype(f32),
        from_spawn=use_spawn)


def reset(key=None, random_start: bool = False, random_phases: bool = False,
          spawn_prob: float = SPAWN_PROB) -> EnvState:
    """The exact default spawn, unless a randomisation option is on (see _sample_start)."""
    if key is None:
        key = jax.random.PRNGKey(0)
    if random_start or random_phases:
        return _sample_start(key, random_start, random_phases, spawn_prob)
    return initial_state(rng=key)


# ---- tile queries -------------------------------------------------------------------------
def _in_map(r, c):
    return (r >= 0) & (r < H) & (c >= 0) & (c < W)


def _grid_at(grid, r, c):
    return grid[jnp.clip(r, 0, H - 1), jnp.clip(c, 0, W - 1)] & _in_map(r, c)


def _solid(r, c):
    """Solid tile?  Side walls outside the map; open sky above and open pit below."""
    inside_c = (c >= 0) & (c < W)
    return jnp.where(inside_c, _grid_at(_SOLID, r, c), True)


def _oneway(r, c):
    return _grid_at(_ONEWAY, r, c)


def _tile_support(x, y):
    """Standing on a solid tile or on top of a one-way tile?"""
    row = jnp.floor(y + 0.01).astype(jnp.int32)
    c_l = jnp.floor(x - HW + EPS).astype(jnp.int32)
    c_r = jnp.floor(x + HW - EPS).astype(jnp.int32)
    solid = _solid(row, c_l) | _solid(row, c_r)
    oneway = (_oneway(row, c_l) | _oneway(row, c_r)) & (y - row < 0.02)
    return solid | oneway


def _on_mplat(x, y, plat_x):
    """(N_PLATS,) bool: standing on each moving platform?"""
    return ((jnp.abs(y - _PLAT_TOP) <= 0.02) &
            (x + HW > plat_x - _PLAT_HALF) & (x - HW < plat_x + _PLAT_HALF))


def _overlap(x, y):
    """(H, W) bool: tiles overlapped by the player's box."""
    return ((_COLS + 1 > x - HW) & (_COLS < x + HW) &
            (_ROWS + 1 > y - HEIGHT) & (_ROWS < y))


def _coin_pos(plat_x):
    """(N_COINS,) x and y of every coin: the static ones, then one per moving platform riding on it."""
    cx = jnp.concatenate([_STATIC_COIN_X, plat_x + L.RIDER_COIN_OFFSET[0]])
    cy = jnp.concatenate([_STATIC_COIN_Y, _PLAT_TOP + L.RIDER_COIN_OFFSET[1]])
    return cx, cy


def _first(mask, values):
    """values[i] for the first True in mask, else 0 (empty-safe: no argmax on a zero-length array)."""
    first = mask & (jnp.cumsum(mask.astype(jnp.int32)) == 1)
    return jnp.sum(jnp.where(first, values, 0.0))


def _nearest(px, py, xs, ys, dirs, x_scale, y_scale):
    """The 3 observation scalars for the nearest of several things: dx, dy (scaled, clipped) and its direction.
    With none in the level: far away to the right, standing still."""
    if xs.shape[0] == 0:
        return jnp.asarray([2.0, 0.0, 0.0], jnp.float32)
    dx, dy = xs - px, ys - py
    i = jnp.argmin(dx * dx + dy * dy)
    return jnp.stack([jnp.clip(dx[i] / x_scale, -2.0, 2.0), jnp.clip(dy[i] / y_scale, -2.0, 2.0), dirs[i]])


# ---- observation --------------------------------------------------------------------------
def observe(s: EnvState):
    f32, i32 = jnp.float32, jnp.int32
    r = jnp.floor(s.y - HEIGHT / 2).astype(i32)
    c = jnp.floor(s.x).astype(i32)
    solid = lax.dynamic_slice(_SOLID_PAD, (r, c), PATCH)
    oneway = lax.dynamic_slice(_ONEWAY_PAD, (r, c), PATCH)
    goal = lax.dynamic_slice(_GOAL_PAD, (r, c), PATCH)

    wr = (r - PR + jnp.arange(PATCH[0]))[:, None]       # tile row of every window cell
    wc = (c - PC + jnp.arange(PATCH[1]))[None, :]       # tile col of every window cell

    def box(left, right, top, bottom):                   # window cells touched by a box
        return (wc + 1 > left) & (wc < right) & (wr + 1 > top) & (wr < bottom)

    def boxes(left, right, top, bottom):                # several boxes (leading axis) -> cells touched by any
        return box(left[:, None, None], right[:, None, None], top[:, None, None], bottom[:, None, None]).any(axis=0)

    plat = boxes(s.plat_x - _PLAT_HALF, s.plat_x + _PLAT_HALF, _PLAT_TOP, _PLAT_TOP + PLAT_THICK)
    enemy = boxes(s.enemy_x - L.ENEMY_HALF_W, s.enemy_x + L.ENEMY_HALF_W, _ENEMY_Y - L.ENEMY_H, _ENEMY_Y)
    cx, cy = _coin_pos(s.plat_x)
    coins = ((jnp.floor(cx)[:, None, None] == wc) & (jnp.floor(cy)[:, None, None] == wr)
             & s.coins_alive[:, None, None]).any(axis=0)

    patches = jnp.concatenate([solid.ravel(), (oneway | plat).ravel(), coins.ravel(),
                               goal.ravel(), enemy.ravel()]).astype(f32)

    scalars = jnp.concatenate([jnp.stack([
        s.vx / RUN_SPEED,
        s.vy / JUMP_SPEED,
        (_tile_support(s.x, s.y) | _on_mplat(s.x, s.y, s.plat_x).any()).astype(f32),
        s.x - jnp.floor(s.x) - 0.5,                  # sub-tile position
        s.y - jnp.floor(s.y) - 0.5,
        s.x / W,                                     # progress through the level
        s.t / MAX_STEPS]),                           # time used
        _nearest(s.x, s.y, s.enemy_x, _ENEMY_Y, s.enemy_dir, 8.0, 4.0),     # nearest enemy: dx, dy, direction
        _nearest(s.x, s.y, s.plat_x, _PLAT_TOP, s.plat_dir, 8.0, 4.0),      # nearest moving platform
    ]).astype(f32)
    return jnp.concatenate([patches, scalars])


# ---- dynamics -----------------------------------------------------------------------------
def step(state: EnvState, action, random_starts: bool = False, random_phases: bool = False,
         spawn_prob: float = SPAWN_PROB, progress_coef: float = 0.0):
    """One tick.  The options are static training settings (bind them with functools.partial)."""
    f32, i32 = jnp.float32, jnp.int32
    action = jnp.asarray(action, i32)
    next_key, reset_key = jax.random.split(state.rng)
    hdir = jnp.asarray([0.0, -1.0, 1.0], f32)[action % 3]
    want_jump = action >= 3

    # moving platform: move, then flip at the ends (like the original script)
    new_px = state.plat_x + state.plat_dir * L.PLATFORM_SPEED * DT
    new_pd = jnp.where(new_px >= _PLAT_X_MAX, -1.0,
                       jnp.where(new_px <= _PLAT_X_MIN, 1.0, state.plat_dir))
    plat_dx = new_px - state.plat_x

    tile_sup = _tile_support(state.x, state.y)
    on_mp = _on_mplat(state.x, state.y, state.plat_x)       # (N_PLATS,)
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
    blocked = (dx != 0) & (_solid(r_top, col) | _solid(r_bot, col))
    nx = jnp.where(blocked, jnp.where(dx > 0, col - HW - EPS, col + 1 + HW + EPS), nx)

    # --- move in y: land on solid tiles / one-way tiles / the moving platform, bump ceilings ---
    ny = state.y + vy * DT
    c_l = jnp.floor(nx - HW + EPS).astype(i32)
    c_r = jnp.floor(nx + HW - EPS).astype(i32)
    row_feet = jnp.floor(ny).astype(i32)
    row_head = jnp.floor(ny - HEIGHT).astype(i32)
    solid_hit = _solid(row_feet, c_l) | _solid(row_feet, c_r)
    oneway_hit = (_oneway(row_feet, c_l) | _oneway(row_feet, c_r)) & (state.y <= row_feet + 0.01)
    landed_tile = (vy > 0) & (solid_hit | oneway_hit)
    over_mp = (nx + HW > new_px - _PLAT_HALF) & (nx - HW < new_px + _PLAT_HALF)
    landed_mp = (vy > 0) & over_mp & (state.y <= _PLAT_TOP + 0.01) & (ny >= _PLAT_TOP)      # (N_PLATS,)
    mp_top = jnp.min(jnp.concatenate([jnp.where(landed_mp, _PLAT_TOP, 1e9), jnp.asarray([1e9], f32)]))  # highest hit
    bonk = (vy < 0) & (_solid(row_head, c_l) | _solid(row_head, c_r))
    ny = jnp.where(landed_tile, row_feet.astype(f32), ny)
    landed_mp = landed_mp.any()
    ny = jnp.where(landed_mp, mp_top, ny)
    ny = jnp.where(bonk, row_head + 1.0 + HEIGHT + EPS, ny)
    vy = jnp.where(landed_tile | landed_mp | bonk, 0.0, vy)

    # --- enemies: walk back and forth between their own x limits (walls play no part) ---
    e_dir = jnp.where(state.enemy_x >= _ENEMY_MAX, -1.0, state.enemy_dir)
    e_dir = jnp.where(state.enemy_x < _ENEMY_MIN, 1.0, e_dir)
    new_ex = state.enemy_x + e_dir * L.ENEMY_SPEED * DT
    hit_enemy = ((nx + HW > new_ex - L.ENEMY_HALF_W) & (nx - HW < new_ex + L.ENEMY_HALF_W) &
                 (ny > _ENEMY_Y - L.ENEMY_H) & (ny - HEIGHT < _ENEMY_Y)).any()

    # --- coins (circles), goal, death ---
    cx, cy = _coin_pos(new_px)
    qx = jnp.clip(cx, nx - HW, nx + HW)
    qy = jnp.clip(cy, ny - HEIGHT, ny)
    touch = (cx - qx) ** 2 + (cy - qy) ** 2 < L.COIN_RADIUS ** 2
    got = state.coins_alive & touch
    n_got = got.sum().astype(f32)
    coins_alive = state.coins_alive & ~touch
    goal = jnp.any(_GOAL & _overlap(nx, ny))
    dead = (ny > H + 1.0) | hit_enemy
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
                   ep_coins=state.ep_coins + n_got)
    final_obs = observe(nxt)
    info = dict(terminated=terminated, truncated=truncated, final_obs=final_obs,
                ep_return=nxt.ep_return, ep_length=t, ep_coins=nxt.ep_coins, goal=goal,
                max_x=nxt.furthest, from_spawn=state.from_spawn)

    fresh = reset(reset_key, random_start=random_starts, random_phases=random_phases,
                  spawn_prob=spawn_prob)                        # auto-reset, still no CPU involved
    out = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), fresh, nxt)
    fresh_obs = observe(fresh) if (random_starts or random_phases) else RESET_OBS
    obs = jnp.where(done, fresh_obs, final_obs)
    return out, obs, reward.astype(f32), done, info


RESET_OBS = observe(initial_state())
