"""Behaviour of the environment on the Godot-layout level.  Run: uv run pytest"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from platformer_ai import env as E
from platformer_ai import level as L

STEP = jax.jit(E.step)
NOOP, LEFT, RIGHT, JUMP, JUMP_LEFT, JUMP_RIGHT = range(6)
HEIGHT_GAP = E.JUMP_SPEED ** 2 / (2 * E.GRAVITY)


def put(x, y, vx=0.0, vy=0.0, **kw):
    """Initial state with the player somewhere else (other fields can be overridden)."""
    f = jnp.float32
    return E.initial_state().replace(x=f(x), y=f(y), vx=f(vx), vy=f(vy), furthest=f(x), **kw)


def run(state, actions):
    """Apply actions; return (final_state, list of per-step dicts)."""
    log = []
    for a in actions:
        state, obs, reward, done, info = STEP(state, a)
        log.append(dict(obs=obs, reward=float(reward), done=bool(done), info=info))
    return state, log


def channel(obs, k):
    n = E.PATCH[0] * E.PATCH[1]
    return np.asarray(obs)[k * n:(k + 1) * n].reshape(E.PATCH)


SOLID_CH, ONEWAY_CH, COIN_CH, GOAL_CH, ENEMY_CH = range(5)


def coin_index(x, y):
    return int(np.flatnonzero((L.COIN_X == x) & (L.COIN_Y == y))[0])


# ---------------------------------------------------------------- level / observation
def test_level_matches_godot_layout():
    assert (L.H, L.W) == (10, 70)
    assert int(L.SOLID.sum()) == 177                       # collision tiles of the Godot level
    assert int(L.ONEWAY.sum()) == 2 and L.N_COINS == 9
    assert (L.PLAT_X_MIN, L.PLAT_X_MAX, L.PLAT_TOP, L.PLAT_HALF) == (28.0, 34.0, 7.0, 1.0)
    assert (L.ENEMY_X, L.ENEMY_Y) == (48.5, 7.0)
    assert L.GOAL.sum() == 2


def test_obs_dim_and_finite():
    obs = E.observe(E.initial_state())
    assert obs.shape == (E.OBS_DIM,)
    assert np.isfinite(np.asarray(obs)).all()


def test_obs_channels_alignment():
    pr, pc = E.PR, E.PC
    # standing on ground: solid directly below the feet, free space at the player's centre
    obs = E.observe(put(14.0, 8.0))
    assert channel(obs, SOLID_CH)[pr + 1, pc] == 1 and channel(obs, SOLID_CH)[pr, pc] == 0
    # at spawn: coin right below, one-way platform two rows below
    obs = E.observe(E.initial_state())
    assert channel(obs, COIN_CH)[pr + 1, pc] == 1
    assert channel(obs, ONEWAY_CH)[pr + 2, pc] == 1 and channel(obs, ONEWAY_CH)[pr + 2, pc - 1] == 1
    # enemy two tiles to the right; moving platform visible in the one-way channel
    obs = E.observe(put(46.0, 7.0))
    assert channel(obs, ENEMY_CH)[pr, pc + 2] == 1
    obs = E.observe(put(26.0, 7.0))
    assert channel(obs, ONEWAY_CH)[pr + 1, pc + 1] == 1 and channel(obs, ONEWAY_CH)[pr + 1, pc + 2] == 1


# ---------------------------------------------------------------- player physics
def test_spawn_drops_onto_one_way_platform_and_takes_coin():
    s, log = run(E.initial_state(), [NOOP] * 30)
    assert float(s.y) == pytest.approx(5.0) and float(s.vy) == 0.0       # standing on the platform
    assert not bool(s.coins_alive[coin_index(17.5, 4.5)])
    assert sum(l["reward"] for l in log) == pytest.approx(E.R_COIN)


def test_gravity_and_terminal_velocity():
    s, _ = run(put(10.5, -10.0), [NOOP] * 3)
    assert float(s.vy) == pytest.approx(3 * E.GRAVITY * E.DT, abs=1e-4)
    s, _ = run(put(10.5, -10.0), [NOOP] * 15)
    assert float(s.vy) == pytest.approx(E.MAX_FALL, abs=1e-4)


def test_run_speed():
    s, _ = run(put(12.0, 8.0), [RIGHT] * 6)
    assert float(s.x) == pytest.approx(12.0 + 6 * E.RUN_SPEED * E.DT, abs=1e-3)
    assert float(s.y) == pytest.approx(8.0)
    s, _ = run(put(12.0, 8.0), [LEFT] * 6)
    assert float(s.x) == pytest.approx(12.0 - 6 * E.RUN_SPEED * E.DT, abs=1e-3)


def test_wall_blocks_horizontal_movement():
    x0 = 22 - E.HW - 0.01                         # one-tile step: row 7, columns 22-26
    s, _ = run(put(x0, 8.0), [RIGHT] * 5)
    assert x0 - 1e-6 <= float(s.x) <= 22 - E.HW + 1e-3


def test_left_map_edge_is_a_wall():
    s, _ = run(put(0.5, 7.0), [LEFT] * 10)
    assert float(s.x) >= E.HW - 1e-3


def test_jump_height_and_landing():
    s = put(14.0, 8.0)
    ys = []
    for a in [JUMP] + [NOOP] * 40:
        s, *_ = STEP(s, a)
        ys.append(float(s.y))
    assert 2.5 < 8.0 - min(ys) < 2.9              # ~2.7 tiles
    assert ys[-1] == pytest.approx(8.0) and float(s.vy) == 0.0


def test_no_jump_in_air():
    s, _ = run(put(10.5, -5.0), [JUMP])
    assert float(s.vy) == pytest.approx(E.GRAVITY * E.DT, abs=1e-4)


def test_head_bonk_under_solid():
    # Overhang: row 6 is solid over column 10, leaving 1 tile of headroom above the ground (y=8).
    s = put(10.9, 8.0)                              # standing on column 11, head under column 10
    tops = []
    for a in [JUMP] + [NOOP] * 30:
        s, *_ = STEP(s, a)
        tops.append(float(s.y) - E.HEIGHT)
    assert min(tops) >= 7.0 - 1e-3                  # head never enters the solid tile
    assert float(s.y) == pytest.approx(8.0)         # and the player drops back to the ground


def test_one_way_platform_cannot_be_reached_from_the_ground():
    s = put(16.5, 8.0)
    ys = []
    for a in [JUMP] + [NOOP] * 40:
        s, *_ = STEP(s, a)
        ys.append(float(s.y))
    assert min(ys) > 5.0                           # head passes through, feet never get above the top
    assert ys[-1] == pytest.approx(8.0)            # falls back to the ground


def test_can_walk_off_one_way_platform():
    s, _ = run(E.initial_state(), [NOOP] * 20 + [RIGHT] * 40)
    assert float(s.y) == pytest.approx(8.0) and float(s.x) > 18.0


# ---------------------------------------------------------------- moving platform
def test_moving_platform_sweeps_and_turns():
    s, _ = run(E.initial_state(), [NOOP] * 20)
    assert float(s.plat_x) == pytest.approx(28.0 + 20 * L.PLATFORM_SPEED * E.DT, abs=1e-3)
    s, _ = run(E.initial_state(), [NOOP] * 57)
    assert float(s.plat_dir) == -1.0 and float(s.plat_x) > L.PLAT_X_MAX
    s, _ = run(E.initial_state(), [NOOP] * 114)
    assert float(s.plat_dir) == 1.0 and float(s.plat_x) < L.PLAT_X_MIN + 0.3


def test_platform_carries_player():
    s, _ = run(put(28.0, 7.0), [NOOP] * 10)
    assert float(s.y) == pytest.approx(7.0)
    assert float(s.x) == pytest.approx(28.0 + 10 * L.PLATFORM_SPEED * E.DT, abs=1e-3)


def test_coin_rides_on_platform():
    i = L.N_COINS - 1
    assert bool(E.initial_state().coins_alive[i])
    s, log = run(put(28.0, 7.0), [NOOP])               # standing on the platform picks it up
    assert not bool(s.coins_alive[i]) and log[0]["reward"] == pytest.approx(E.R_COIN)


# ---------------------------------------------------------------- enemy
def test_enemy_patrols_between_walls():
    s = E.initial_state()
    xs, dirs = [], []
    for _ in range(400):
        s, *_ = STEP(s, NOOP)
        xs.append(float(s.enemy_x)); dirs.append(float(s.enemy_dir))
    assert 44.4 < min(xs) and max(xs) < 52.6
    assert sum(1 for a, b in zip(dirs, dirs[1:]) if a != b) >= 3        # turned around several times
    assert xs[9] == pytest.approx(48.5 + 10 * L.ENEMY_SPEED * E.DT, abs=1e-3)


def test_enemy_touch_kills():
    s, log = run(put(48.5, 7.0), [NOOP])
    assert log[0]["done"] and bool(log[0]["info"]["terminated"])
    assert log[0]["reward"] == pytest.approx(E.R_DEATH)


# ---------------------------------------------------------------- pickups, goal, death, episode
def test_coin_pickup_once():
    i = coin_index(20.5, 7.5)
    s, log = run(put(20.5, 8.0), [NOOP])
    assert log[0]["reward"] == pytest.approx(E.R_COIN) and not bool(s.coins_alive[i])
    _, log2 = run(s, [NOOP])
    assert log2[0]["reward"] == 0.0


def test_fall_in_pit_terminates_and_resets():
    s = put(31.0, 7.0)                              # inside the 8-tile pit, away from the platform
    for _ in range(60):
        s, obs, r, d, info = STEP(s, NOOP)
        if bool(d):
            break
    assert bool(d) and bool(info["terminated"]) and float(r) == pytest.approx(E.R_DEATH)
    assert float(s.x) == pytest.approx(L.SPAWN_X) and int(s.t) == 0           # auto-reset
    np.testing.assert_allclose(np.asarray(obs), np.asarray(E.RESET_OBS))


def test_goal_terminates():
    s, log = run(put(69.5, 4.0), [NOOP])
    assert log[0]["done"] and bool(log[0]["info"]["goal"]) and log[0]["reward"] >= E.R_GOAL


def test_truncation_after_max_steps():
    s = E.initial_state()
    for i in range(E.MAX_STEPS):
        s, obs, r, d, info = STEP(s, NOOP)
        assert bool(d) == (i == E.MAX_STEPS - 1)
    assert bool(info["truncated"]) and not bool(info["terminated"]) and int(s.t) == 0


def test_batched_step_with_vmap():
    n = 8
    states = jax.vmap(E.reset)(jax.random.split(jax.random.PRNGKey(0), n))
    actions = jnp.arange(n) % E.N_ACTIONS
    out, obs, r, d, info = jax.jit(jax.vmap(E.step))(states, actions)
    assert obs.shape == (n, E.OBS_DIM) and r.shape == (n,) and d.shape == (n,)
    assert info["final_obs"].shape == (n, E.OBS_DIM)


# ---------------------------------------------------------------- training helpers
def _box_overlap_enemy(s):
    return bool((float(s.x) + E.HW > float(s.enemy_x) - L.ENEMY_HALF_W) &
                (float(s.x) - E.HW < float(s.enemy_x) + L.ENEMY_HALF_W) &
                (float(s.y) > L.ENEMY_Y - L.ENEMY_H) & (float(s.y) - E.HEIGHT < L.ENEMY_Y))


def test_start_points_and_patrol_limits_come_from_the_level():
    assert L.N_STARTS == 8 and (L.ENEMY_MIN, L.ENEMY_MAX) == (44.5625, 52.4375)


def test_random_starts_are_valid_and_varied():
    reset = jax.jit(functools.partial(E.reset, random_start=True))
    states = [reset(jax.random.PRNGKey(i)) for i in range(300)]
    for s in states:
        assert E.HW <= float(s.x) <= L.W - E.HW
        supported = bool(E._tile_support(s.x, s.y)) or bool(E._on_mplat(s.x, s.y, s.plat_x))
        assert supported or bool(s.from_spawn), (float(s.x), float(s.y))    # stands on something
        assert not np.any(L.SOLID & np.asarray(E._overlap(s.x, s.y)))       # not inside a wall
        assert L.PLAT_X_MIN <= float(s.plat_x) <= L.PLAT_X_MAX
        assert L.ENEMY_MIN <= float(s.enemy_x) <= L.ENEMY_MAX
        assert not _box_overlap_enemy(s)                                    # never spawns on the enemy
        assert float(s.furthest) == pytest.approx(float(s.x))
    frac_spawn = np.mean([bool(s.from_spawn) for s in states])
    assert 0.35 < frac_spawn < 0.65                                         # ~SPAWN_PROB (0.5)
    assert len({round(float(s.x), 1) for s in states}) >= 8                 # many different starts
    assert any(28.0 <= float(s.x) <= 34.0 and float(s.y) == L.PLAT_TOP for s in states)   # on the ferry
    for s in states:
        if bool(s.from_spawn):                                              # real spawn = exact default state
            assert (float(s.x), float(s.y), float(s.plat_x), float(s.enemy_x)) == \
                   (L.SPAWN_X, L.SPAWN_Y, L.PLAT_X_MIN, L.ENEMY_X)


def test_random_phases_at_the_spawn():
    reset = jax.jit(functools.partial(E.reset, random_phases=True))
    states = [reset(jax.random.PRNGKey(i)) for i in range(100)]
    assert all(bool(s.from_spawn) for s in states)                          # still the spawn, just varied
    for s in states:
        assert abs(float(s.x) - L.SPAWN_X) <= E.SPAWN_JITTER_X + 1e-6
        assert L.SPAWN_Y - E.SPAWN_JITTER_Y - 1e-6 <= float(s.y) <= L.SPAWN_Y + 1e-6
        assert L.PLAT_X_MIN <= float(s.plat_x) <= L.PLAT_X_MAX
        assert L.ENEMY_MIN <= float(s.enemy_x) <= L.ENEMY_MAX
        assert float(s.plat_dir) in (-1.0, 1.0) and float(s.enemy_dir) in (-1.0, 1.0)
        assert float(s.furthest) == pytest.approx(float(s.x))
    assert len({round(float(s.plat_x), 2) for s in states}) > 50           # phases really differ
    assert len({round(float(s.enemy_x), 2) for s in states}) > 50
    assert {float(s.plat_dir) for s in states} == {-1.0, 1.0}
    assert len({round(float(s.x), 2) for s in states}) > 50                 # and so does the spawn x
    # the jittered spawn still drops onto the platform and picks up the first coin
    for s0 in states[:10]:
        s, log = run(s0, [NOOP] * 25)
        assert float(s.y) == pytest.approx(5.0) and not bool(s.coins_alive[coin_index(17.5, 4.5)])


def test_auto_reset_randomizes_phases_when_enabled():
    step = jax.jit(functools.partial(E.step, random_phases=True))
    plats, enemies = set(), set()
    for i in range(30):
        s = put(31.0, 7.0).replace(rng=jax.random.PRNGKey(i))               # falls into the pit
        for _ in range(60):
            s, obs, r, d, info = step(s, NOOP)
            if bool(d):
                break
        assert bool(d) and bool(s.from_spawn)
        np.testing.assert_allclose(np.asarray(obs), np.asarray(E.observe(s)))
        plats.add(round(float(s.plat_x), 2)); enemies.add(round(float(s.enemy_x), 2))
    assert len(plats) > 20 and len(enemies) > 20


def test_default_reset_is_the_real_spawn():
    s = E.reset(jax.random.PRNGKey(5))
    assert bool(s.from_spawn) and float(s.x) == L.SPAWN_X and float(s.y) == L.SPAWN_Y


def test_auto_reset_picks_random_starts_when_enabled():
    step = jax.jit(functools.partial(E.step, random_starts=True))
    xs = set()
    for i in range(40):
        s = put(31.0, 7.0).replace(rng=jax.random.PRNGKey(i))               # falls into the pit
        for _ in range(60):
            s, obs, r, d, info = step(s, NOOP)
            if bool(d):
                break
        assert bool(d) and int(s.t) == 0
        np.testing.assert_allclose(np.asarray(obs), np.asarray(E.observe(s)))   # obs of the NEW state
        xs.add(round(float(s.x), 2))
    assert len(xs) >= 5


def test_progress_bonus_pays_only_new_ground():
    step = jax.jit(functools.partial(E.step, progress_coef=0.1))
    s, total, rewards = put(12.0, 8.0), 0.0, []
    for a in [RIGHT] * 5 + [LEFT] * 5 + [RIGHT] * 5:      # 1 tile forward, back, forward again
        s, obs, r, d, info = step(s, a)
        rewards.append(float(r))
        total += float(r)
    assert min(rewards) >= 0.0                            # walking backwards never costs anything
    assert sum(rewards[5:]) == 0.0                        # and re-walking old ground pays nothing
    assert total == pytest.approx(0.1 * 1.0, abs=1e-3)
    assert float(info["max_x"]) == pytest.approx(13.0, abs=1e-3)
    assert float(s.ep_return) == 0.0                      # the logged task return has no shaping


def test_progress_bonus_is_off_by_default():
    _, log = run(put(12.0, 8.0), [RIGHT] * 5)
    assert all(l["reward"] == 0.0 for l in log)


# ---------------------------------------------------------------- the level can be finished
def _solid(r, c):
    return c < 0 or c >= L.W or (0 <= r < L.H and bool(L.SOLID[r, c]))


def _bot(s):
    """Scripted player: run right; jump at walls; wait for the ferry; hop over the enemy."""
    x, y = float(s.x), float(s.y)
    px, pdir, ex = float(s.plat_x), float(s.plat_dir), float(s.enemy_x)
    sup = bool(E._tile_support(s.x, s.y))
    if 26.0 <= x < 35.0:                                         # the 8-tile pit
        if bool(E._on_mplat(s.x, s.y, s.plat_x)) and not sup:    # riding: wait for the far end
            return RIGHT if px >= 33.6 else NOOP
        if sup and x < 30.0:                                     # at the edge: wait for the platform
            if px <= 28.3 and pdir > 0:
                return RIGHT
            return NOOP if x >= 26.3 else RIGHT
    if 43.0 <= x < 53.0 and sup and 1.0 < ex - x < 2.0:          # enemy corridor
        return JUMP_RIGHT
    if _solid(int(np.floor(y - 0.4)), int(np.floor(x + 0.7))):   # wall ahead
        return JUMP_RIGHT
    return RIGHT


def test_level_is_completable():
    """Reaches the goal: the level is feasible with these physics, the ferry and the enemy."""
    s = E.initial_state()
    for _ in range(E.MAX_STEPS):
        s, obs, r, d, info = STEP(s, _bot(s))
        if bool(d):
            assert bool(info["goal"]), "bot died or timed out"
            return
    pytest.fail("bot never finished")
