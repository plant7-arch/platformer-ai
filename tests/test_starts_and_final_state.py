"""Random starts never begin inside an enemy, and step() reports the state an episode really ended in.

Bug (0.1.0): a random start riding a moving platform with an enemy on it began inside that enemy in half of those
starts when the platform was narrow (the enemy is pushed 3 tiles away, but clipped to its patrol on the platform): on
docs/levels/03_many.txt 12.6% of all training starts, 6.4% dead within 3 steps whatever the agent did.  Needs jax.
"""
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from platformer_ai import env as E
from platformer_ai import level as L

DOCS = Path(__file__).resolve().parents[1] / "docs"
N = 4000


def training_starts(path, n=N, seed=1, **kw):
    lv = E.make_levels([L.load(str(path))])
    reset = functools.partial(E.reset, random_start=True, random_phases=True, levels=lv, **kw)
    return lv, jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(seed), n))


def overlapping(lv, s):
    """(n,) bool: the player's box overlaps an enemy (the hit test of env.step)."""
    p = E._pick(lv, 0)
    x, y = s.x[:, None], s.y[:, None]
    return np.asarray(((x + E.HW > s.enemy_x - L.ENEMY_HALF_W) & (x - E.HW < s.enemy_x + L.ENEMY_HALF_W) &
                       (y > p.enemy_y - L.ENEMY_H) & (y - E.HEIGHT < p.enemy_y) & p.enemy_on).any(1))


@pytest.mark.parametrize("name", ["levels/03_many.txt", "levels/02_enemies_and_ferry.txt", "builtin.txt"])
def test_no_random_start_begins_inside_an_enemy(name):
    lv, s = training_starts(DOCS / name)
    assert not overlapping(lv, s).any()


def test_rider_level_still_gets_platform_starts():
    """03_many has a rider on a 2-tile platform: the starts on that platform that hit the rider become spawn starts,
    the other random starts stay (S points and the safe platform starts)."""
    lv, s = training_starts(DOCS / "levels" / "03_many.txt")
    p = E._pick(lv, 0)
    on_plat = np.isin(np.round(np.asarray(s.y), 3), np.round(np.asarray(p.plat_top), 3)) & ~np.asarray(s.from_spawn)
    away = 1.0 - float(np.asarray(s.from_spawn).mean())
    assert on_plat.mean() > 0.05                     # riding starts still happen
    assert 0.3 < away < 0.5                          # ~0.5 random starts, minus the ~1/8 that fell back to the spawn


def test_level_without_the_problem_keeps_its_share_of_random_starts():
    """The fallback only touches starts inside an enemy: on the built-in level (the enemy can always be pushed away)
    half of the starts are still random ones, as spawn_prob = 0.5 says."""
    _, s = training_starts(DOCS / "builtin.txt", n=8000)
    assert abs(float(np.asarray(s.from_spawn).mean()) - E.SPAWN_PROB) < 0.02


def test_unsafe_start_is_the_hit_test():
    lv = E.make_levels([L.load(str(DOCS / "levels" / "03_many.txt"))])
    p = E._pick(lv, 0)
    ex = np.asarray(p.enemy_x0)
    i = 1                                             # a ground enemy
    x, y = float(ex[i]), float(p.enemy_y[i])
    assert bool(E._unsafe_start(x, y, p.enemy_x0, p))                                    # standing in it
    assert bool(E._unsafe_start(x + E.HW + L.ENEMY_HALF_W - 0.01, y, p.enemy_x0, p))     # just touching
    assert not bool(E._unsafe_start(x + E.HW + L.ENEMY_HALF_W + 0.01, y, p.enemy_x0, p))  # just clear
    assert not bool(E._unsafe_start(x, y - 3.0, p.enemy_x0, p))                          # above it


def test_final_state_is_where_the_episode_ended():
    """step() auto-resets; info["final_state"] is the state before that (play and the viewer draw it)."""
    lv = E.make_levels([L.load(str(DOCS / "levels" / "01_walk.txt"))])
    step = jax.jit(functools.partial(E.step, levels=lv))
    s = E.reset(levels=lv)
    for _ in range(200):
        s, _, _, done, info = step(s, 2)                       # run right into the flag
        if bool(done):
            break
    assert bool(info["goal"])
    fin = info["final_state"]
    assert int(s.t) == 0 and float(s.x) == pytest.approx(0.5)                      # the next episode
    assert float(fin.x) > 10.0 and int(fin.t) > 0 and float(fin.ep_coins) == 1.0   # on the flag, with the coin
    assert float(fin.ep_return) == pytest.approx(float(info["ep_return"]))


def test_final_state_matches_the_state_when_not_done():
    s = E.reset(jax.random.PRNGKey(0))
    s2, _, _, done, info = E.step(s, 0)
    assert not bool(done)
    for a, b in zip(jax.tree_util.tree_leaves(s2), jax.tree_util.tree_leaves(info["final_state"]), strict=True):
        assert bool(jnp.all(a == b))
