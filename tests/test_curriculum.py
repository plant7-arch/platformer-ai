"""Adaptive curriculum: the level choice follows the weights, the weights follow the failures, it is off by default,
and the training script shows and saves it."""
import jax
import jax.numpy as jnp
import numpy as np

from platformer_ai import env as E
from platformer_ai import level as L
from platformer_ai import ppo
from platformer_ai import train

LEVELS = E.make_levels(L.load_all(["docs/levels/"]))


def test_reset_follows_level_logits():
    p = jnp.array([0.1, 0.1, 0.8])
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)
    lv = jax.vmap(lambda k: E.reset(k, levels=LEVELS, level_logits=jnp.log(p)).level)(keys)
    share = np.bincount(np.asarray(lv), minlength=3) / 4000
    np.testing.assert_allclose(share, [0.1, 0.1, 0.8], atol=0.03)
    uniform = jax.vmap(lambda k: E.reset(k, levels=LEVELS).level)(keys)          # no logits: uniform as before
    np.testing.assert_allclose(np.bincount(np.asarray(uniform), minlength=3) / 4000, [1 / 3] * 3, atol=0.03)


def test_update_sends_resets_where_the_agent_fails():
    c = ppo.curriculum_init(3)
    assert np.allclose(c["p"], 1 / 3)
    level = jnp.array([0] * 20 + [1] * 20 + [2] * 20)
    ended = jnp.ones(60)
    reached = jnp.array([1.0] * 20 + [0.0] * 20 + [1.0] * 10 + [0.0] * 10)  # level 0 always, 1 never, 2 half
    for _ in range(10):
        c = ppo.curriculum_update(c, level, ended, reached, mix=0.3, decay=0.95)
    p, rate = np.asarray(c["p"]), np.asarray(c["rate"])
    assert rate[0] > 0.95 and rate[1] < 0.05 and 0.4 < rate[2] < 0.6
    assert p[1] > p[2] > p[0] and abs(p.sum() - 1) < 1e-6
    assert p[0] >= 0.3 / 3 - 1e-6                     # the uniform share: a mastered level is still practised
    np.testing.assert_allclose(np.asarray(c["logits"]), np.log(p), rtol=1e-6)


def test_counts_fade():
    c = ppo.curriculum_init(2)
    c = ppo.curriculum_update(c, jnp.array([0, 0]), jnp.array([1.0, 1.0]), jnp.array([0.0, 0.0]), 0.3, 0.5)
    c = ppo.curriculum_update(c, jnp.array([0]), jnp.array([0.0]), jnp.array([0.0]), 0.3, 0.5)
    assert float(c["n"][0]) == 1.0 and float(c["n"][1]) == 0.0


def test_training_with_curriculum():
    cfg = ppo.Config(num_envs=32, num_steps=32, total_timesteps=32 * 32 * 2, curriculum=True)
    init, update, _ = ppo.make_train(cfg, LEVELS)
    r = init(jax.random.PRNGKey(0))
    for _ in range(2):
        r, m = update(r)
    assert m["level_p"].shape == (3,) and abs(float(m["level_p"].sum()) - 1) < 1e-5
    assert float(r[4]["n"].sum()) >= 0
    off = ppo.make_train(ppo.Config(num_envs=8, num_steps=8, total_timesteps=64), LEVELS)
    _, m_off = off[1](off[0](jax.random.PRNGKey(0)))
    assert "level_p" not in m_off                       # off by default
    one = ppo.make_train(ppo.Config(num_envs=8, num_steps=8, total_timesteps=64, curriculum=True))
    _, m_one = one[1](one[0](jax.random.PRNGKey(0)))
    assert "level_p" not in m_one                       # a single level: nothing to choose


def test_log_lines_and_csv(tmp_path):
    names = ["gen/train/t1_0000.txt", "gen/train/t3_0001.txt", "gen/train/t3_0002.txt"]
    m = dict(level_p=np.array([0.1, 0.3, 0.6]), level_rate=np.array([0.95, 0.6, 0.2]), level_n=np.array([10, 10, 10.]))
    lines = train.curriculum_lines(m, names, detail=True)
    assert "practice: tier 1 10%, tier 3 90%" in lines[0] and "hardest 1 levels get 60% of resets" in lines[0]
    assert lines[1].startswith("        most practised: t3_0002.txt 60.0%")
    f = tmp_path / "curriculum.csv"
    train.write_curriculum_csv(f, names, dict(rate=m["level_rate"], n=m["level_n"], p=m["level_p"]))
    rows = f.read_text().splitlines()
    assert rows[0] == "level,tier,est_goal,episodes,pick_chance" and rows[3].startswith("gen/train/t3_0002.txt,3,0.2000")
