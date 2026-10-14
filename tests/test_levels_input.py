"""The level as an input: several levels of different sizes in one env.Level, padded to one shape.

Runs in-process (no PLATFORMER_LEVEL needed): levels are data, built from text with level.build().
"""
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from platformer_ai import cli
from platformer_ai import env as E
from platformer_ai import level as L
from platformer_ai import levelcheck
from platformer_ai.evaluate import make_eval

NOOP, LEFT, RIGHT, JUMP, JUMP_LEFT, JUMP_RIGHT = range(6)

# small: 3 x 12, two coins, nothing that moves, no 'S'
A_TEXT = """\
............
P....o.o...G
############
"""
# bigger: 7 x 30, two enemies, a moving platform with a coin stuck on it, a pit, one 'S'
B_TEXT = """\
..............................
..............................
.............o................
............MM................
..P.......E.......S.....E....G
##############....############
##############....############
"""
A, B = L.build(A_TEXT, "a.txt"), L.build(B_TEXT, "b.txt")
BOTH = E.make_levels([A, B])


def step_fn(levels, **kw):
    return jax.jit(functools.partial(E.step, levels=levels, **kw))


def put(s, **kw):
    return s.replace(**{k: jnp.asarray(v, jnp.float32) for k, v in kw.items()})


def lv_of(levels, i):
    return E._pick(levels, i)


def scalars(obs):
    return np.asarray(obs[-E.N_SCALARS:])


def channel(obs, k):
    n = E.PATCH[0] * E.PATCH[1]
    return np.asarray(obs)[k * n:(k + 1) * n].reshape(E.PATCH)


# ---- building the padded level data ----------------------------------------------------------------------------------
def test_make_levels_pads_everything_to_the_largest_level():
    assert E.n_levels(BOTH) == 2
    assert BOTH.solid.shape == (2, 7, 30)
    assert BOTH.solid_pad.shape == (2, 7 + 2 * E.PR + E.ABOVE, 30 + 2 * E.PC)
    assert BOTH.enemy_on.shape == (2, 2) and BOTH.plat_on.shape == (2, 1) and BOTH.coin_on.shape == (2, 2)
    assert np.asarray(BOTH.h).tolist() == [3, 7] and np.asarray(BOTH.w).tolist() == [12, 30]
    assert np.asarray(BOTH.enemy_on).tolist() == [[False, False], [True, True]]
    assert np.asarray(BOTH.plat_on).tolist() == [[False], [True]]
    assert np.asarray(BOTH.coin_on).tolist() == [[True, True], [True, False]]      # B has one coin, one empty slot
    assert np.asarray(BOTH.n_starts).tolist() == [1, 1] and np.asarray(BOTH.n_plats).tolist() == [0, 1]
    assert float(BOTH.start_x[0, 0]) == A.SPAWN_X                                   # no 'S': the spawn is the start


def test_the_side_wall_of_a_narrow_level_is_at_its_own_width():
    pad = np.asarray(BOTH.solid_pad[0])
    assert pad[:, :E.PC].all() and pad[:, E.PC + 12:].all()          # walls left of col 0 and right of col 11
    assert not pad[E.ABOVE + E.PR + 1, E.PC + 11]                     # col 11 (the goal) is open
    lv = lv_of(BOTH, 0)
    for r in (-3, 0, 1, 5):                                           # also above and below the map
        assert bool(E._solid(jnp.int32(r), jnp.int32(12), lv))
        assert bool(E._solid(jnp.int32(r), jnp.int32(-1), lv))
    assert not bool(E._solid(jnp.int32(1), jnp.int32(11), lv))
    assert not bool(E._solid(jnp.int32(4), jnp.int32(3), lv))         # row 4 is outside A (3 rows): open pit


@pytest.mark.parametrize("levels", [E.make_levels([A]), BOTH])
def test_above_the_top_of_the_map_the_window_shows_sky(levels):
    """Jumping above row 0: the window must not wrap round to the bottom rows of the map."""
    for y, floor_row in ((0.0, 7), (-1.0, 8), (2.0, 5)):            # window row of A's floor (map row 2)
        s = put(E.reset(levels=levels, level=0), x=0.9, y=y)
        solid = channel(E.observe(s, levels), 0)
        assert solid[:, :E.PC].all()                                 # wall left of the map (x = 0.9: col 0 is centre)
        assert solid[floor_row, E.PC:].all(), (y, solid.astype(int))
        assert not solid[:floor_row, E.PC:].any(), (y, solid.astype(int))     # sky above the floor


def test_the_default_levels_are_the_level_of_level_py():
    assert E.n_levels(E.DEFAULT_LEVELS) == 1
    assert np.array_equal(np.asarray(E.DEFAULT_LEVELS.solid[0]), L.SOLID)
    assert int(E.initial_state().level) == 0
    assert np.array_equal(np.asarray(E.observe(E.initial_state())), np.asarray(E.RESET_OBS))


# ---- a level in a set behaves exactly like on its own ----------------------------------------------------------------
@pytest.mark.parametrize("i,arrays", [(0, A), (1, B)])
def test_a_level_in_a_set_plays_exactly_like_on_its_own(i, arrays):
    alone = E.make_levels([arrays])
    step_set, step_alone = step_fn(BOTH, keep_level=True), step_fn(alone)
    s_set, s_alone = E.reset(levels=BOTH, level=i), E.reset(levels=alone)
    rng = np.random.default_rng(i)
    k, n = alone.plat_on.shape[1], alone.enemy_on.shape[1]
    for t in range(300):
        a = int(rng.choice(6, p=[.05, .05, .3, .1, .1, .4]))
        y_before = float(s_alone.y)
        s_set, o_set, r_set, d_set, _ = step_set(s_set, a)
        s_alone, o_alone, r_alone, d_alone, _ = step_alone(s_alone, a)
        assert float(r_set) == float(r_alone) and bool(d_set) == bool(d_alone), t
        assert int(s_set.level) == i
        for f in ("x", "y", "vx", "vy"):
            assert float(getattr(s_set, f)) == float(getattr(s_alone, f)), (t, f)
        assert np.array_equal(np.asarray(s_set.plat_x)[:k], np.asarray(s_alone.plat_x))
        assert np.array_equal(np.asarray(s_set.enemy_x)[:n], np.asarray(s_alone.enemy_x))
        if not bool(d_set) and max(y_before, float(s_alone.y)) < arrays.H + 0.4:   # (below a short map the window
            o_set, o_alone = np.asarray(o_set), np.asarray(o_alone)                 #  of the taller set shows more)
            # the grids are exactly equal; the scalars may differ in the last float bit: with one level XLA folds
            # x / width into x * (1 / width), with several the width is looked up and really divided
            g = E.OBS_DIM - E.N_SCALARS                                        # grid part
            assert np.array_equal(o_set[:g], o_alone[:g]), (t, np.flatnonzero(o_set[:g] != o_alone[:g]))
            np.testing.assert_allclose(o_set[g:], o_alone[g:], rtol=1e-6, atol=1e-7, err_msg=str(t))


# ---- padding slots are inert -----------------------------------------------------------------------------------------
def test_a_level_without_enemies_or_platforms_sees_none_in_a_set_that_has_them():
    s = E.reset(levels=BOTH, level=0)
    obs = E.observe(s, BOTH)
    assert obs.shape == (E.OBS_DIM,)
    assert scalars(obs)[7:10].tolist() == list(E.NONE_NEAR)          # nearest enemy: none
    assert scalars(obs)[10:13].tolist() == list(E.NONE_NEAR)         # nearest platform: none
    assert not channel(obs, 4).any()                                 # enemy channel empty
    assert np.asarray(s.coins_alive).tolist() == [True, True]


def test_an_empty_coin_slot_starts_collected_and_never_counts():
    s = E.reset(levels=BOTH, level=1)
    assert np.asarray(s.coins_alive).tolist() == [True, False]
    s = put(s, x=13.0, y=3.0)                                         # standing on the platform under its coin
    s, obs, r, d, info = step_fn(BOTH)(s, NOOP)
    assert float(r) == pytest.approx(E.R_COIN) and not np.asarray(s.coins_alive).any()


def test_death_is_below_the_level_s_own_height():
    s = put(E.reset(levels=BOTH, level=0), x=3.0, y=4.2)              # below A (3 rows), inside the 7-row padding
    s, obs, r, done, info = step_fn(BOTH)(s, NOOP)
    assert bool(done) and bool(info["terminated"]) and float(r) == pytest.approx(E.R_DEATH)


def test_the_progress_scalar_uses_each_level_s_width():
    for i, a in ((0, A), (1, B)):
        s = E.reset(levels=BOTH, level=i)
        assert scalars(E.observe(s, BOTH))[5] == pytest.approx(a.SPAWN_X / a.W)


# ---- which level an environment plays --------------------------------------------------------------------------------
def test_reset_without_a_level_picks_them_at_random():
    seen = {int(E.reset(jax.random.PRNGKey(k), levels=BOTH).level) for k in range(40)}
    assert seen == {0, 1}
    for k in range(10):                                               # and it starts at that level's spawn
        s = E.reset(jax.random.PRNGKey(k), levels=BOTH)
        assert float(s.x) == (A, B)[int(s.level)].SPAWN_X


def test_auto_reset_switches_levels_unless_told_to_keep_them():
    for keep, expect in ((False, {0, 1}), (True, {1})):
        step = step_fn(BOTH, keep_level=keep)
        s, seen = E.reset(levels=BOTH, level=1), set()
        for t in range(40):
            s = put(s, y=20.0)                                         # far below the map: dies at once
            s, obs, r, done, info = step(s, NOOP)
            assert bool(done)
            seen.add(int(s.level))
            assert np.array_equal(np.asarray(obs), np.asarray(E.observe(s, BOTH)))   # obs of the new level
        assert seen == expect, keep


def test_random_starts_use_each_level_s_own_points():
    step = step_fn(BOTH, random_starts=True, random_phases=True, spawn_prob=0.0)
    s = E.reset(levels=BOTH, level=0)
    for t in range(60):
        s = put(s, y=20.0)
        s, obs, r, done, info = step(s, NOOP)
        a = (A, B)[int(s.level)]
        if int(s.level) == 0:                                        # A: no 'S', no platform -> the spawn point
            assert float(s.x) == A.SPAWN_X and float(s.y) == A.SPAWN_Y
        else:                                                         # B: its 'S' or riding its platform
            assert (float(s.x), float(s.y)) == (18.5, 5.0) or float(s.y) == 3.0, (float(s.x), float(s.y))
        assert 0 < float(s.x) < a.W


# ---- evaluation splits the environments over the levels --------------------------------------------------------------
def test_evaluation_reports_every_level():
    run_right = lambda params, obs, key: jnp.full((obs.shape[0],), RIGHT, jnp.int32)     # noqa: E731
    m = jax.device_get(make_eval(run_right, n_envs=3, levels=BOTH)(None, jax.random.PRNGKey(0)))
    p = m["per_level"]
    assert np.asarray(p["goal"]).tolist() == [1.0, 0.0]                # A: walks to the goal; B: runs into the enemy
    assert np.asarray(p["ret"]).tolist() == pytest.approx([2 * E.R_COIN + E.R_GOAL, E.R_DEATH])
    assert np.asarray(p["coins"]).tolist() == [2.0, 0.0]
    assert float(m["goal"]) == pytest.approx(0.5) and float(m["finished"]) == 1.0


# ---- files, folders, command line ------------------------------------------------------------------------------------
def test_load_all_reads_files_and_folders(tmp_path):
    d = tmp_path / "levels"
    d.mkdir()
    (d / "b.txt").write_text(B_TEXT)
    (d / "a.txt").write_text(A_TEXT)
    (d / "notes.md").write_text("not a level")
    got = L.load_all([str(d)])
    assert [os.path.basename(a.name) for a in got] == ["a.txt", "b.txt"]          # sorted, only *.txt
    assert [a.W for a in got] == [12, 30]
    assert len(L.load_all([str(d), str(d / "a.txt")])) == 3


def test_load_all_names_every_broken_file(tmp_path):
    (tmp_path / "ok.txt").write_text(A_TEXT)
    (tmp_path / "two_p.txt").write_text(A_TEXT.replace("....o", "P...o"))
    with pytest.raises(levelcheck.LevelError) as e:
        L.load_all([str(tmp_path), str(tmp_path / "missing.txt")])
    msg = str(e.value)
    assert "two_p.txt" in msg and "spawn-count" in msg and "missing.txt" in msg and "ok.txt" not in msg
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(levelcheck.LevelError, match="no level files"):
        L.load_all([str(empty)])


def test_levelcheck_checks_several_files(tmp_path, capsys):
    (tmp_path / "a.txt").write_text(A_TEXT)
    (tmp_path / "bad.txt").write_text(A_TEXT.replace("G", "x"))
    assert levelcheck.main([str(tmp_path / "a.txt")]) == 0
    capsys.readouterr()
    assert levelcheck.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "bad.txt" in out and "unknown-symbol" in out and "2 levels, 1 with errors" in out


def test_level_argument_sets_the_level_file(monkeypatch):
    # a private copy of the environment: the variable set here must not leak into later tests (or their subprocesses)
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "PLATFORMER_LEVEL"})
    assert cli.level_from_argv(["--params", "x"]) is None and "PLATFORMER_LEVEL" not in os.environ
    assert cli.level_from_argv(["--level", "my.txt", "--human"]) == "my.txt"
    assert os.environ["PLATFORMER_LEVEL"] == "my.txt"
    assert cli.level_from_argv(["--level=other.txt"]) == "other.txt"
    assert os.environ["PLATFORMER_LEVEL"] == "other.txt"
