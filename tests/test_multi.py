"""Several enemies and moving platforms (or none): behaviour of the real environment on small purpose-built levels.

The level is baked into env.py when it is imported, so every scenario runs in its own Python process, with the level
chosen by PLATFORMER_LEVEL (see level.py).  Needs jax; runs on the CPU.  About 10-30 s per scenario (compilation).
Run only these with:  uv run pytest tests/test_multi.py -v
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

PRELUDE = """
import functools
import jax
import jax.numpy as jnp
import numpy as np
from platformer_ai import env as E, level as L

PR, PC, PATCH = E.PR, E.PC, E.PATCH[0] * E.PATCH[1]
def channel(obs, k):                                   # (9, 17) window, channels: solid, one-way, coin, goal, enemy
    return np.asarray(obs[k * PATCH:(k + 1) * PATCH]).reshape(E.PATCH)
def scalars(obs):
    return np.asarray(obs[-E.N_SCALARS:])
def put(s, **kw):                                      # move things around in a state
    return s.replace(**{k: jnp.asarray(v, jnp.float32) for k, v in kw.items()})
def run(s, n, action=0):
    for _ in range(n):
        s, obs, r, d, info = E.step(s, action)
    return s, obs, r, d, info
"""

# ---- levels --------------------------------------------------------------------------------------------------------
NOTHING = """\
....................
P.....o.....G.......
####################
"""

TWO_ENEMIES = """\
....................
....................
....................
....................
P..E.........E.....o
####################

enemy 5 4 from 2 to 6
enemy 5 14 from 10 to 17
"""

TWO_PLATFORMS = """\
....................
....................
..........MM........
..MMM...............
P...................
####################
"""

PATHS = """\
....................
....................
................MM..
....................
P....E..............
####################

platform 3 17 to 5
enemy 5 6 from 3 to 12
"""

# ---- scenarios: (level, code run after the prelude; it must print OK) ----------------------------------------------
SCENARIOS = {
    "no enemies, no platforms": (NOTHING, """
assert (L.N_ENEMIES, L.N_PLATS, L.N_COINS) == (0, 0, 1)
s = E.reset()
assert s.enemy_x.shape == (0,) and s.plat_x.shape == (0,) and s.coins_alive.shape == (1,)
obs = E.observe(s)
assert obs.shape == (E.OBS_DIM,) == (778,)
assert scalars(obs)[7:].tolist() == [2.0, 0.0, 0.0, 2.0, 0.0, 0.0]          # "nothing there": far to the right, still
assert channel(obs, 4).sum() == 0 and channel(obs, 1).sum() == 0
total, goal = 0.0, False
for t in range(200):
    s, obs, r, d, info = E.step(s, 2)                                     # just run right
    total += float(r)
    if bool(d):
        goal = bool(info["goal"])
        break
assert goal and total == 6.0 and t < 100, (goal, total, t)               # the coin, then the goal
# options that touch enemies / platforms must also work with none
for rs, rp in ((True, False), (False, True), (True, True)):
    st = E.reset(jax.random.PRNGKey(3), random_start=rs, random_phases=rp)
    st, obs, r, d, info = E.step(st, 0, random_starts=rs, random_phases=rp)
    assert bool(jnp.isfinite(obs).all())
print("OK")
"""),

    "two enemies": (TWO_ENEMIES, """
assert L.N_ENEMIES == 2 and L.N_PLATS == 0
assert L.ENEMY_X.tolist() == [3.5, 13.5] and L.ENEMY_Y.tolist() == [5.0, 5.0]
s = E.reset()
assert s.enemy_x.tolist() == [3.5, 13.5] and s.enemy_dir.tolist() == [1.0, 1.0]
# both enemies are in the picture of a player standing between them
s2 = put(s, x=8.5, y=5.0)
enemy = channel(E.observe(s2), 4)
assert enemy.sum() == 2 and enemy[PR, PC + 3 - 8] == 1 and enemy[PR, PC + 13 - 8] == 1, enemy     # window cols = tile - 8 + PC
# touching either one kills, in the same way
for x in (3.5, 13.5):
    s3, obs, r, d, info = E.step(put(s, x=x, y=5.0), 0)
    assert float(r) == -1.0 and bool(d) and bool(info["terminated"]) and not bool(info["goal"]), x
# between them it is safe
s4, obs, r, d, info = E.step(put(s, x=8.5, y=5.0), 0)
assert float(r) == 0.0 and not bool(d)
# they walk independently: after 30 steps the left one is 2 tiles on; the right one reached its own limit (x=17) and turned
s5 = put(s, x=8.5, y=5.0, enemy_x=[3.5, 16.9])
s5, *_ = run(s5, 30)
assert s5.enemy_dir.tolist() == [1.0, -1.0], s5.enemy_dir
assert abs(float(s5.enemy_x[0]) - 5.5) < 1e-3 and float(s5.enemy_x[1]) < 16.9
# the scalars describe the NEAREST enemy
s6 = put(s, x=6.5, y=5.0, enemy_dir=[1.0, -1.0])
sc = scalars(E.observe(s6))
assert sc[7:10].tolist() == pytest.approx([-3.0 / 8, 0.0, 1.0])            # left one: 3 tiles away
s7 = put(s, x=10.5, y=5.0, enemy_dir=[1.0, -1.0])
sc = scalars(E.observe(s7))
assert sc[7:10].tolist() == pytest.approx([3.0 / 8, 0.0, -1.0])            # right one
# random phases: both enemies start somewhere in their patrol, never on top of the player
xs = []
for i in range(60):
    st = E.reset(jax.random.PRNGKey(i), random_phases=True)
    ex = np.asarray(st.enemy_x)
    assert (ex >= L.ENEMY_MIN).all() and (ex <= L.ENEMY_MAX).all()
    assert (np.abs(ex - float(st.x)) >= 1.5 - 1e-3).all(), (ex, float(st.x))
    xs.append(ex)
xs = np.array(xs)
assert len(set(xs[:, 0].round(2))) > 30 and len(set(xs[:, 1].round(2))) > 30
assert not np.allclose(xs[:, 0], xs[:, 1])                                   # not in lockstep
print("OK")
"""),

    "two platforms": (TWO_PLATFORMS, """
assert L.N_PLATS == 2 and L.N_ENEMIES == 0 and L.N_COINS == 2
assert L.PLAT_TOP.tolist() == [2.0, 3.0] and L.PLAT_HALF.tolist() == [1.0, 1.5]       # row by row: the upper one first
assert L.PLAT_X_MIN.tolist() == [11.0, 3.5] and L.PLAT_X_MAX.tolist() == [17.0, 9.5]
s = E.reset()
assert s.plat_x.tolist() == [11.0, 3.5] and s.coins_alive.tolist() == [True, True]
# riding the lower platform (index 1) carries the player 3.2 * 15 / 30 = 1.6 tiles; the upper one is not involved
s1, obs, r, d, info = run(put(s, x=3.5, y=3.0), 15)
assert float(s1.y) == 3.0 and abs(float(s1.x) - 5.1) < 1e-3, (float(s1.x), float(s1.y))
assert scalars(obs)[2] == 1.0                                                         # "on floor"
# the upper platform carries the player just as well, and the other one does not move him
s2, *_ = run(put(s, x=11.0, y=2.0), 15)
assert float(s2.y) == 2.0 and abs(float(s2.x) - 12.6) < 1e-3
# both platforms are drawn into the one-way channel
ow = channel(E.observe(put(s, x=8.5, y=5.0)), 1)
assert ow[PR + 3 - 4, PC + 3 - 8] == 1 and ow[PR + 2 - 4, PC + 11 - 8] == 1, ow
# each platform has its own rider coin: standing on the lower platform takes only the second coin
s3, obs, r, d, info = E.step(put(s, x=3.5, y=3.0), 0)
assert float(r) == 1.0 and s3.coins_alive.tolist() == [True, False]
s4, obs, r, d, info = E.step(put(s, x=11.0, y=2.0), 0)
assert float(r) == 1.0 and s4.coins_alive.tolist() == [False, True]
# a jumper falling onto the lower platform lands on it (not on the floor below, not through it)
s5, *_ = run(put(s, x=3.5, y=2.5), 8)
assert float(s5.y) == 3.0 and float(s5.vy) == 0.0
# the nearest platform is the one in the scalars
sc = scalars(E.observe(put(s, x=3.5, y=4.0)))
assert sc[10:13].tolist() == pytest.approx([0.0, -0.25, 1.0])                  # lower platform, 1 tile above
sc = scalars(E.observe(put(s, x=14.0, y=4.0)))
assert sc[10] == pytest.approx((11.0 - 14.0) / 8) and sc[11] == pytest.approx(-2.0 / 4)  # the upper one
# platforms keep their own direction at the ends of their sweeps
s6, *_ = run(put(s, plat_x=[16.9, 3.6]), 3)
assert s6.plat_dir.tolist() == [-1.0, 1.0]
# random phases / ferry starts: both platforms get used, positions stay inside their own range
rides = set()
for i in range(120):
    st = E.reset(jax.random.PRNGKey(i), random_start=True, random_phases=True, spawn_prob=0.0)
    px = np.asarray(st.plat_x)
    assert (px >= L.PLAT_X_MIN).all() and (px <= L.PLAT_X_MAX).all()
    if float(st.y) in (2.0, 3.0):
        k = int(np.where(L.PLAT_TOP == float(st.y))[0][0])
        assert abs(float(st.x) - px[k]) < 1e-4                                  # starts on top of its platform
        rides.add(k)
assert rides == {0, 1}, rides
print("OK")
"""),

    "paths: start/stop x, walls play no part": (PATHS, """
assert L.PLAT_START_X.tolist() == [17.0] and L.PLAT_STOP_X.tolist() == [5.0]
assert L.PLAT_X_MIN.tolist() == [5.0] and L.PLAT_X_MAX.tolist() == [17.0] and L.PLAT_DIR0.tolist() == [-1.0]
assert L.ENEMY_MIN.tolist() == [3.0] and L.ENEMY_MAX.tolist() == [12.0] and L.ENEMY_X.tolist() == [5.5]
s = E.reset()
assert s.plat_x.tolist() == [17.0] and s.plat_dir.tolist() == [-1.0]               # starts at its right end, heads left
s1, *_ = run(s, 30)
assert abs(float(s1.plat_x[0]) - (17.0 - 3.2)) < 1e-3 and s1.plat_dir.tolist() == [-1.0]
# it turns round at ITS stop x (5), not after a fixed distance
s2, *_ = run(put(s, plat_x=[5.1]), 1)
assert s2.plat_dir.tolist() == [1.0] and float(s2.plat_x[0]) < 5.0
# the enemy turns round at its own limits in an open field: no wall anywhere near x=12 or x=3
s3, *_ = run(put(s, enemy_x=[11.95]), 2)
assert s3.enemy_dir.tolist() == [-1.0] and 12.0 < float(s3.enemy_x[0]) + 0.07
s4, *_ = run(put(s, enemy_x=[3.05], enemy_dir=[-1.0]), 2)
assert s4.enemy_dir.tolist() == [1.0]
# it keeps going from limit to limit for as long as you like
t, lo, hi = s, 99.0, 0.0
for _ in range(300):
    t, *_ = E.step(t, 0)
    lo, hi = min(lo, float(t.enemy_x[0])), max(hi, float(t.enemy_x[0]))
    if bool(_[2]):
        break
assert lo >= 3.0 - 0.07 and hi <= 12.0 + 0.07 and hi - lo > 8.5, (lo, hi)
# random phases stay inside the stated ranges
for i in range(40):
    st = E.reset(jax.random.PRNGKey(i), random_phases=True)
    assert 5.0 <= float(st.plat_x[0]) <= 17.0 and 3.0 <= float(st.enemy_x[0]) <= 12.0
print("OK")
"""),

    "one enemy, no platform": (
        "P....E..........\n################\n", """
assert (L.N_ENEMIES, L.N_PLATS) == (1, 0) and E.OBS_DIM == 778
obs = E.observe(E.reset())
assert scalars(obs)[10:13].tolist() == [2.0, 0.0, 0.0] and scalars(obs)[7] != 2.0
print("OK")
"""),
}


def run_child(level_text: str, code: str):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        f.write(level_text)
        f.close()
        env = {**os.environ, "PLATFORMER_LEVEL": f.name, "JAX_PLATFORMS": "cpu",
               "PYTHONPATH": os.pathsep.join([str(SRC)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
        script = "import pytest\n" + PRELUDE + code
        return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=900)
    finally:
        os.unlink(f.name)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(name):
    level, code = SCENARIOS[name]
    r = run_child(level, code)
    assert r.returncode == 0, f"{name}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr[-3500:]}"
    assert r.stdout.strip().endswith("OK")


# ---- drawing: the ASCII renderer always, the pixel renderer when Pillow and the assets are there ---------------------
MANY = """\
........................................
..MM....................................
........................................
..........MMM...........................
........................................
P..E.....E....E.......................G.
########################################

enemy 6 4 from 2 to 4
enemy 6 10 from 7 to 11.5
enemy 6 15 from 13 to 16
"""

RENDER = """
from pathlib import Path
from platformer_ai import render, scene, viewlogic as V
state = E.reset()
txt = render.frame(state)
assert txt.count("X") == L.N_ENEMIES and txt.count("M") == int(sum(2 * L.PLAT_HALF)), txt
assert txt.count("@") == 1
assets = Path("@ASSETS@")
if (assets / "sprites" / "knight.png").is_file():
    a = scene.load_assets(assets)
    cam = V.Camera(); cam.snap(*V.to_px(L.SPAWN_X, L.SPAWN_Y))
    def frame(enemy_x, plat_x):
        return scene.compose(a, cam.top_left(), player_xy=(L.SPAWN_X, L.SPAWN_Y),
                             player_frame=V.ANIMS["player"]["idle"]["frames"][0], player_flip=False,
                             enemy_x=enemy_x, enemy_flip=[False] * len(enemy_x),
                             enemy_frame=V.ANIMS["enemy"]["walk"]["frames"][0],
                             coin_frame=V.ANIMS["coin"]["spin"]["frames"][0], coins_alive=[True] * L.N_COINS, plat_x=plat_x)
    base = frame(list(L.ENEMY_X), list(L.PLAT_X_MIN))
    assert base.shape == (V.SCREEN_H, V.SCREEN_W, 3)
    near = frame([L.SPAWN_X + 2] * L.N_ENEMIES, list(L.PLAT_X_MIN))                 # every slime next to the knight
    assert np.array_equal(base, near) == (L.N_ENEMIES == 0)
    if L.N_ENEMIES > 1:                                                              # each slime is drawn on its own
        one = frame([L.SPAWN_X + 2] + list(L.ENEMY_X[1:]), list(L.PLAT_X_MIN))
        assert not np.array_equal(one, near)
    moved = frame(list(L.ENEMY_X), [x + 2 for x in L.PLAT_X_MIN])
    assert np.array_equal(base, moved) == (L.N_PLATS == 0)                           # the whole level is on screen
print("OK")
"""


@pytest.mark.parametrize("name,level", [("none", NOTHING), ("many", MANY)])
def test_renderers_on_levels_with_none_and_many(name, level):
    pytest.importorskip("PIL")
    root = Path(__file__).resolve().parents[1]
    r = run_child(level, RENDER.replace("@ASSETS@", str(root / "assets")))
    assert r.returncode == 0, f"{name}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr[-3500:]}"
    assert r.stdout.strip().endswith("OK")
