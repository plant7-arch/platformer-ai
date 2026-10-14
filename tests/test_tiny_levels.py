"""The real environment (reset / step / observe, auto-reset, random starts and phases) on tiny and odd levels.

The level is baked into the environment when `env.py` is imported, so each level runs in its own Python process,
selected with PLATFORMER_LEVEL (see level.py).  Needs jax; runs on the CPU so it never touches the GPU.
About 15-30 s per level (compilation).  Run only these with:  uv run pytest tests/test_tiny_levels.py -v
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from platformer_ai import levelcheck

SRC = Path(__file__).resolve().parents[1] / "src"


def rows(*r):
    return "\n".join(r) + "\n"


LEVELS = {
    "one row 1x12": rows("PE.MM......."),
    "two rows 2x8": rows("PE......", "MM......"),
    "no ground at all, with a goal 3x14": rows("P.E.MM.......G", "..............", ".............."),
    "tall 30x14": rows("P.E.MM........", *[".............."] * 28, "##############"),
    "floor 4x14": rows("..............", "P.E.MM........", "##############", "##############"),
    "wide 5x120": rows("P.E.MM" + "." * 114, "." * 120, "." * 120, "." * 120, "#" * 120),
}

CHILD = """
import functools
import jax
import jax.numpy as jnp
import numpy as np
from platformer_ai import env as E, level as L

N, STEPS = 8, 100
episodes = 0
for random_starts, random_phases in ((False, False), (True, True)):
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    state = jax.vmap(functools.partial(E.reset, random_start=random_starts, random_phases=random_phases))(keys)
    step = jax.jit(jax.vmap(functools.partial(E.step, random_starts=random_starts, random_phases=random_phases,
                                              progress_coef=0.1)))
    obs = jax.vmap(E.observe)(state)
    assert obs.shape == (N, E.OBS_DIM), obs.shape
    key = jax.random.PRNGKey(1)
    for t in range(STEPS):
        key, k = jax.random.split(key)
        action = jax.random.randint(k, (N,), 0, E.N_ACTIONS)
        state, obs, reward, done, info = step(state, action)
        assert obs.shape == (N, E.OBS_DIM) and bool(jnp.isfinite(obs).all()), "bad observation"
        assert bool(jnp.isfinite(reward).all()) and bool(jnp.isfinite(info["final_obs"]).all()), "bad reward / final_obs"
        assert state.coins_alive.shape == (N, L.N_COINS)
        x, y = np.asarray(state.x), np.asarray(state.y)
        assert np.isfinite(x).all() and np.isfinite(y).all(), "non-finite position"
        assert (x > -1.0).all() and (x < L.W + 1.0).all(), ("x out of the map", x)
        assert (y < L.H + 3.0).all(), ("y below the map and not reset", y)
        episodes += int(np.asarray(done).sum())
print("OK", L.H, "x", L.W, "episodes finished:", episodes)
"""


@pytest.mark.parametrize("name", list(LEVELS))
def test_the_level_is_valid(name):
    rep = levelcheck.check(LEVELS[name])
    assert rep.ok, [str(e) for e in rep.errors]              # warnings (no ground, no coins...) are fine here


@pytest.mark.parametrize("name", list(LEVELS))
def test_environment_runs_on_the_level(name):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        f.write(LEVELS[name])
        f.close()
        env = {**os.environ, "PLATFORMER_LEVEL": f.name, "PYTHONPATH": str(SRC), "JAX_PLATFORMS": "cpu"}
        r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600)
    finally:
        os.unlink(f.name)
    assert r.returncode == 0, f"{name}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr[-3000:]}"
    assert r.stdout.strip().startswith("OK")
