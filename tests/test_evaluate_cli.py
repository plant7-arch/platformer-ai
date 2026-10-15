"""python -m platformer_ai.evaluate: batching gives the same numbers, episode counts, the printout and the CSV."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from platformer_ai import env as E
from platformer_ai import evaluate as EV
from platformer_ai import level as L
from platformer_ai.ppo import ActorCritic

LEVELS = ["docs/levels/"]
RIGHT = 2


def run_right(params, obs, key):
    return jnp.full((obs.shape[0],), RIGHT, jnp.int32)


def test_batches_do_not_change_the_numbers():
    arrays = L.load_all(LEVELS)
    key = jax.random.PRNGKey(0)
    one = EV.evaluate_levels(run_right, None, arrays, 4, key, batch=4096, random_phases=False)
    split = EV.evaluate_levels(run_right, None, arrays, 4, key, batch=2, random_phases=False)   # one level per call
    for k in ("goal", "ret", "coins", "max_x", "length"):
        np.testing.assert_allclose(one[k], split[k], rtol=1e-6)
    assert one["goal"][0] == 1.0                     # 01_walk: running right reaches the flag


def test_episode_counts():
    arrays = L.load_all(LEVELS)
    r = EV.evaluate_levels(run_right, None, arrays, 10, jax.random.PRNGKey(1), batch=8)   # 2 per level per call
    assert r["n"].tolist() == [10, 10, 10]
    r = EV.evaluate_levels(run_right, None, arrays, 7, jax.random.PRNGKey(1), batch=9)    # 3 per call -> 3 calls of 3
    assert r["n"].tolist() == [9, 9, 9]


def test_command_line(tmp_path, capsys):
    net = ActorCritic(E.N_ACTIONS, 128)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((E.OBS_DIM,), jnp.float32))
    p = tmp_path / "params.msgpack"
    p.write_bytes(serialization.to_bytes(params))
    csv = tmp_path / "out.csv"
    assert EV.main(["--params", str(p), "--levels", *LEVELS, "--episodes", "3", "--all", "--csv", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "3 levels x 3 episodes, greedy, random phases" in out and "all           3 levels  goal" in out
    assert "every level, worst first" in out and "95%:" in out
    rows = csv.read_text().splitlines()
    assert rows[0] == "level,tier,episodes,goal,return,coins,max_x,length" and len(rows) == 4
    assert EV.main(["--params", str(tmp_path / "missing.msgpack"), "--levels", *LEVELS]) == 2
