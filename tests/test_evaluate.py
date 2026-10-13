"""The fixed evaluation harness (no network needed: it takes any action function)."""
import jax
import jax.numpy as jnp
import pytest

from platformer_ai import env as E
from platformer_ai.evaluate import make_eval


def test_idle_policy_gets_exactly_the_first_coin_and_times_out():
    """Standing still from a jittered spawn: the drop always takes coin 3, then the clock runs out."""
    evaluate = make_eval(lambda params, obs, key: jnp.zeros(obs.shape[0], jnp.int32), n_envs=8)
    m = evaluate(None, jax.random.PRNGKey(0))
    assert float(m["finished"]) == 1.0                      # every episode was run to its end
    assert float(m["length"]) == E.MAX_STEPS
    assert float(m["goal"]) == 0.0
    assert float(m["ret"]) == pytest.approx(1.0)            # task return: one coin
    assert float(m["coins"]) == pytest.approx(1.0)
    assert 16.0 < float(m["max_x"]) < 19.0                  # never leaves the spawn platform


def test_always_right_policy_dies_or_times_out_but_is_counted_once():
    evaluate = make_eval(lambda params, obs, key: jnp.full((obs.shape[0],), E.N_ACTIONS - 1, jnp.int32),
                         n_envs=8)                          # run right + jump forever
    m = evaluate(None, jax.random.PRNGKey(1))
    assert float(m["finished"]) == 1.0
    assert 0 < float(m["length"]) < E.MAX_STEPS             # falls into a pit well before the timeout
    assert float(m["goal"]) == 0.0
    assert float(m["ret"]) < 3.0


def test_eval_matches_a_manual_rollout_of_the_first_episode():
    import functools
    action = E.N_ACTIONS - 1                                # run right + jump: deterministic policy
    key = jax.random.PRNGKey(3)
    m = make_eval(lambda params, obs, k: jnp.full((obs.shape[0],), action, jnp.int32), n_envs=1)(None, key)
    _, k_env = jax.random.split(key)                        # same key derivation as the harness
    s = E.reset(jax.random.split(k_env, 1)[0], random_phases=True)
    step = jax.jit(functools.partial(E.step, random_phases=True))
    for _ in range(E.MAX_STEPS):
        s, obs, r, done, info = step(s, action)
        if bool(done):
            break
    assert float(m["length"]) == int(info["ep_length"])
    assert float(m["ret"]) == pytest.approx(float(info["ep_return"]))
    assert float(m["coins"]) == pytest.approx(float(info["ep_coins"]))
    assert float(m["max_x"]) == pytest.approx(float(info["max_x"]), abs=1e-4)
