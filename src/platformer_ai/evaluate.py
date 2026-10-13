"""Fixed evaluation: fresh episodes from the real spawn with random platform / enemy phases.

Unlike the running `SPAWN` numbers in the training log (which only average episodes that have
already finished, so they look better than reality while many episodes are still running),
this runs every episode to its end, all on the GPU in one jitted scan (MAX_STEPS ticks).

    evaluate = make_eval(greedy(network), n_envs=512)
    metrics = evaluate(params, jax.random.PRNGKey(0))       # dict of scalars
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from platformer_ai import env as E


def greedy(network):
    """Always take the most likely action."""
    def act(params, obs, key):
        logits, _ = network.apply(params, obs)
        return jnp.argmax(logits, axis=-1)
    return act


def sampled(network):
    """Sample actions like the training policy does."""
    def act(params, obs, key):
        logits, _ = network.apply(params, obs)
        return jax.random.categorical(key, logits)
    return act


def make_eval(act_fn, n_envs: int = 512):
    """act_fn(params, obs[N, OBS_DIM], key) -> actions[N].  Returns a jitted evaluate(params, key)."""
    v_step = jax.vmap(functools.partial(E.step, random_starts=False, random_phases=True))
    v_reset = jax.vmap(functools.partial(E.reset, random_start=False, random_phases=True))

    def evaluate(params, key):
        key, k_env = jax.random.split(key)
        states = v_reset(jax.random.split(k_env, n_envs))
        obs = jax.vmap(E.observe)(states)
        zeros = jnp.zeros(n_envs, jnp.float32)
        init = (states, obs, key, jnp.zeros(n_envs, bool), zeros, zeros, zeros, zeros, zeros)

        def body(carry, _):
            states, obs, key, finished, ret, coins, goal, max_x, length = carry
            key, k = jax.random.split(key)
            action = act_fn(params, obs, k)
            states, obs, _, done, info = v_step(states, action)
            first = done & ~finished                      # only the first episode of each env counts

            def rec(old, new):
                return jnp.where(first, new.astype(jnp.float32), old)

            return (states, obs, key, finished | done, rec(ret, info["ep_return"]),
                    rec(coins, info["ep_coins"]), rec(goal, info["goal"]),
                    rec(max_x, info["max_x"]), rec(length, info["ep_length"])), None

        carry, _ = jax.lax.scan(body, init, None, length=E.MAX_STEPS)
        _, _, _, finished, ret, coins, goal, max_x, length = carry
        return dict(ret=ret.mean(), goal=goal.mean(), coins=coins.mean(), max_x=max_x.mean(),
                    length=length.mean(), finished=finished.mean())

    return jax.jit(evaluate)
