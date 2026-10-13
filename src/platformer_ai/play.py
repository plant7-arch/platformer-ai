"""Watch a trained policy in the terminal.   uv run python -m platformer_ai.play --params runs/latest/params.msgpack"""
from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
from flax import serialization

from platformer_ai import env as E
from platformer_ai import render
from platformer_ai.terminal import Screen
from platformer_ai.ppo import ActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="runs/latest/params.msgpack")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--random-phases", action="store_true",
                    help="random platform / enemy phases and jittered spawn (shows if it reacts)")
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    net = ActorCritic(E.N_ACTIONS, args.hidden)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((E.OBS_DIM,), jnp.float32))
    with open(args.params, "rb") as f:
        params = serialization.from_bytes(template, f.read())

    @jax.jit
    def act(params, obs, key):
        logits, _ = net.apply(params, obs)
        return jax.random.categorical(key, logits) if args.sample else jnp.argmax(logits)

    step = jax.jit(functools.partial(E.step, random_phases=args.random_phases))
    key = jax.random.PRNGKey(0)
    results = []
    try:
        with Screen() as screen:          # alternate screen: your terminal history is left untouched
            for ep in range(args.episodes):
                key, k_reset = jax.random.split(key)
                state = E.reset(k_reset, random_phases=args.random_phases)
                obs = E.observe(state)
                total, done = 0.0, False
                while not done:
                    key, k = jax.random.split(key)
                    state, obs, reward, done, info = step(state, act(params, obs, k))
                    total += float(reward)
                    screen.draw(render.frame(state) + f"\n\nepisode {ep + 1}  return {total:.1f}")
                    time.sleep(1.0 / args.fps)
                results.append((ep + 1, total, bool(info["goal"])))
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    for ep, total, goal in results:       # printed on the normal screen, after the game view is gone
        print(f"episode {ep} finished: return {total:.1f}, goal={goal}")


if __name__ == "__main__":
    main()
