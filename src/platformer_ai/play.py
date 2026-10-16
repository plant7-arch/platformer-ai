"""Watch a trained policy in the terminal.   uv run python -m platformer_ai.play [--params RUN]
(default: the latest run, see runs.py)"""
from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp

from platformer_ai import cli

if __name__ == "__main__":
    cli.level_from_argv()          # --level FILE: has to be set before the level is loaded by the imports below

from platformer_ai import env as E
from platformer_ai import render, runs
from platformer_ai.ppo import ActorCritic, restore_params
from platformer_ai.terminal import Screen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="latest",
                    help="'latest' (default), a run folder or run name under runs/, or a params.msgpack file")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--random-phases", action="store_true",
                    help="random platform / enemy phases and jittered spawn (shows if it reacts)")
    ap.add_argument("--hidden", type=int, default=128)
    cli.add_level_argument(ap)
    args = ap.parse_args()

    net = ActorCritic(E.N_ACTIONS, args.hidden)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((E.OBS_DIM,), jnp.float32))
    path = runs.resolve_params(args.params)
    if not path.is_file():
        raise SystemExit(f"no policy at {path} (train one first, or pick a run: uv run python -m platformer_ai.runs)")
    print(f"policy: {path}")
    try:
        params = restore_params(template, path.read_bytes())
    except ValueError as e:
        raise SystemExit(f"{path} does not fit a network of width {args.hidden} ({e}; try --hidden)") from None

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
