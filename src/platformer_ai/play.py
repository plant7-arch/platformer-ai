"""Watch a trained policy in the terminal.   uv run python -m platformer_ai.play [--params RUN]
(default: the latest run, see runs.py)

Runs on the CPU (JAX_PLATFORMS=cpu unless you set it yourself), like the viewer, so it never takes GPU memory from a
run that is training.  It follows the policy: a newer params.msgpack is loaded before the next episode
(policywatch.py; --no-follow keeps the first one).  --episodes 0 plays until Ctrl-C: watch a run learn in the terminal."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")      # do not touch the GPU that is training (JAX_PLATFORMS=cuda to override)

import argparse
import functools
import itertools
import time

import jax
import jax.numpy as jnp

from platformer_ai import cli

if __name__ == "__main__":
    cli.level_from_argv()          # --level FILE: has to be set before the level is loaded by the imports below

from platformer_ai import env as E
from platformer_ai import render, runs
from platformer_ai.policywatch import PolicyWatch
from platformer_ai.ppo import ActorCritic, restore_params
from platformer_ai.terminal import Screen

END_SECONDS = 1.0                  # the last frame of an episode stays this long, with how it ended


def outcome(info) -> str:
    """How an episode ended, from the info of its last step."""
    if bool(info["goal"]):
        return "goal reached!"
    if bool(info["truncated"]):
        return "time is up"
    return "game over"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m platformer_ai.play")
    ap.add_argument("--params", default="latest",
                    help="'latest' (default), a run folder or run name under runs/, or a params.msgpack file")
    ap.add_argument("--episodes", type=int, default=3, help="how many episodes (0 = until Ctrl-C)")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--random-phases", action="store_true",
                    help="random platform / enemy phases and jittered spawn (shows if it reacts)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no-follow", action="store_true",
                    help="keep the policy it started with (default: load a newer one before each episode)")
    cli.add_level_argument(ap)
    args = ap.parse_args(argv)
    if args.episodes < 0 or args.fps <= 0:
        ap.error("--episodes must be 0 or more and --fps more than 0")

    net = ActorCritic(E.N_ACTIONS, args.hidden)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((E.OBS_DIM,), jnp.float32))
    path = runs.resolve_params(args.params)
    if not path.is_file():
        raise SystemExit(f"no policy at {path} (train one first, or pick a run: uv run python -m platformer_ai.runs)")
    print(f"policy: {path}")
    try:
        watch = PolicyWatch(args.params, lambda data: restore_params(template, data), follow=not args.no_follow)
    except ValueError as e:
        raise SystemExit(f"{path} does not fit a network of width {args.hidden} ({e}; try --hidden)") from None

    @jax.jit
    def act(params, obs, key):
        logits, _ = net.apply(params, obs)
        return jax.random.categorical(key, logits) if args.sample else jnp.argmax(logits)

    step = jax.jit(functools.partial(E.step, random_phases=args.random_phases))
    key = jax.random.PRNGKey(0)
    results, log = [], []
    episodes = itertools.count() if args.episodes == 0 else range(args.episodes)
    try:
        with Screen() as screen:          # alternate screen: your terminal history is left untouched
            for ep in episodes:
                msg = watch.poll()
                if msg:
                    log.append(f"before episode {ep + 1}: {msg}")
                key, k_reset = jax.random.split(key)
                state = E.reset(k_reset, random_phases=args.random_phases)
                obs = E.observe(state)
                total, done = 0.0, False
                while not done:
                    key, k = jax.random.split(key)
                    state, obs, reward, done, info = step(state, act(watch.params, obs, k))
                    total += float(reward)
                    shown = info["final_state"] if done else state       # the last frame: where it really ended
                    status = f"episode {ep + 1}  return {total:.1f}" + (f"   {outcome(info)}" if done else "")
                    screen.draw(render.frame(shown) + f"\n\n{status}\npolicy {watch.label}")
                    time.sleep(1.0 / args.fps)
                results.append((ep + 1, total, bool(info["goal"]), watch.label))
                time.sleep(END_SECONDS)
    except KeyboardInterrupt:
        pass
    for line in log:                      # printed on the normal screen, after the game view is gone
        print(line)
    for ep, total, goal, label in results:
        print(f"episode {ep} finished: return {total:.1f}, goal={goal}   ({label})")


if __name__ == "__main__":
    main()
