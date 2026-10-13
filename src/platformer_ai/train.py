"""Train PPO on the platformer.   uv run python -m platformer_ai.train --total-steps 25000000"""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import jax
from flax import serialization

from platformer_ai.evaluate import greedy, make_eval, sampled
from platformer_ai.ppo import Config, make_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=Config.total_timesteps)
    ap.add_argument("--num-envs", type=int, default=Config.num_envs)
    ap.add_argument("--num-steps", type=int, default=Config.num_steps)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--gamma", type=float, default=Config.gamma)
    ap.add_argument("--progress-coef", type=float, default=Config.progress_coef,
                    help="bonus per tile of new furthest-right progress (0 = off)")
    ap.add_argument("--no-random-starts", action="store_true",
                    help="never start at the 'S' points / on the ferry (only at the spawn)")
    ap.add_argument("--no-random-phases", action="store_true",
                    help="platform and enemy always start at the same phase (the old, memorisable setup)")
    ap.add_argument("--spawn-prob", type=float, default=Config.spawn_prob,
                    help="with random starts: share of episodes that start at the spawn")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="run the fixed evaluation every N updates (0 = never)")
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--out", default="runs/latest")
    args = ap.parse_args()

    cfg = replace(Config(), total_timesteps=args.total_steps, num_envs=args.num_envs,
                  num_steps=args.num_steps, lr=args.lr, seed=args.seed, gamma=args.gamma,
                  progress_coef=args.progress_coef, random_starts=not args.no_random_starts,
                  random_phases=not args.no_random_phases, spawn_prob=args.spawn_prob)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"devices: {jax.devices()}")
    print(f"{cfg.n_updates} updates x {cfg.batch_size} steps = {cfg.n_updates * cfg.batch_size:,} steps")

    init, update, network = make_train(cfg)
    runner = init(jax.random.PRNGKey(cfg.seed))
    eval_greedy, eval_sampled = make_eval(greedy(network)), make_eval(sampled(network))
    eval_key = jax.random.PRNGKey(12345)       # same 512 start conditions at every evaluation

    def save():
        (out / "params.msgpack").write_bytes(serialization.to_bytes(runner[0].params))

    keys = ("n_all", "all_return", "all_goal", "n_spawn", "sp_return", "sp_coins", "sp_goal",
            "sp_maxx", "sp_len")
    window = deque(maxlen=10)          # average over the last 10 updates so rare episode ends show up
    t_start = time.time()
    for i in range(1, cfg.n_updates + 1):
        t0 = time.time()
        runner, m = update(runner)
        m = jax.device_get(m)                      # the only host sync per update
        dt = time.time() - t0
        window.append(m)
        a = {k: float(sum(w[k] for w in window)) for k in keys}
        n_sp, n_all = max(a["n_spawn"], 1.0), max(a["n_all"], 1.0)
        tag = " (includes compile)" if i == 1 else ""
        print(f"upd {i:4d}/{cfg.n_updates}  steps {i * cfg.batch_size:>11,}  "
              f"SPAWN eps {int(a['n_spawn']):5d} return {a['sp_return'] / n_sp:5.2f} "
              f"coins {a['sp_coins'] / n_sp:4.1f} goal {a['sp_goal'] / n_sp:4.2f} "
              f"max_x {a['sp_maxx'] / n_sp:5.1f} len {a['sp_len'] / n_sp:5.0f} | "
              f"ALL return {a['all_return'] / n_all:5.2f} goal {a['all_goal'] / n_all:4.2f} | "
              f"ent {m['entropy']:.2f} v_loss {m['v_loss']:.3f} {cfg.batch_size / dt:>9,.0f} sps{tag}")
        if args.eval_every and (i % args.eval_every == 0 or i == cfg.n_updates):
            for name, fn in (("greedy ", eval_greedy), ("sampled", eval_sampled)):
                e = jax.device_get(fn(runner[0].params, eval_key))
                print(f"    EVAL {name}  512 fresh episodes, random phases:  return {e['ret']:5.2f}  "
                      f"coins {e['coins']:4.1f}  goal {e['goal']:4.2f}  max_x {e['max_x']:5.1f}  "
                      f"len {e['length']:4.0f}")
        if i % 25 == 0:
            save()
    save()
    print(f"done in {time.time() - t_start:.0f}s, saved {out / 'params.msgpack'}")


if __name__ == "__main__":
    main()
