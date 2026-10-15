"""Train PPO on the platformer.   uv run python -m platformer_ai.train --total-steps 25000000

    --levels my_levels/ extra.txt      train on all these levels at once (a folder = all its *.txt files)
    --test-levels unseen/              also evaluate on levels it never trains on (does it generalise?)
Without --levels it trains on the level of level.py (or PLATFORMER_LEVEL).
Every run gets its own folder runs/<date>_<time>[_NAME] (--name NAME, or --out DIR); see runs.py.
List them:  uv run python -m platformer_ai.runs
Continue a run that stopped (Ctrl-C, crash, or to train longer):
    uv run python -m platformer_ai.train --resume [RUN] [--total-steps N]
Start a NEW run from the weights of another policy (any run, also runs/pretrained/; any levels and settings):
    uv run python -m platformer_ai.train --init-from pretrained --levels my_levels/"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from platformer_ai import env as E
from platformer_ai import level as L
from platformer_ai import levelgen, runs
from platformer_ai.evaluate import greedy, make_eval, sampled
from platformer_ai.levelcheck import LevelError
from platformer_ai.ppo import Config, make_train


def load_levels(paths, what: str):
    """Level files / folders -> (names, env.Level); exits with a short message if any level is broken."""
    try:
        arrays = L.load_all(paths)
    except LevelError as e:
        raise SystemExit(f"{what}: {e}") from None
    return [a.name for a in arrays], E.make_levels(arrays)


PER_LEVEL_LINES = 12      # up to this many levels the evaluation prints one line per level, above it a summary
WORST = 5                 # ... with the worst levels named


def eval_lines(name: str, e: dict, names: list[str], n_envs: int) -> list[str]:
    """The evaluation printout: one line over all levels, plus one per level when there are a few, or a summary (per
    tier for generated levels, the spread of the per-level goal rates and the worst levels) when there are many."""
    out = [f"    EVAL {name}  {n_envs} fresh episodes, random phases:  return {e['ret']:5.2f}  "
           f"coins {e['coins']:4.1f}  goal {e['goal']:4.2f}  max_x {e['max_x']:5.1f}  len {e['length']:4.0f}"]
    if len(names) <= 1:
        return out
    p = {k: np.asarray(v) for k, v in e["per_level"].items()}
    if len(names) <= PER_LEVEL_LINES:
        for i, n in enumerate(names):
            out.append(f"        {Path(n).name[-28:]:<28} return {p['ret'][i]:5.2f}  coins {p['coins'][i]:4.1f}  "
                       f"goal {p['goal'][i]:4.2f}  max_x {p['max_x'][i]:5.1f}")
        return out
    tiers = np.array([levelgen.tier_of(n) or 0 for n in names])
    for t in sorted(set(tiers.tolist())):
        sel = tiers == t
        label = f"tier {t}" if t else "other"
        out.append(f"        {label:<8} {int(sel.sum()):4d} levels   return {p['ret'][sel].mean():5.2f}  "
                   f"coins {p['coins'][sel].mean():4.1f}  goal {p['goal'][sel].mean():4.2f}  "
                   f"max_x {p['max_x'][sel].mean():5.1f}")
    g = p["goal"]
    out.append(f"        levels by goal rate:  0: {int((g == 0).sum())}   <0.5: {int(((g > 0) & (g < 0.5)).sum())}   "
               f"0.5-0.9: {int(((g >= 0.5) & (g < 0.9)).sum())}   >=0.9: {int((g >= 0.9).sum())}")
    worst = np.lexsort((p["max_x"], g))[:WORST]
    out.append("        worst: " + ",  ".join(f"{Path(names[i]).name} {g[i]:.2f} (x {p['max_x'][i]:.0f})" for i in worst))
    return out


def level_summary(names: list[str], args) -> str:
    """'3 levels: a.txt, b.txt, c.txt', or for many: '300 levels from gen/train/ (tier 1: 100, tier 2: 100, ...)'."""
    n = len(names)
    if n <= PER_LEVEL_LINES:
        return f"{n} level{'s' if n > 1 else ''}: {', '.join(names)}"
    tiers = [levelgen.tier_of(x) for x in names]
    counts = ", ".join(f"tier {t}: {tiers.count(t)}" for t in sorted({t for t in tiers if t}))
    other = tiers.count(None)
    detail = counts + (f", other: {other}" if other and counts else "")
    return f"{n} levels from {' '.join(args or [])}" + (f" ({detail})" if detail else "")


def eval_envs(n_levels: int, per_level: int) -> int:
    """Episodes per evaluation: at least 512, at least per_level for every level, a multiple of the level count."""
    n = max(512, per_level * n_levels)
    return -(-n // n_levels) * n_levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=None,
                    help=f"steps in total (default {Config.total_timesteps:,}); with --resume: a bigger number "
                         "trains on past the original end")
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
    ap.add_argument("--eval-per-level", type=int, default=4,
                    help="evaluation episodes per level (at least 512 in total; default 4)")
    ap.add_argument("--levels", nargs="+", metavar="PATH",
                    help="level files or folders (all *.txt inside) to train on together "
                         "(default: the level of level.py / PLATFORMER_LEVEL)")
    ap.add_argument("--test-levels", nargs="+", metavar="PATH",
                    help="levels that are only evaluated, never trained on (generalisation)")
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--name", help="label added to the run folder name: runs/<date>_<time>_NAME")
    ap.add_argument("--out", help="run folder (default: a new runs/<date>_<time>[_NAME])")
    ap.add_argument("--resume", metavar="RUN", nargs="?", const="latest",
                    help="continue a run where it stopped: 'latest' (also: just --resume), a run name or folder. "
                         "Settings and levels come from its run.json; only --total-steps / --eval-every apply")
    ap.add_argument("--init-from", metavar="RUN",
                    help="start the network from this policy ('pretrained', a run name or folder, or a .msgpack "
                         "file) instead of random weights; a new run with fresh optimiser and learning-rate schedule")
    args = ap.parse_args()
    if args.resume and args.init_from:
        raise SystemExit("--resume continues a run, --init-from starts a new one: use one of them")
    start_params = None
    if args.init_from:
        start_params = runs.resolve_params(args.init_from)
        if not start_params.is_file():
            raise SystemExit(f"--init-from: no policy at {start_params} (see: uv run python -m platformer_ai.runs)")

    resumed = None
    if args.resume:
        out, info, cfg, resumed = resume_point(args)
        level_args, test_args = info["level_args"], info["test_level_args"]
        if info.get("platformer_level") != os.environ.get("PLATFORMER_LEVEL") and not level_args:
            raise SystemExit(f"this run trained on PLATFORMER_LEVEL={info.get('platformer_level')}; "
                             f"set it the same way to resume")
    else:
        cfg = replace(Config(), total_timesteps=args.total_steps or Config.total_timesteps, num_envs=args.num_envs,
                      num_steps=args.num_steps, lr=args.lr, seed=args.seed, gamma=args.gamma,
                      progress_coef=args.progress_coef, random_starts=not args.no_random_starts,
                      random_phases=not args.no_random_phases, spawn_prob=args.spawn_prob)
        level_args, test_args = args.levels, args.test_levels
    if level_args:
        names, levels = load_levels(level_args, "--levels")
    else:
        names, levels = [L.LEVEL_SOURCE], E.DEFAULT_LEVELS
    tests = load_levels(test_args, "--test-levels") if test_args else None

    if not resumed:
        out = Path(args.out) if args.out else runs.new_run_dir(args.name)
        out.mkdir(parents=True, exist_ok=True)
        info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), command=" ".join(sys.argv),
                    levels=names, test_levels=tests[0] if tests else [], level_args=level_args,
                    test_level_args=test_args, platformer_level=os.environ.get("PLATFORMER_LEVEL"),
                    config=dataclasses.asdict(cfg), updates=0, steps=0, last_eval={}, finished=False,
                    init_from=str(start_params) if start_params else None)
        (out / "levels.txt").write_text("".join(f"train {n}\n" for n in names) +
                                        "".join(f"test {n}\n" for n in (tests[0] if tests else [])))
    else:
        info.setdefault("resumed", []).append(dict(at=datetime.datetime.now().isoformat(timespec="seconds"),
                                                   from_update=resumed["update"], command=" ".join(sys.argv)))
        info.update(config=dataclasses.asdict(cfg), finished=False)
    runs.write_info(out, info)
    print(f"run folder: {out}" + (f"   (resuming after update {resumed['update']})" if resumed else ""))
    print(f"devices: {jax.devices()}")
    print(f"training on {level_summary(names, level_args)}"
          + (f";  testing on {level_summary(tests[0], test_args)} unseen" if tests else ""))
    print(f"{cfg.n_updates} updates x {cfg.batch_size} steps = {cfg.n_updates * cfg.batch_size:,} steps")

    init, update, network = make_train(cfg, levels)
    runner = init(jax.random.PRNGKey(cfg.seed))
    if start_params:
        template = runner[0].params
        try:
            params = serialization.from_bytes(template, start_params.read_bytes())
            leaves, want = jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(template)
            fits = (jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(template)
                    and all(a.shape == b.shape for a, b in zip(leaves, want)))
        except (ValueError, KeyError, TypeError):
            fits = False
        if not fits:
            if not args.out:
                shutil.rmtree(out, ignore_errors=True)       # the new, still empty run folder
            raise SystemExit(f"--init-from: {start_params} does not fit this network (another network size or "
                             "observation size)")
        runner = (runner[0].replace(params=params),) + tuple(runner[1:])
        print(f"network starts from {start_params}")
    first = 1
    if resumed:
        template = dict(train_state=runner[0], rng=runner[3])
        saved = serialization.from_bytes(template, resumed["bytes"])
        # the network, optimiser (Adam moments, step count -> learning-rate schedule) and random stream go on where
        # they stopped; the environments simply start fresh episodes
        runner = (saved["train_state"], runner[1], runner[2], saved["rng"])
        first = resumed["update"] + 1
    n_eval = eval_envs(len(names), args.eval_per_level)
    evals = [("greedy ", names, n_eval, make_eval(greedy(network), n_eval, levels)),
             ("sampled", names, n_eval, make_eval(sampled(network), n_eval, levels))]
    if tests:
        n_test = eval_envs(len(tests[0]), args.eval_per_level)
        evals.append(("UNSEEN greedy", tests[0], n_test, make_eval(greedy(network), n_test, tests[1])))
    eval_key = jax.random.PRNGKey(12345)       # same start conditions at every evaluation

    def save(i: int, finished: bool = False):
        (out / runs.PARAMS).write_bytes(serialization.to_bytes(runner[0].params))
        state = serialization.to_bytes(dict(train_state=runner[0], rng=runner[3]))
        tmp = out / (runs.STATE + ".tmp")
        tmp.write_bytes(state)
        tmp.replace(out / runs.STATE)            # (an interrupted save never leaves a broken checkpoint)
        info.update(updates=i, steps=i * cfg.batch_size, finished=finished)
        runs.write_info(out, info)

    keys = ("n_all", "all_return", "all_goal", "n_spawn", "sp_return", "sp_coins", "sp_goal",
            "sp_maxx", "sp_len")
    window = deque(maxlen=10)          # average over the last 10 updates so rare episode ends show up
    t_start = time.time()
    done = first - 1                                   # last update that finished
    try:
        for i in range(first, cfg.n_updates + 1):
            t0 = time.time()
            runner, m = update(runner)
            done = i
            m = jax.device_get(m)                      # the only host sync per update
            dt = time.time() - t0
            window.append(m)
            a = {k: float(sum(w[k] for w in window)) for k in keys}
            n_sp, n_all = max(a["n_spawn"], 1.0), max(a["n_all"], 1.0)
            tag = " (includes compile)" if i == first else ""
            print(f"upd {i:4d}/{cfg.n_updates}  steps {i * cfg.batch_size:>11,}  "
                  f"SPAWN eps {int(a['n_spawn']):5d} return {a['sp_return'] / n_sp:5.2f} "
                  f"coins {a['sp_coins'] / n_sp:4.1f} goal {a['sp_goal'] / n_sp:4.2f} "
                  f"max_x {a['sp_maxx'] / n_sp:5.1f} len {a['sp_len'] / n_sp:5.0f} | "
                  f"ALL return {a['all_return'] / n_all:5.2f} goal {a['all_goal'] / n_all:4.2f} | "
                  f"ent {m['entropy']:.2f} v_loss {m['v_loss']:.3f} {cfg.batch_size / dt:>9,.0f} sps{tag}")
            if args.eval_every and (i % args.eval_every == 0 or i == cfg.n_updates):
                for name, lv_names, n_envs, fn in evals:
                    e = jax.device_get(fn(runner[0].params, eval_key))
                    print("\n".join(eval_lines(name, e, lv_names, n_envs)))
                    info["last_eval"][name.strip()] = {k: round(float(e[k]), 4)
                                                       for k in ("ret", "coins", "goal", "max_x", "length")}
            if i % 25 == 0:
                save(i)
    except KeyboardInterrupt:                          # Ctrl-C: keep everything up to the last finished update
        if done >= first:
            save(done)
        print(f"\nstopped after update {info['updates']}; saved in {out}.  Continue with:\n"
              f"    uv run python -m platformer_ai.train --resume {out.name}")
        return
    save(cfg.n_updates, finished=True)
    print(f"done in {time.time() - t_start:.0f}s, saved {out / runs.PARAMS}")


def resume_point(args):
    """--resume: -> (run folder, its run.json, Config, dict(update=last saved update, bytes=checkpoint))."""
    if args.levels or args.test_levels or args.out or args.name:
        raise SystemExit("--resume continues a run with its own levels and folder: leave out --levels / "
                         "--test-levels / --out / --name")
    run = runs.resolve_run(args.resume)
    if run is None:
        raise SystemExit(f"no run '{args.resume}' to resume (see: uv run python -m platformer_ai.runs)")
    info = runs.read_info(run)
    if not info.get("config") or "level_args" not in info:
        raise SystemExit(f"{run} has no run.json / checkpoint (trained before --resume existed, or only its policy "
                         f"was kept), so it cannot be resumed.  Train on from its policy in a new run:\n"
                         f"    uv run python -m platformer_ai.train --init-from {run.name} [--levels ...]")
    state = run / runs.STATE
    if not state.is_file():
        raise SystemExit(f"{run} has no {runs.STATE} (saved every 25 updates): nothing to resume from.  To train on "
                         f"from its policy in a new run:  uv run python -m platformer_ai.train --init-from {run.name}")
    cfg = Config(**info["config"])
    if args.total_steps:
        cfg = replace(cfg, total_timesteps=args.total_steps)
        if args.total_steps != info["config"]["total_timesteps"]:
            print(f"note: total steps {info['config']['total_timesteps']:,} -> {args.total_steps:,}: the learning-rate "
                  "schedule is stretched to the new end (it goes on from lr x (1 - done / new total) down to 0)")
    if int(info["updates"]) >= cfg.n_updates:           # checked before anything is written into the run
        raise SystemExit(f"the run already did all {cfg.n_updates} updates; give a bigger --total-steps "
                         f"(now {cfg.total_timesteps:,}) to train on")
    return run, info, cfg, dict(update=int(info["updates"]), bytes=state.read_bytes())


if __name__ == "__main__":
    main()
