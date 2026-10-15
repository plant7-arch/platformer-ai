"""Fixed evaluation: fresh episodes from the real spawn with random platform / enemy phases.

Unlike the running `SPAWN` numbers in the training log (which only average episodes that have
already finished, so they look better than reality while many episodes are still running),
this runs every episode to its end, all on the GPU in one jitted scan (MAX_STEPS ticks).

    evaluate = make_eval(greedy(network), n_envs=512, levels=levels)
    metrics = evaluate(params, jax.random.PRNGKey(0))       # dict of scalars, plus metrics["per_level"] (arrays)
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


def make_eval(act_fn, n_envs: int = 512, levels=None, random_phases: bool = True):
    """act_fn(params, obs[N, OBS_DIM], key) -> actions[N].  Returns a jitted evaluate(params, key).
    levels: env.make_levels(...) (default: the level of level.py).  The environments are split evenly over the levels
    (n_envs is rounded up to a multiple of their number) and every one stays on its level."""
    n_lv = E.n_levels(E.DEFAULT_LEVELS if levels is None else levels)
    n_envs = -(-n_envs // n_lv) * n_lv
    level_of = jnp.arange(n_envs) % n_lv                    # env i plays level i % n_levels
    onehot = (level_of[:, None] == jnp.arange(n_lv)[None, :]).astype(jnp.float32)
    v_step = jax.vmap(functools.partial(E.step, random_starts=False, random_phases=random_phases, levels=levels,
                                        keep_level=True))
    v_reset = jax.vmap(lambda k, i: E.reset(k, random_start=False, random_phases=random_phases, levels=levels,
                                            level=i))
    v_observe = jax.vmap(functools.partial(E.observe, levels=levels))

    def evaluate(params, key):
        key, k_env = jax.random.split(key)
        states = v_reset(jax.random.split(k_env, n_envs), level_of)
        obs = v_observe(states)
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
        per = dict(ret=ret, goal=goal, coins=coins, max_x=max_x, length=length, finished=finished.astype(jnp.float32))
        out = {k: v.mean() for k, v in per.items()}                              # over all levels (equal shares)
        out["per_level"] = {k: (onehot * v[:, None]).sum(0) / onehot.sum(0) for k, v in per.items()}   # (n_levels,)
        return out

    return jax.jit(evaluate)


# ---- command line: score a policy on level files, without training ------------------------------------------------
def evaluate_levels(act_fn, params, arrays, episodes: int, key, batch: int = 8192, random_phases: bool = True,
                    log=None) -> dict:
    """At least `episodes` fresh episodes on every level of `arrays` (level.Arrays), in batches of at most `batch`
    environments (several calls, each with its own starts; n says how many ran).  -> per-level numpy arrays: goal / ret / coins / max_x / length (means)
    and n (episodes).  Packing levels together does not change what happens in them, so batches can be any mix."""
    import numpy as np
    n_lv = len(arrays)
    group = max(1, min(n_lv, batch))                              # levels per call
    sums = {k: np.zeros(n_lv) for k in ("goal", "ret", "coins", "max_x", "length")}
    count = np.zeros(n_lv)
    call = 0
    for g0 in range(0, n_lv, group):
        idx = list(range(g0, min(n_lv, g0 + group)))
        levels = E.make_levels([arrays[i] for i in idx])
        most = max(1, min(episodes, batch // len(idx)))           # episodes of each level one call can hold
        calls = -(-episodes // most)
        per_call = -(-episodes // calls)                          # equal calls (a few more episodes than asked)
        fn = make_eval(act_fn, per_call * len(idx), levels, random_phases=random_phases)
        for _ in range(calls):
            e = jax.device_get(fn(params, jax.random.fold_in(key, call)))
            call += 1
            for k in sums:
                sums[k][idx] += np.asarray(e["per_level"][k]) * per_call
            count[idx] += per_call
            if log:
                log(f"  levels {idx[0] + 1}-{idx[-1] + 1} of {n_lv}: {int(count[idx[0]])}/{episodes} episodes")
    out = {k: v / np.maximum(count, 1) for k, v in sums.items()}
    out["n"] = count
    return out


def main(argv=None) -> int:
    import argparse
    import sys
    from pathlib import Path

    import numpy as np
    from flax import serialization

    from platformer_ai import level as L
    from platformer_ai import levelcheck, levelgen, runs
    from platformer_ai.ppo import ActorCritic, Config

    ap = argparse.ArgumentParser(
        prog="python -m platformer_ai.evaluate",
        description="Score a trained policy on level files: goal rate per level and overall, with many episodes "
                    "(random platform / enemy phases and a jittered spawn, like the evaluation during training).")
    ap.add_argument("--params", default="latest",
                    help="'latest' (default), 'pretrained', a run name or folder under runs/, or a .msgpack file")
    ap.add_argument("--levels", nargs="+", metavar="PATH", required=True,
                    help="level files or folders (all *.txt inside)")
    ap.add_argument("--episodes", type=int, default=100, help="episodes per level (default 100)")
    ap.add_argument("--sample", action="store_true", help="sample actions like training instead of the most likely one")
    ap.add_argument("--fixed-phases", action="store_true",
                    help="exact spawn and the level's own platform / enemy start (every greedy episode is the same)")
    ap.add_argument("--seed", type=int, default=0, help="which random starts (default 0)")
    ap.add_argument("--batch", type=int, default=8192,
                    help="at most this many episodes run at once (lower it if the GPU runs out of memory)")
    ap.add_argument("--show", type=int, default=10, metavar="N",
                    help="list the N worst levels (default 10; 0 = none); --all lists every level")
    ap.add_argument("--all", action="store_true", help="list every level, worst first")
    ap.add_argument("--csv", metavar="FILE", help="also write every level's numbers to a CSV file")
    ap.add_argument("--hidden", type=int, default=Config.hidden, help="network width (default %(default)s)")
    a = ap.parse_args(argv)

    path = runs.resolve_params(a.params)
    if not path.is_file():
        print(f"no policy at {path} (see: uv run python -m platformer_ai.runs)", file=sys.stderr)
        return 2
    try:
        arrays = L.load_all(a.levels)
    except levelcheck.LevelError as e:
        print(f"--levels: {e}", file=sys.stderr)
        return 2
    net = ActorCritic(E.N_ACTIONS, a.hidden)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((E.OBS_DIM,), jnp.float32))
    try:
        params = serialization.from_bytes(template, path.read_bytes())
    except (ValueError, KeyError, TypeError):
        print(f"{path} does not fit a network of width {a.hidden} (try --hidden)", file=sys.stderr)
        return 2
    mode = "sampled" if a.sample else "greedy"
    phases = "fixed phases" if a.fixed_phases else "random phases"
    print(f"policy {path}   devices {jax.devices()}")
    print(f"{len(arrays)} level{'s' if len(arrays) > 1 else ''} x {a.episodes} episodes, {mode}, {phases}, seed {a.seed}")

    r = evaluate_levels((sampled if a.sample else greedy)(net), params, arrays, a.episodes,
                        jax.random.PRNGKey(a.seed), batch=a.batch, random_phases=not a.fixed_phases)
    names = [x.name for x in arrays]
    g, n = r["goal"], r["n"]

    def line(label, sel):
        k = int(sel.sum())
        ng = float((g[sel] * n[sel]).sum())
        tot = float(n[sel].sum())
        return (f"{label:<10} {k:4d} level{'s' if k != 1 else ' '}  goal {ng / tot:5.3f} ({int(round(ng))}/{int(tot)})  "
                f"return {r['ret'][sel].mean():5.2f}  coins {r['coins'][sel].mean():4.1f}  "
                f"max_x {r['max_x'][sel].mean():5.1f}  len {r['length'][sel].mean():4.0f}")

    everything = np.ones(len(names), bool)
    print(line("all", everything))
    tiers = np.array([levelgen.tier_of(x) or 0 for x in names])
    if len(set(tiers.tolist())) > 1 or tiers[0]:
        for t in sorted(set(tiers.tolist())):
            print(line(f"tier {t}" if t else "other", tiers == t))
    if len(names) > 1:
        print(f"levels by goal rate:  0: {int((g == 0).sum())}   <0.5: {int(((g > 0) & (g < 0.5)).sum())}   "
              f"0.5-0.9: {int(((g >= 0.5) & (g < 0.9)).sum())}   >=0.9: {int((g >= 0.9).sum())}")
    order = np.lexsort((r["max_x"], g))
    shown = order if a.all else order[:a.show]
    if len(shown) and (len(names) > 1 or a.all):
        print(("every level" if a.all else f"worst {len(shown)}") + ", worst first:")
        for i in shown:
            # 95% interval of the goal rate (Wilson), so a level's number shows how sure it is
            k, m = g[i] * n[i], n[i]
            z = 1.96
            c = (k + z * z / 2) / (m + z * z)
            h = z * np.sqrt(k * (m - k) / m + z * z / 4) / (m + z * z)
            print(f"  {Path(names[i]).name[-28:]:<28} goal {g[i]:5.3f} ({int(round(k))}/{int(m)}, 95%: {max(0, c - h):.2f}"
                  f"-{min(1, c + h):.2f})  coins {r['coins'][i]:4.1f}  max_x {r['max_x'][i]:5.1f}  len {r['length'][i]:4.0f}")
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("level,tier,episodes,goal,return,coins,max_x,length\n")
            for i, nm in enumerate(names):
                f.write(f"{nm},{tiers[i] or ''},{int(n[i])},{g[i]:.4f},{r['ret'][i]:.4f},{r['coins'][i]:.4f},"
                        f"{r['max_x'][i]:.3f},{r['length'][i]:.1f}\n")
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
