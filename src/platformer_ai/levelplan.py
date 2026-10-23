"""Can a level be finished?  A beam search over the real environment (env.step, the exact training physics).

    from platformer_ai import levelplan
    ok = levelplan.solvable([text1, text2, ...])          # [bool]: reaches the goal from every tried start
    uv run python -m platformer_ai.levelplan gen/test/    # check level files, print the ones it cannot finish

levelcheck only finds levels that are broken; this finds levels that are too hard for the physics (a gap too wide, a
ledge too high, an enemy that cannot be passed, a ferry that is too slow for the 900-step limit).  For every level it
tries several starts from the real spawn: fixed phases, and random platform / enemy phases with a jittered spawn (the
way training and evaluation start).  The level counts as beatable only if every start reaches the goal.

The search: a "macro" action holds one of the 6 actions for MACRO env steps; every macro step expands each kept state
by all 6 actions, drops the ones that died, merges states that are in the same small (x, y, vy) cell (all states of a
search share the time, so the platforms and enemies are in the same place for all of them) and keeps the BEAM best:
most of them the furthest right, the rest picked at random (so a state that waits for a ferry or an enemy survives).
A level it cannot finish is very likely too hard, but the search is not complete: it may in rare cases miss a route.
All levels of a call are searched at once (padded with env.make_levels), in chunks.
"""
from __future__ import annotations

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

from platformer_ai import env as E
from platformer_ai import level as L
from platformer_ai import levelcheck

MACRO = 3                 # env steps per search step
BEAM = 64                 # states kept per (level, start)
GREEDY = 0.75             # share of the beam that is the furthest right; the rest is random
CELL = (0.25, 0.25, 3.0)  # merge states closer than this in x, y, vy


def _macro_fn(levels):
    """jit(vmap(state, action) -> (state after MACRO steps of that action, reached the goal, died))."""
    step = functools.partial(E.step, levels=levels, keep_level=True)

    def run(state, action):
        def body(c, _):
            s, won, dead, over = c
            s2, _, _, done, info = step(s, action)
            won = won | (info["goal"] & ~over)
            dead = dead | ((done & ~info["goal"]) & ~over)          # fell, hit an enemy, or out of time
            s = jax.tree_util.tree_map(lambda a, b: jnp.where(over | done, a, b), s, s2)
            return (s, won, dead, over | done), None
        f = jnp.bool_(False)
        (s, won, dead, _), _ = jax.lax.scan(body, (state, f, f, f), None, length=MACRO)
        return s, won, dead

    return jax.jit(jax.vmap(run))


def _starts(levels, n: int, trials: int, seed: int):
    """trials starts per level: the first with fixed phases, the others with random phases (+ jittered spawn)."""
    lv_idx = jnp.repeat(jnp.arange(n), trials)
    keys = jax.random.split(jax.random.PRNGKey(seed), n * trials)
    fixed = jax.vmap(lambda k, i: E.reset(k, levels=levels, level=i))(keys, lv_idx)
    rand = jax.vmap(lambda k, i: E.reset(k, random_phases=True, levels=levels, level=i))(keys, lv_idx)
    first = (jnp.arange(n * trials) % trials) == 0
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(first.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), fixed, rand)


def _search(texts, trials: int, beam: int, seed: int, names=None):
    """(solved[n] bool, steps[n] = env steps of the slowest start, or -1).  names: for the level messages."""
    n = len(texts)
    names = names or [f"candidate {i}" for i in range(n)]
    levels = E.make_levels([L.build(t, nm) for t, nm in zip(texts, names, strict=True)])
    macro = _macro_fn(levels)
    rng = np.random.default_rng(seed)

    g = n * trials                      # independent searches
    na = E.N_ACTIONS
    start = _starts(levels, n, trials, seed)
    states = jax.tree_util.tree_map(lambda a: jnp.repeat(a, beam, axis=0), start)      # (g * beam, ...)
    valid = np.zeros((g, beam), bool)
    valid[:, 0] = True
    solved_at = np.full(g, -1)
    actions = jnp.tile(jnp.arange(na, dtype=jnp.int32), g * beam)
    group = np.repeat(np.arange(g), beam * na)
    m = beam * na
    n_keep = int(beam * GREEDY)

    for it in range(E.MAX_STEPS // MACRO + 1):
        active = (solved_at < 0) & valid.any(1)
        if not active.any():
            break
        exp = jax.tree_util.tree_map(lambda a: jnp.repeat(a, na, axis=0), states)
        new, won, dead = macro(exp, actions)
        won, dead = np.asarray(won).reshape(g, m), np.asarray(dead).reshape(g, m)
        ok = np.repeat(valid, na, axis=1) & active[:, None]
        hit = (won & ok).any(1) & (solved_at < 0)
        solved_at[hit] = (it + 1) * MACRO
        alive = (ok & ~dead & ~won & (solved_at < 0)[:, None]).ravel()

        x, y, vy = (np.asarray(v) for v in (new.x, new.y, new.vy))
        key = (np.floor(x / CELL[0]).astype(np.int64) * 4096 + np.floor(y / CELL[1]).astype(np.int64) + 64) * 64 \
            + np.floor(vy / CELL[2]).astype(np.int64) + 16
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            valid[:] = False
            continue
        # one state per (search, cell): the one furthest right
        o = np.lexsort((-x[idx], key[idx], group[idx]))
        idx = idx[o]
        first = np.ones(idx.size, bool)
        first[1:] = (group[idx][1:] != group[idx][:-1]) | (key[idx][1:] != key[idx][:-1])
        idx = idx[first]
        # rank by x inside each search: the best n_keep stay; the rest compete with random priorities
        gi = group[idx]
        o = np.lexsort((-x[idx], gi))
        idx, gi = idx[o], gi[o]
        starts = np.searchsorted(gi, gi, side="left")
        rank = np.arange(idx.size) - starts
        top = rank < n_keep
        rest = idx[~top]
        pr = rng.random(rest.size)
        o = np.lexsort((pr, group[rest]))
        rest = rest[o]
        rg = group[rest]
        rrank = np.arange(rest.size) - np.searchsorted(rg, rg, side="left") + n_keep
        keep_idx = np.concatenate([idx[top], rest[rrank < beam]])
        keep_rank = np.concatenate([rank[top], rrank[rrank < beam]])
        keep_g = group[keep_idx]

        table = np.repeat(np.arange(g) * m, beam).reshape(g, beam)       # filler: any state of the search
        valid = np.zeros((g, beam), bool)
        table[keep_g, keep_rank] = keep_idx
        valid[keep_g, keep_rank] = True
        take = jnp.asarray(table.ravel())
        states = jax.tree_util.tree_map(lambda a, take=take: a[take], new)

    per = solved_at.reshape(n, trials)
    solved = (per >= 0).all(1)
    return solved, np.where(solved, per.max(1), -1)


def solvable(texts, trials: int = 3, beam: int = BEAM, seed: int = 0, chunk: int = 48, with_steps: bool = False,
             names=None):
    """[bool] per level text: the search reached the goal from every one of `trials` starts (with_steps: also the
    env steps the slowest start needed, -1 if not solved).  names: file names for the level warnings (default
    'candidate N')."""
    out, steps = [], []
    for i in range(0, len(texts), chunk):
        part = None if names is None else list(names[i:i + chunk])
        s, st = _search(list(texts[i:i + chunk]), trials, beam, seed, part)
        out += s.tolist()
        steps += st.tolist()
    return (out, steps) if with_steps else out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m platformer_ai.levelplan LEVEL_FILES_OR_FOLDERS...")
        return 2
    try:
        files = levelcheck.level_files(argv)
    except levelcheck.LevelError as e:              # a folder without level files
        print(f"levelplan: {e}", file=sys.stderr)
        return 2
    texts, problems = [], []                        # every broken or unreadable file at once, without a traceback
    for f in files:
        try:
            with open(f) as fh:
                texts.append(fh.read())
        except OSError as e:
            problems.append(f"cannot read the level file {f}: {e.strerror}")
            continue
        limits = (L.DEFAULT_PLATFORM_TRAVEL, L.DEFAULT_ENEMY_REACH, L.ENEMY_HALF_W, L.ENEMY_H)
        if levelcheck.check(texts[-1], *limits).errors:          # (check() logs nothing: warnings come once, below)
            try:
                L.build(texts[-1], f)
            except levelcheck.LevelError as e:
                problems.append(str(e))
    if problems:
        print("levelplan: " + "\n".join(problems), file=sys.stderr)
        return 2
    ok, steps = solvable(texts, with_steps=True, names=files)
    for f, good, st in zip(files, ok, steps):
        print(f"{'ok     ' if good else 'NOT OK '} {f}" + (f"  ({st} steps)" if good else ""))
    print(f"{sum(ok)} of {len(ok)} levels finished by the search")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
