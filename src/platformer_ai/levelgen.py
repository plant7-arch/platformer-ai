"""Random levels: valid (levelcheck) and beatable (checked with the real env, see levelplan.py), in 3 difficulty tiers.

    uv run python -m platformer_ai.levelgen                       # gen/train/ (300 levels) and gen/test/ (60), seed 0
    uv run python -m platformer_ai.levelgen --train 600 --test 90 --seed 1 --out gen2
    uv run python -m platformer_ai.levelgen --show 3 --tier 3     # print a few tier-3 levels, write nothing
    uv run python -m platformer_ai.train --levels gen/train/ --test-levels gen/test/

A level is built left to right from segments (flat ground, steps, gaps, ledges, ferries over pits, patrolling enemies,
stepping stones, upper platforms, riders), spawn on the left, goal flag on the right, 40..80 tiles wide, 10..12 rows.
Every segment stays inside what the physics allow (env.py: the feet rise at most ~2.2 tiles, a jump covers ~4 tiles):

    step up      <= 2 tiles (no gap)            gap at the same height  <= 3 tiles
    gap + up 1   <= 2 tiles                     ledges / upper platforms 2 tiles above the ground

and every segment ends on at least 3 tiles of ground, so the next one starts from a safe landing.  A start point 'S'
goes in front of each hard part (random-start training practises it directly).

Tiers (in the file name, t1_0007.txt, and in a comment line, so a curriculum can pick them up later):
    1  40..55 wide, gaps 1-2, steps of 1, at most one enemy and one ferry
    2  50..70 wide, gaps 1-3, steps of 2, enemies, ferries, stepping stones, upper platforms with coins
    3  60..80 wide, gaps 2-3, more enemies (two on one stretch, on ledges), riders on ferries and upper platforms

The generator retries until levelcheck reports no errors and no warnings; levelplan then drops the levels its search
cannot finish (from the spawn with fixed phases and with random phases) and they are replaced by new ones.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field

from platformer_ai import levelcheck

TIERS = (1, 2, 3)
WIDTH = {1: (40, 55), 2: (50, 70), 3: (60, 80)}
HEIGHT = (10, 12)
G_MIN = 5                 # the highest ground row: room for a ledge (g-2), an enemy on it (g-3) and a coin (g-4)
SPAWN_SAFE = 7            # no enemy segment starts closer than this to the spawn column

# segment weights per tier
WEIGHTS = {
    1: dict(flat=25, step=20, gap=30, enemy=15, ledge=10, ferry=10),
    2: dict(flat=10, step=15, gap=22, enemy=25, ledge=10, ferry=10, stones=8, upper=7),
    3: dict(flat=5, step=10, gap=20, enemy=20, ledge=10, ferry=15, stones=10, upper=10),
}
LIMITS = {1: dict(enemy=1, ferry=1), 2: dict(ferry=2), 3: dict(ferry=3)}


class Retry(Exception):
    """This segment does not fit here: the generator tries another one."""


@dataclass
class Builder:
    h: int
    tier: int
    rng: random.Random
    ground: list = field(default_factory=list)      # surface row per column (None = pit)
    cells: dict = field(default_factory=dict)       # (row, col) -> symbol on top of the terrain
    plats: list = field(default_factory=list)       # (row, c0, c1, stop_x)
    enemies: list = field(default_factory=list)     # (row, col, lo, hi)
    cur: int = 0                                    # current ground row
    used: dict = field(default_factory=dict)        # segment kind -> count

    # ---- primitives ----
    @property
    def w(self) -> int:
        return len(self.ground)

    def add_ground(self, n: int, row: int | None = None) -> int:
        """n columns of ground at `row` (default: the current one); returns the first column."""
        row = self.cur if row is None else row
        c0 = self.w
        self.ground += [row] * n
        self.cur = row
        return c0

    def add_pit(self, n: int) -> int:
        c0 = self.w
        self.ground += [None] * n
        return c0

    def put(self, r: int, c: int, ch: str) -> None:
        assert 0 <= r < self.h and (r, c) not in self.cells, (r, c, ch)
        self.cells[(r, c)] = ch

    def free(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and (r, c) not in self.cells and (self.ground[c] is None or r < self.ground[c])

    def coin(self, r: int, c: int) -> None:
        if self.free(r, c):
            self.put(r, c, "o")

    def mark_start(self) -> None:
        """A start point 'S' on the last ground column (in front of the hard part that follows)."""
        c = self.w - 1
        if self.ground[c] is not None and self.free(self.ground[c] - 1, c):
            self.put(self.ground[c] - 1, c, "S")

    def tier_coin(self, p: float) -> bool:
        return self.rng.random() < p

    def heights(self, lo_dh: int, hi_dh: int) -> list[int]:
        """New ground rows cur+dh (dh in tiles UP, so row = cur - dh) that stay inside [G_MIN, h-2]."""
        return [self.cur - dh for dh in range(lo_dh, hi_dh + 1) if G_MIN <= self.cur - dh <= self.h - 2]

    # ---- segments (each ends on >= 3 tiles of ground) ----
    def seg_flat(self):
        n = self.rng.randint(3, 7)
        c0 = self.add_ground(n)
        if self.tier_coin(0.75):
            k = self.rng.randint(1, min(3, n))
            s = self.rng.randint(c0, c0 + n - k)
            row = self.cur - self.rng.choice((1, 1, 2))
            for c in range(s, s + k):
                self.coin(row, c)

    def seg_step(self):
        up = 1 if self.tier == 1 else 2
        opts = [r for r in self.heights(-3, up) if r != self.cur]
        if not opts:
            raise Retry
        self.add_ground(self.rng.randint(3, 4), self.rng.choice(opts))
        if self.tier_coin(0.5):
            self.coin(self.cur - 1, self.w - 1)

    def seg_gap(self):
        n = self.rng.randint(*{1: (1, 2), 2: (1, 3), 3: (2, 3)}[self.tier])
        opts = self.heights(-2, 1 if n <= 2 else 0)
        if not opts:
            raise Retry
        new = self.rng.choice(opts)
        if n >= 2:
            self.mark_start()
        top = min(self.cur, new)
        c0 = self.add_pit(n)
        if self.tier_coin(0.6):
            self.coin(top - 2, c0 + n // 2)
        self.add_ground(self.rng.randint(3, 5), new)

    def seg_enemy(self):
        if self.w < SPAWN_SAFE + 2:
            raise Retry
        two = self.tier == 3 and self.tier_coin(0.4)
        n = self.rng.randint(13, 15) if two else self.rng.randint(7, 10)
        self.mark_start()
        c0 = self.add_ground(n)
        c1 = c0 + n - 1
        g = self.cur
        # the patrol leaves 2.5 tiles at the start of the stretch and 1.5 at the end (room to land after the hop)
        spans = [(c0 + 2.5, c1 + 1 - 1.5)]
        if two:
            mid = (c0 + c1 + 1) / 2
            spans = [(c0 + 2.5, mid - 0.6), (mid + 0.6, c1 + 1 - 1.5)]
        for lo, hi in spans:
            col = self.rng.randint(math.ceil(lo - 0.5), math.floor(hi - 0.5))     # its centre col + 0.5 inside the patrol
            self.put(g - 1, col, "E")
            self.enemies.append((g - 1, col, lo, hi))
            if self.tier_coin(0.6):
                self.coin(g - 2 - self.rng.randint(0, 1), int((lo + hi) / 2))

    def seg_ledge(self):
        n = self.rng.randint(7, 9)
        c0 = self.add_ground(n)
        g = self.cur
        k = self.rng.randint(3, 4)
        l0 = c0 + 2
        for c in range(l0, l0 + k):
            self.put(g - 2, c, "-")
        rider = self.tier == 3 and self.tier_coin(0.5) and c0 >= SPAWN_SAFE
        if rider:
            col = self.rng.randint(l0, l0 + k - 1)
            self.put(g - 3, col, "E")
            self.enemies.append((g - 3, col, l0 + 0.375, l0 + k - 0.375))
            self.coin(g - 4, l0 + k // 2)
        else:
            for c in self.rng.sample(range(l0, l0 + k), self.rng.randint(1, 2)):
                self.coin(g - 3, c)

    def seg_ferry(self):
        rider = self.tier == 3 and self.tier_coin(0.35)
        pw = 4 if rider else self.rng.randint(2, 3)
        travel = self.rng.randint(4, 6) if rider else self.rng.randint(4, 7)
        opts = self.heights(-1, 0)
        if not opts:
            raise Retry
        new = self.rng.choice(opts)
        self.mark_start()
        g = self.cur
        p0 = self.add_pit(pw + travel)
        for c in range(p0, p0 + pw):
            self.put(g, c, "M")
        start = p0 + pw / 2
        self.plats.append((g, p0, p0 + pw - 1, start + travel))
        if rider:                                    # an enemy walks the right part; the player rides the left part
            self.put(g - 1, p0 + 3, "E")
            self.enemies.append((g - 1, p0 + 3, start + 0.7, start + 1.62))
            if self.tier_coin(0.5):
                self.coin(g - 1, p0)
        elif self.tier_coin(0.5):
            self.coin(g - 1, p0 + self.rng.randint(0, pw - 1))
        self.add_ground(self.rng.randint(3, 5), new)

    def seg_stones(self):
        stones = self.rng.randint(2, 3)
        sw = 1 if (self.tier == 3 and self.tier_coin(0.3)) else 2
        g = self.cur
        self.mark_start()
        for _ in range(stones):
            gap = self.rng.randint(2, 3 if self.tier == 3 else 2)
            c0 = self.add_pit(gap)
            if self.tier_coin(0.4):
                self.coin(g - 2, c0 + gap // 2)
            s0 = self.add_pit(sw)
            for c in range(s0, s0 + sw):
                self.put(g, c, "-")
        self.add_pit(self.rng.randint(2, 3 if self.tier == 3 else 2))
        self.add_ground(self.rng.randint(3, 5), g)

    def seg_upper(self):
        rider = self.tier == 3 and self.tier_coin(0.5) and self.cur - 3 >= 0 and self.w >= SPAWN_SAFE
        pw = 4 if rider else self.rng.randint(2, 3)
        travel = self.rng.randint(3, 5)
        n = pw + travel + 2
        c0 = self.add_ground(n)
        g = self.cur
        p0 = c0 + 1
        for c in range(p0, p0 + pw):
            self.put(g - 2, c, "M")
        start = p0 + pw / 2
        self.plats.append((g - 2, p0, p0 + pw - 1, start + travel))
        if rider:
            self.put(g - 3, p0 + 3, "E")
            self.enemies.append((g - 3, p0 + 3, start + 0.7, start + 1.62))
            self.coin(g - 3, p0)
        else:
            self.coin(g - 3, p0 + self.rng.randint(0, pw - 1))

    # ---- output ----
    def text(self, comment: str = "") -> str:
        rows = [["."] * self.w for _ in range(self.h)]
        for c, g in enumerate(self.ground):
            if g is not None:
                for r in range(g, self.h):
                    rows[r][c] = "#"
        for (r, c), ch in self.cells.items():
            rows[r][c] = ch
        lines = ["".join(r) for r in rows] + [""]
        if comment:
            lines.append(f"# {comment}")
        for r, c0, _, stop in self.plats:
            lines.append(f"platform {r + 1} {c0 + 1} to {stop:g}")
        for r, c, lo, hi in self.enemies:
            lines.append(f"enemy {r + 1} {c + 1} from {lo:g} to {hi:g}")
        return "\n".join(lines) + "\n"


def generate(tier: int, rng: random.Random, max_tries: int = 200) -> str:
    """One random level of this tier that levelcheck accepts with no errors and no warnings."""
    for _ in range(max_tries):
        text = _attempt(tier, rng)
        if text is not None:
            rep = levelcheck.check(text)
            if not rep.errors and not rep.warnings:
                return text
    raise RuntimeError(f"no valid tier-{tier} level in {max_tries} tries")


def _attempt(tier: int, rng: random.Random) -> str | None:
    h = rng.randint(*HEIGHT)
    width = rng.randint(*WIDTH[tier])
    b = Builder(h=h, tier=tier, rng=rng)
    b.cur = rng.randint(G_MIN + 1, h - 2)
    b.add_ground(rng.randint(5, 7))
    b.put(b.cur - 1, rng.randint(1, 2), "P")
    goal_len = rng.randint(4, 6)
    kinds, weights = zip(*WEIGHTS[tier].items())
    tries = 0
    while b.w < width - goal_len and tries < 100:
        tries += 1
        kind = rng.choices(kinds, weights)[0]
        if b.used.get(kind, 0) >= LIMITS[tier].get(kind, 99):
            continue
        snap = (list(b.ground), dict(b.cells), list(b.plats), list(b.enemies), b.cur)
        try:
            getattr(b, "seg_" + kind)()
        except Retry:
            b.ground, b.cells, b.plats, b.enemies, b.cur = snap
            continue
        if b.w > WIDTH[tier][1] - goal_len:              # too wide: undo, try a shorter segment
            b.ground, b.cells, b.plats, b.enemies, b.cur = snap
            continue
        b.used[kind] = b.used.get(kind, 0) + 1
    if b.w < WIDTH[tier][0] - goal_len:
        return None
    b.add_ground(goal_len)
    b.put(b.cur - 1, b.w - 1, "G")
    b.put(b.cur - 2, b.w - 1, "G")
    return b.text(f"generated by levelgen: tier {tier}")


def tier_of(path_or_text: str) -> int | None:
    """The tier of a generated level (from its file name t2_... or its comment line), None if unknown."""
    name = os.path.basename(path_or_text)
    if len(name) > 2 and name[0] == "t" and name[1].isdigit() and name[2] == "_":
        return int(name[1])
    for line in path_or_text.splitlines():
        if line.startswith("# generated by levelgen: tier "):
            return int(line.rsplit(" ", 1)[1])
    return None


def generate_set(n: int, seed: int, verify: bool = True, log=print) -> list[tuple[int, str]]:
    """n levels, tiers in equal shares (1, 2, 3, 1, 2, 3, ...), each valid and (verify) beatable: [(tier, text)]."""
    rng = random.Random(seed)
    want = [TIERS[i % len(TIERS)] for i in range(n)]
    done: list = [None] * n
    rounds = 0
    while any(d is None for d in done):
        rounds += 1
        todo = [i for i, d in enumerate(done) if d is None]
        cand = {i: generate(want[i], rng) for i in todo}
        if verify:
            from platformer_ai import levelplan
            ok = levelplan.solvable([cand[i] for i in todo])
        else:
            ok = [True] * len(todo)
        for i, good in zip(todo, ok):
            if good:
                done[i] = (want[i], cand[i])
        bad = len(todo) - sum(ok)
        log(f"  round {rounds}: {len(todo)} candidates, {sum(ok)} beatable" + (f", {bad} replaced" if bad else ""))
        if rounds > 30:
            raise RuntimeError("too many unbeatable levels; is the generator broken?")
    return done


def write_set(levels, folder: str) -> list[str]:
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):                            # a fresh set: old generated files go
        if f.endswith(".txt") and tier_of(f) is not None:
            os.remove(os.path.join(folder, f))
    paths = []
    for i, (tier, text) in enumerate(levels):
        p = os.path.join(folder, f"t{tier}_{i:04d}.txt")
        with open(p, "w") as fh:
            fh.write(text)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="gen", help="folder for train/ and test/ (default gen)")
    ap.add_argument("--train", type=int, default=300, help="training levels (default 300)")
    ap.add_argument("--test", type=int, default=60, help="held-out levels (default 60)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true", help="skip the beatability search (levelcheck still runs)")
    ap.add_argument("--show", type=int, default=0, metavar="N", help="print N levels and write nothing")
    ap.add_argument("--tier", type=int, choices=TIERS, help="with --show: only this tier")
    a = ap.parse_args(argv)

    if a.show:
        rng = random.Random(a.seed)
        for i in range(a.show):
            print(generate(a.tier or TIERS[i % 3], rng))
        return 0
    for name, n, seed in (("train", a.train, a.seed), ("test", a.test, a.seed + 1_000_003)):
        if n <= 0:
            continue
        print(f"{name}: {n} levels (seed {seed})")
        paths = write_set(generate_set(n, seed, verify=not a.no_verify), os.path.join(a.out, name))
        print(f"  wrote {len(paths)} files to {os.path.join(a.out, name)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
