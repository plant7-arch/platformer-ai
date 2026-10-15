"""Random levels: valid (levelcheck) and beatable (checked with the real env, see levelplan.py), in 3 difficulty tiers,
shaped by settings (levelsettings.py) from a TOML file, command-line options, or both.

    uv run python -m platformer_ai.levelgen                               # gen/train/ (300) + gen/test/ (60), seed 0
    uv run python -m platformer_ai.levelgen --config docs/levelgen/tier3_drill.toml
    uv run python -m platformer_ai.levelgen --tiers 1=1,2=1,3=2 --weight enemy=2 --min ferry=1 --out gen_t3
    uv run python -m platformer_ai.levelgen --config my.toml --gap 3 --print-settings      # what would be used
    uv run python -m platformer_ai.levelgen --show 3 --tiers 3                           # print a few, write nothing
    uv run python -m platformer_ai.train --levels gen/train/ --test-levels gen/test/

A level is built left to right from segments (flat ground, steps, gaps, ledges, ferries over pits, patrolling enemies,
stepping stones, upper platforms, riders), spawn on the left, goal flag on the right.  Every segment stays inside what
the physics allow (env.py: the feet rise at most ~2.2 tiles, a jump covers ~4 tiles):

    step up      <= 2 tiles (no gap)            gap at the same height  <= 3 tiles (4: at the very edge)
    gap + up 1   <= 2 tiles                     ledges / upper platforms 2 tiles above the ground

and every segment ends on at least 3 tiles of ground, so the next one starts from a safe landing.  A start point 'S'
goes in front of each hard part (random-start training practises it directly).

The built-in tiers (levelsettings.TIER_DEFAULTS; the tier is in the file name, t1_0007.txt, and a comment line):
    1  40..55 wide, gaps 1-2, steps of 1, at most one enemy and one ferry
    2  50..70 wide, gaps 1-3, steps of 2, enemies, ferries, stepping stones, upper platforms with coins
    3  60..80 wide, gaps 2-3, more enemies (two on one stretch, on ledges), riders on ferries and upper platforms
Settings change them (see levelsettings.py and --help); the settings of a run are saved as OUT/settings.toml.

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
from platformer_ai import levelsettings as LS

TIERS = LS.TIER_IDS
G_MIN = 5                 # the highest ground row: room for a ledge (g-2), an enemy on it (g-3) and a coin (g-4)
SPAWN_SAFE = 7            # no enemy segment starts closer than this to the spawn column


class Retry(Exception):
    """This segment does not fit here: the generator tries another one."""


@dataclass
class Builder:
    h: int
    cfg: LS.TierConfig
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

    def chance(self, p: float) -> bool:
        """True with probability p (no random number is drawn when p is 0 or >= 1)."""
        if p <= 0:
            return False
        if p >= 1:
            return True
        return self.rng.random() < p

    def coin_chance(self, p: float) -> bool:
        return self.chance(p * self.cfg.coins)

    def rider_chance(self, p: float) -> bool:
        return self.chance(p * self.cfg.riders)

    def heights(self, lo_dh: int, hi_dh: int) -> list[int]:
        """New ground rows cur+dh (dh in tiles UP, so row = cur - dh) that stay inside [G_MIN, h-2]."""
        return [self.cur - dh for dh in range(lo_dh, hi_dh + 1) if G_MIN <= self.cur - dh <= self.h - 2]

    # ---- segments (each ends on >= 3 tiles of ground) ----
    def seg_flat(self):
        n = self.rng.randint(3, 7)
        c0 = self.add_ground(n)
        if self.coin_chance(0.75):
            k = self.rng.randint(1, min(3, n))
            s = self.rng.randint(c0, c0 + n - k)
            row = self.cur - self.rng.choice((1, 1, 2))
            for c in range(s, s + k):
                self.coin(row, c)

    def seg_step(self):
        up = self.cfg.max_step
        opts = [r for r in self.heights(-3, up) if r != self.cur]
        if not opts:
            raise Retry
        self.add_ground(self.rng.randint(3, 4), self.rng.choice(opts))
        if self.coin_chance(0.5):
            self.coin(self.cur - 1, self.w - 1)

    def seg_gap(self):
        n = self.rng.randint(*self.cfg.gap)
        opts = self.heights(-2, 1 if n <= 2 else 0)
        if not opts:
            raise Retry
        new = self.rng.choice(opts)
        if n >= 2:
            self.mark_start()
        top = min(self.cur, new)
        c0 = self.add_pit(n)
        if self.coin_chance(0.6):
            self.coin(top - 2, c0 + n // 2)
        self.add_ground(self.rng.randint(3, 5), new)

    def seg_enemy(self):
        if self.w < SPAWN_SAFE + 2:
            raise Retry
        two = self.chance(self.cfg.pairs)
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
            if self.coin_chance(0.6):
                self.coin(g - 2 - self.rng.randint(0, 1), int((lo + hi) / 2))

    def seg_ledge(self):
        n = self.rng.randint(7, 9)
        c0 = self.add_ground(n)
        g = self.cur
        k = self.rng.randint(3, 4)
        l0 = c0 + 2
        for c in range(l0, l0 + k):
            self.put(g - 2, c, "-")
        rider = self.rider_chance(0.5) and c0 >= SPAWN_SAFE
        if rider:
            col = self.rng.randint(l0, l0 + k - 1)
            self.put(g - 3, col, "E")
            self.enemies.append((g - 3, col, l0 + 0.375, l0 + k - 0.375))
            if self.coin_chance(1.0):
                self.coin(g - 4, l0 + k // 2)
        elif self.coin_chance(1.0):
            for c in self.rng.sample(range(l0, l0 + k), self.rng.randint(1, 2)):
                self.coin(g - 3, c)

    def seg_ferry(self):
        rider = self.rider_chance(0.35)
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
            if self.coin_chance(0.5):
                self.coin(g - 1, p0)
        elif self.coin_chance(0.5):
            self.coin(g - 1, p0 + self.rng.randint(0, pw - 1))
        self.add_ground(self.rng.randint(3, 5), new)

    def seg_stones(self):
        stones = self.rng.randint(2, 3)
        sw = 1 if self.chance(self.cfg.narrow_stones) else 2
        g = self.cur
        self.mark_start()
        for _ in range(stones):
            gap = self.rng.randint(*self.cfg.stone_gap)
            c0 = self.add_pit(gap)
            if self.coin_chance(0.4):
                self.coin(g - 2, c0 + gap // 2)
            s0 = self.add_pit(sw)
            for c in range(s0, s0 + sw):
                self.put(g, c, "-")
        self.add_pit(self.rng.randint(*self.cfg.stone_gap))
        self.add_ground(self.rng.randint(3, 5), g)

    def seg_upper(self):
        rider = self.rider_chance(0.5) and self.cur - 3 >= 0 and self.w >= SPAWN_SAFE
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
            if self.coin_chance(1.0):
                self.coin(g - 3, p0)
        elif self.coin_chance(1.0):
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


def generate(tier, rng: random.Random, max_tries: int = 200) -> str:
    """One random level that levelcheck accepts with no errors and no warnings.  tier: a tier number (built-in
    defaults) or a levelsettings.TierConfig."""
    cfg = tier if isinstance(tier, LS.TierConfig) else LS.resolve({}, "train", tier)
    for _ in range(max_tries):
        text = _attempt(cfg, rng)
        if text is not None:
            rep = levelcheck.check(text)
            if not rep.errors and not rep.warnings:
                return text
    need = f" with {', '.join(f'at least {n} {k}' for k, n in cfg.min.items() if n)}" if any(cfg.min.values()) else ""
    raise LS.SettingsError(f"no valid tier-{cfg.tier} level{need} in {max_tries} tries: the settings ask for more "
                           f"than fits (try wider levels, a lower min, fewer limits).  {cfg.describe()}")


def _attempt(cfg: LS.TierConfig, rng: random.Random) -> str | None:
    h = rng.randint(*cfg.height)
    width = rng.randint(*cfg.width)
    b = Builder(h=h, cfg=cfg, rng=rng)
    b.cur = rng.randint(G_MIN + 1, h - 2)
    b.add_ground(rng.randint(5, 7))
    b.put(b.cur - 1, rng.randint(1, 2), "P")
    goal_len = rng.randint(4, 6)
    kinds, weights = zip(*[(k, w) for k, w in cfg.weights.items() if w > 0])
    tries = 0
    while b.w < width - goal_len and tries < 100:
        tries += 1
        kind = rng.choices(kinds, weights)[0]
        if b.used.get(kind, 0) >= cfg.limit.get(kind, 99):
            continue
        snap = (list(b.ground), dict(b.cells), list(b.plats), list(b.enemies), b.cur)
        try:
            getattr(b, "seg_" + kind)()
        except Retry:
            b.ground, b.cells, b.plats, b.enemies, b.cur = snap
            continue
        too_many = cfg.max_enemies is not None and len(b.enemies) > cfg.max_enemies
        if b.w > cfg.width[1] - goal_len or too_many:     # too wide / too many enemies: undo, try another segment
            b.ground, b.cells, b.plats, b.enemies, b.cur = snap
            continue
        b.used[kind] = b.used.get(kind, 0) + 1
    if b.w < cfg.width[0] - goal_len:
        return None
    if any(b.used.get(k, 0) < n for k, n in cfg.min.items()):
        return None
    b.add_ground(goal_len)
    b.put(b.cur - 1, b.w - 1, "G")
    b.put(b.cur - 2, b.w - 1, "G")
    return b.text(f"generated by levelgen: tier {cfg.tier}")


def tier_of(path_or_text: str) -> int | None:
    """The tier of a generated level (from its file name t2_... or its comment line), None if unknown."""
    name = os.path.basename(path_or_text)
    if len(name) > 2 and name[0] == "t" and name[1].isdigit() and name[2] == "_":
        return int(name[1])
    for line in path_or_text.splitlines():
        if line.startswith("# generated by levelgen: tier "):
            return int(line.rsplit(" ", 1)[1])
    return None


def generate_set(n: int, seed: int, verify: bool = True, log=print, settings: dict | None = None,
                 part: str = "train") -> list[tuple[int, str]]:
    """n levels of the 'train' or 'test' set, each valid and (verify) beatable: [(tier, text)].  The tiers come in
    the shares of the settings (default: equal thirds, interleaved 1, 2, 3, 1, 2, 3, ...)."""
    settings = settings or {}
    run = LS.run_settings(settings)
    plan = LS.tier_plan(n, run["tiers"] if part == "train" else run["test_tiers"])
    cfgs = {t: LS.resolve(settings, part, t) for t in set(plan)}
    rng = random.Random(seed)
    done: list = [None] * n
    rounds = 0
    while any(d is None for d in done):
        rounds += 1
        todo = [i for i, d in enumerate(done) if d is None]
        cand = {i: generate(cfgs[plan[i]], rng) for i in todo}
        if verify:
            from platformer_ai import levelplan
            ok = levelplan.solvable([cand[i] for i in todo])
        else:
            ok = [True] * len(todo)
        for i, good in zip(todo, ok):
            if good:
                done[i] = (plan[i], cand[i])
        bad = len(todo) - sum(ok)
        log(f"  round {rounds}: {len(todo)} candidates, {sum(ok)} beatable" + (f", {bad} replaced" if bad else ""))
        if rounds > 30:
            raise RuntimeError("too many unbeatable levels; the settings may ask for more than the physics allow")
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


# command-line options -> settings names (the same names as in a settings file)
OPTIONS = [
    # flag, settings name, metavar, help
    ("--out", "out", "DIR", "folder for train/ and test/ (default gen)"),
    ("--seed", "seed", "N", "random seed of the training set (default 0)"),
    ("--train", "train", "N", "training levels (default 300)"),
    ("--test", "test", "N", "held-out levels (default 60)"),
    ("--test-seed", "test_seed", "N", f"seed of the held-out set (default seed + {LS.TEST_SEED_OFFSET})"),
    ("--tiers", "tiers", "SPEC", "which tiers in which shares: '3', '2,3', '1=1,2=1,3=2' (default 1=1,2=1,3=1)"),
    ("--width", "width", "LO-HI", "level width in tiles, e.g. 60-80 (default per tier)"),
    ("--height", "height", "LO-HI", "level height in tiles (default 10-12)"),
    ("--gap", "gap", "LO-HI", "pit width (1-4; 4 is at the edge of the jump)"),
    ("--stone-gap", "stone_gap", "LO-HI", "gaps between stepping stones (1-3)"),
    ("--max-step", "max_step", "N", "highest step up (0-2)"),
    ("--max-enemies", "max_enemies", "N", "at most N enemies per level"),
    ("--only", "only", "KINDS", "use only these segment kinds, e.g. gap,enemy,ferry"),
    ("--without", "without", "KINDS", "never use these segment kinds, e.g. stones,upper"),
    ("--coins", "coins", "X", LS.TUNING["coins"]),
    ("--riders", "riders", "X", LS.TUNING["riders"]),
    ("--pairs", "pairs", "P", LS.TUNING["pairs"]),
    ("--narrow-stones", "narrow_stones", "P", LS.TUNING["narrow_stones"]),
]
MAP_OPTIONS = [
    ("--weight", "weight", "how often a kind is picked, times its built-in weight: enemy=2,gap=0.5 (0 = never)"),
    ("--limit", "limit", "at most N segments of a kind per level: ferry=2"),
    ("--min", "min", "at least N segments of a kind per level: ferry=1,enemy=2"),
]


def build_parser() -> argparse.ArgumentParser:
    kinds = "\n".join(f"  {k:<7} {h}" for k, h in LS.KIND_HELP.items())
    ap = argparse.ArgumentParser(
        prog="python -m platformer_ai.levelgen", formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0],
        epilog=f"segment kinds:\n{kinds}\n\nEvery option is a setting of the same name in a --config file "
               "(dashes become underscores); --set NAME=VALUE reaches every setting, also per tier and for the held-"
               "out set: --set tier.3.riders=2, --set test_set.tiers=1,2,3.  The command line wins over the file.")
    ap.add_argument("--config", metavar="FILE", help="a settings file (TOML); options given here override it")
    for flag, _, metavar, help_ in OPTIONS:
        ap.add_argument(flag, metavar=metavar, help=help_)
    for flag, _, help_ in MAP_OPTIONS:
        ap.add_argument(flag, metavar="KIND=N", action="append", help=help_ + " (repeatable)")
    ap.add_argument("--verify", dest="verify", action="store_const", const=True, default=None,
                    help="check every level with the beatability search (default)")
    ap.add_argument("--no-verify", dest="verify", action="store_const", const=False,
                    help="skip the beatability search (levelcheck still runs)")
    ap.add_argument("--set", metavar="NAME=VALUE", action="append", default=[],
                    help="any setting by name, e.g. tier.3.gap=2-3 or test_set.weight.enemy=1 (repeatable)")
    ap.add_argument("--print-settings", action="store_true",
                    help="print the settings that would be used (as a settings file) and stop")
    ap.add_argument("--show", type=int, default=0, metavar="N", help="print N levels of the training set, write nothing")
    ap.add_argument("--tier", type=int, choices=TIERS, help=argparse.SUPPRESS)       # old name: --tiers T
    return ap


def settings_from_args(a) -> dict:
    """--config file, then the options on top (they win)."""
    s = LS.load_file(a.config) if a.config else {}
    raw = {name: getattr(a, name) for _, name, _, _ in OPTIONS if getattr(a, name) is not None}
    if a.tier and "tiers" not in raw:
        raw["tiers"] = str(a.tier)
    for _, name, _ in MAP_OPTIONS:
        for item in getattr(a, name) or []:                  # 'enemy=2,gap=0.5' (repeatable)
            for part in [p for p in item.replace(" ", "").split(",") if p]:
                if "=" not in part:
                    raise LS.SettingsError(f"--{name} {item!r}: expected KIND=VALUE, e.g. enemy=2")
                k, v = part.split("=", 1)
                raw.setdefault(name, {})[k] = v
    if a.verify is not None:
        raw["verify"] = a.verify
    s = LS.merge(s, LS.normalize(raw))
    for item in a.set:
        s = LS.merge(s, LS.normalize(LS.parse_set(item)))
    return s


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        s = settings_from_args(a)
        run = LS.run_settings(s)
        resolved = [LS.resolve(s, "train", t) for t in TIERS if run["tiers"].get(t)]
        resolved_test = [LS.resolve(s, "test", t) for t in TIERS if run["test_tiers"].get(t)]
        if a.print_settings:
            print(LS.dump(s, resolved), end="")
            if s.get("test_set"):
                print("# held-out set:\n" + "".join(f"#   {c.describe()}\n" for c in resolved_test), end="")
            return 0
        if a.show:
            rng = random.Random(run["seed"])
            cfgs = {c.tier: c for c in resolved}
            for t in LS.tier_plan(a.show, run["tiers"]):
                print(generate(cfgs[t], rng))
            return 0
        os.makedirs(run["out"], exist_ok=True)
        with open(os.path.join(run["out"], "settings.toml"), "w") as f:
            f.write(LS.dump(s, resolved))
        for name, n, seed in (("train", run["train"], run["seed"]), ("test", run["test"], run["test_seed"])):
            if n <= 0:
                continue
            print(f"{name}: {n} levels (seed {seed})")
            paths = write_set(generate_set(n, seed, verify=run["verify"], settings=s, part=name),
                              os.path.join(run["out"], name))
            print(f"  wrote {len(paths)} files to {os.path.join(run['out'], name)}/")
        print(f"settings saved in {os.path.join(run['out'], 'settings.toml')} (use it with --config to make this set again)")
    except LS.SettingsError as e:
        print(f"levelgen: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
