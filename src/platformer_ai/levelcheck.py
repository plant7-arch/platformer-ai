"""Checks the ASCII level before the game is built from it.

    errors    the level cannot be built (raised as LevelError, all of them at once, with row / column)
    warnings  the level loads but something looks wrong (logged on the "platformer_ai.level" logger)

Run it on the level in level.py, or on a text file (map rows, optionally followed by a blank line and path lines):

    uv run python -m platformer_ai.levelcheck [my_level.txt ...] [levels_folder/ ...]

Rows and columns in the messages count from 1, like the line numbers in an editor.
A level has exactly one 'P' and any number of enemies ('E') and moving platforms (runs of 'M'), also none.

Paths.  After the map and a blank line, optional lines say where things move (x = tile units, centre of the thing):

    platform ROW COL to STOP_X               platform with an M tile at ROW, COL: start = where the M tiles are
    enemy ROW COL from MIN_X to MAX_X        enemy at ROW, COL: patrols between MIN_X and MAX_X (walls do not matter)

Without a line a platform moves 6 tiles to the right of its start and an enemy walks 3 tiles either way of its start.
Riders: an 'E' standing directly on an 'M' tile rides that moving platform (it walks on it, bounded by its ends, and is
carried along); an 'E' on a run of '-' tiles patrols only that run.  An 'o' directly above an 'M' tile is stuck to the
platform and moves with it.  Without a path line a rider walks the whole platform; with one, MIN_X / MAX_X are x on the
platform at its start (its body must stay on the platform).  A platform has a coin only if the map
draws an 'o' right above it.
Every path is checked on its own: inside the map, nothing in the way (tiles, one-way tiles, coins, goal, spawn),
no other platform / enemy on the same stretch, ground under every enemy step, an enemy not within reach of the spawn.
What it can NOT tell you is whether a level is *beatable*: it finds goals and coins that are completely walled in,
but not ones that are merely too high or too far for the jump (apex 2.7 tiles).  Train an agent or use a bot for that.
"""
from __future__ import annotations

import logging
import re
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("platformer_ai.level")

SYMBOLS = ".#-oGPESM"
HINTS = {"p": "P", "e": "E", "m": "M", "s": "S", "g": "G", "0": "o", "O": "o", "=": "-", "_": "-", " ": "."}
MAX_DROP = 4            # a start may be this many tiles above its floor (the built-in spawn drops 2)
MAX_LISTED = 5          # cells named in one message
MIN_PATROL = 0.2        # an enemy needs at least this much room to walk
COIN_RADIUS = 0.3125    # same as level.COIN_RADIUS
NEAR_SPAWN = 3.0        # warning: an enemy path this close to the spawn (tiles)
PLAYER_HW, PLAYER_H = 0.3, 0.8      # player box, same as env.HW / env.HEIGHT

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_PLAT = re.compile(rf"^platform\s+(\d+)\s+(\d+)(?:\s+to\s+({_NUM}))?(\s+no-coin)?$", re.I)   # no-coin: still accepted (old level files), does nothing now
_ENEMY = re.compile(rf"^enemy\s+(\d+)\s+(\d+)\s+from\s+({_NUM})\s+to\s+({_NUM})$", re.I)
FORMS = "'platform ROW COL [to STOP_X]'  or  'enemy ROW COL from MIN_X to MAX_X'"

# what may not stand in a path, by map symbol
IN_THE_WAY = {"#": ("solid", "solid tiles"), "-": ("static", "static one-way tiles ('-')"), "G": ("goal", "the goal 'G'"),
              "o": ("coin", "a coin"), "P": ("spawn", "the player spawn 'P'"), "S": ("start", "a start point 'S'")}


class LevelError(ValueError):
    """The ASCII level cannot be used.  The message lists every problem found."""


SOURCE = None           # where the level comes from ("docs/my_level.txt"), set by level.py; shows up in messages


def short_errors_on_exit(hint: str) -> None:
    """An uncaught LevelError prints just its message (and `hint`) instead of a Python traceback.  Other errors are
    untouched.  level.py calls this when the level it loads is not valid."""
    previous = sys.excepthook

    def hook(kind, value, tb):
        if issubclass(kind, LevelError):
            print(f"\n{value}\n\n{hint}\n", file=sys.stderr)
        else:
            previous(kind, value, tb)

    sys.excepthook = hook


@dataclass
class Problem:
    code: str
    message: str
    cells: list = field(default_factory=list)       # (row, col), 0-based

    def __str__(self):
        return f"[{self.code}] {self.message}"


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class Platform:
    row: int
    c0: int
    c1: int
    start: float            # centre x at the start (where the M tiles are)
    stop: float             # centre x at the other end

    @property
    def half(self) -> float:
        return (self.c1 - self.c0 + 1) / 2.0

    @property
    def lo(self) -> float:
        return min(self.start, self.stop)

    @property
    def hi(self) -> float:
        return max(self.start, self.stop)

    @property
    def span(self):
        """x range the platform ever covers (its edges)."""
        return self.lo - self.half, self.hi + self.half

    def desc(self) -> str:
        return f"the moving platform at row {self.row + 1}, cols {self.c0 + 1}..{self.c1 + 1}"


@dataclass
class Enemy:
    row: int
    col: int
    lo: float               # patrol range of its centre x (for a rider: x on the platform at its start)
    hi: float
    ride: int = -1          # index of the moving platform it stands on, or -1
    run: tuple = None       # (first col, last col) of the run of '-' tiles it stands on, or None

    @property
    def x(self) -> float:
        return self.col + 0.5

    @property
    def y(self) -> float:
        return self.row + 1.0       # feet

    def span(self, half_w: float, plats=()):
        """x range its body ever covers (a rider also moves with its platform)."""
        if self.ride >= 0:
            p = plats[self.ride]
            return p.lo - p.start + self.lo - half_w, p.hi - p.start + self.hi + half_w
        return self.lo - half_w, self.hi + half_w

    def desc(self) -> str:
        return f"the enemy at row {self.row + 1}, col {self.col + 1}"


def _where(cells) -> str:
    shown = [f"row {r + 1}, col {c + 1}" for r, c in cells[:MAX_LISTED]]
    more = f" (+{len(cells) - MAX_LISTED} more)" if len(cells) > MAX_LISTED else ""
    return "; ".join(shown) + more


def split_text(text: str):
    """(map rows, path lines).  The map ends at the first blank line; after it come the optional path lines
    ('# ...' lines are comments).  Path lines are returned as (line number, text), numbered from 1."""
    lines = text.strip("\n").split("\n") if text.strip("\n") else []
    n = 0
    while n < len(lines) and lines[n].strip():
        n += 1
    spec = [(i + 1, lines[i].strip()) for i in range(n, len(lines)) if lines[i].strip() and not lines[i].strip().startswith("#")]
    return lines[:n], spec


def entities(grid: np.ndarray, spec, platform_travel: float, enemy_reach: float, err, enemy_half_w: float = 0.375):
    """Platforms and enemies of a map with their paths (defaults, then the path lines).  Problems go to err(code, msg, cells)."""
    h, w = grid.shape
    plats = []
    for r in range(h):
        c = 0
        while c < w:
            if grid[r, c] == "M":
                c0 = c
                while c + 1 < w and grid[r, c + 1] == "M":
                    c += 1
                x0 = (c0 + c + 1) / 2.0
                plats.append(Platform(r, c0, c, x0, x0 + platform_travel))
            c += 1

    def platform_at(r, c):
        return next((i for i, p in enumerate(plats) if p.row == r and p.c0 <= c <= p.c1), -1)

    enemies = []
    for r, c in np.argwhere(grid == "E"):
        r, c = int(r), int(c)
        below = grid[r + 1, c] if r + 1 < grid.shape[0] else "."
        e = Enemy(r, c, c + 0.5 - enemy_reach, c + 0.5 + enemy_reach)
        if below == "M":                                       # standing on a moving platform: walks all of it
            e.ride = platform_at(r + 1, c)
            p = plats[e.ride]
            e.lo, e.hi = p.start - p.half + enemy_half_w, p.start + p.half - enemy_half_w
        elif below == "-":                                     # standing on a static one-way platform: all of it
            c0 = c1 = c
            while c0 > 0 and grid[r + 1, c0 - 1] == "-":
                c0 -= 1
            while c1 + 1 < grid.shape[1] and grid[r + 1, c1 + 1] == "-":
                c1 += 1
            e.run = (c0, c1)
            e.lo, e.hi = c0 + enemy_half_w, c1 + 1 - enemy_half_w
        enemies.append(e)
    seen = set()
    for ln, text in spec:
        m = _PLAT.match(text)
        if m:
            r, c = int(m[1]) - 1, int(m[2]) - 1
            p = next((p for p in plats if p.row == r and p.c0 <= c <= p.c1), None)
            if p is None:
                err("path-target", f"line {ln}: there is no moving platform ('M') at row {r + 1}, col {c + 1}: {text!r}")
            elif id(p) in seen:
                err("path-duplicate", f"line {ln}: {p.desc()} already has a path line: {text!r}")
            else:
                seen.add(id(p))
                if m[3] is not None:
                    p.stop = float(m[3])
            continue
        m = _ENEMY.match(text)
        if m:
            r, c = int(m[1]) - 1, int(m[2]) - 1
            e = next((e for e in enemies if e.row == r and e.col == c), None)
            if e is None:
                err("path-target", f"line {ln}: there is no enemy ('E') at row {r + 1}, col {c + 1}: {text!r}")
            elif id(e) in seen:
                err("path-duplicate", f"line {ln}: {e.desc()} already has a path line: {text!r}")
            elif float(m[3]) > float(m[4]):
                err("path-order", f"line {ln}: 'from' must not be larger than 'to': {text!r}")
            else:
                seen.add(id(e))
                e.lo, e.hi = float(m[3]), float(m[4])
            continue
        hint = " (this looks like a map row: a blank line ends the map)" if re.fullmatch(f"[{re.escape(SYMBOLS)}]+", text) else ""
        err("path-syntax", f"line {ln}: not a path line: {text!r}{hint}.  Allowed: {FORMS}")
    return plats, enemies


def sticky_coins(grid: np.ndarray, plats):
    """[(row, col, platform index)] of the coins ('o') standing directly on a moving platform's tile: they move with it."""
    out = []
    for r, c in np.argwhere(grid == "o"):
        r, c = int(r), int(c)
        if r + 1 < grid.shape[0] and grid[r + 1, c] == "M":
            out.append((r, c, next(i for i, p in enumerate(plats) if p.row == r + 1 and p.c0 <= c <= p.c1)))
    return out


def layout(text: str, platform_travel: float, enemy_reach: float, enemy_half_w: float = 0.375):
    """(map rows, platforms, enemies, sticky coins) of a level that already passed validate()."""
    rows, spec = split_text(text)
    grid = np.array([list(r) for r in rows])
    plats, enemies = entities(grid, spec, platform_travel, enemy_reach, lambda *a: None, enemy_half_w)
    return rows, plats, enemies, sticky_coins(grid, plats)


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _cells(row: int, x_lo: float, x_hi: float, w: int):
    return [(row, c) for c in range(max(0, int(np.floor(x_lo))), min(w, int(np.ceil(x_hi))))]


def check(text: str, platform_travel: float = 6.0, enemy_reach: float = 3.0, enemy_half_w: float = 0.375,
          enemy_h: float = 0.875) -> Report:
    """Find everything wrong with a level text.  Never raises.  platform_travel / enemy_reach: paths without a path line."""
    rep = Report()
    err = lambda code, msg, cells=(): rep.errors.append(Problem(code, msg, list(cells)))         # noqa: E731
    warn = lambda code, msg, cells=(): rep.warnings.append(Problem(code, msg, list(cells)))     # noqa: E731

    rows, spec = split_text(text)
    if not rows:
        err("empty-level", "the level has no rows")
        return rep
    lens = [len(r) for r in rows]
    width = max(set(lens), key=lens.count)                      # the most common row length is taken as the right one
    if len(set(lens)) != 1:
        bad = [f"row {i + 1} has {n}" for i, n in enumerate(lens) if n != width]
        err("ragged-rows", f"all rows must have the same length ({width}); " + ", ".join(bad[:MAX_LISTED]) +
            (" ..." if len(bad) > MAX_LISTED else ""))
        return rep                                              # nothing below makes sense without a rectangle
    grid = np.array([list(r) for r in rows])
    h, w = grid.shape

    unknown = [(r, c) for r, c in np.argwhere(~np.isin(grid, list(SYMBOLS)))]
    if unknown:
        details = []
        for r, c in unknown[:MAX_LISTED]:
            ch = str(grid[r, c])
            hint = f" (did you mean '{HINTS[ch]}'?)" if ch in HINTS else ""
            details.append(f"{ch!r} at row {r + 1}, col {c + 1}{hint}")
        err("unknown-symbol", "not a level symbol: " + "; ".join(details) +
            (f" (+{len(unknown) - MAX_LISTED} more)" if len(unknown) > MAX_LISTED else "") +
            f".  Valid symbols: {' '.join(SYMBOLS)}", [tuple(map(int, u)) for u in unknown])
        grid = np.where(np.isin(grid, list(SYMBOLS)), grid, ".")    # carry on, treating them as empty

    solid, oneway = grid == "#", grid == "-"
    support = solid | oneway | (grid == "M")

    # --- exactly one player; any number of enemies and platforms ------------------------------------------------------
    spawn = [tuple(map(int, x)) for x in np.argwhere(grid == "P")]
    if len(spawn) != 1:
        now = "none" if not spawn else f"{len(spawn)}: {_where(spawn)}"
        err("spawn-count", f"needs exactly one 'P' (player spawn), found {now}", spawn)

    plats, enemies = entities(grid, spec, platform_travel, enemy_reach, err, enemy_half_w)
    sticky = sticky_coins(grid, plats)
    sticky_cells = {(r, c) for r, c, _ in sticky}
    hw = enemy_half_w

    # --- every platform path on its own: inside the map, nothing in the way ----------------------------------------------
    for p in plats:
        x0, x1 = p.span
        here = [(p.row, c) for c in range(p.c0, p.c1 + 1)]
        if p.stop == p.start:
            err("platform-no-travel", f"{p.desc()} starts and stops at x={p.start:g}: it would not move", here)
        if x0 < 0 or x1 > w:
            err("platform-sweep-map", f"{p.desc()} moves between x={p.lo:g} and x={p.hi:g}, so with its width it covers "
                f"x {x0:g}..{x1:g}, but the map spans 0..{w}", here)
            continue
        found = {}
        for r, c in _cells(p.row, x0, x1, w):
            kind = IN_THE_WAY.get(str(grid[r, c]))
            if kind:
                found.setdefault(kind, []).append((r, c))
        for (code, what), cells in found.items():
            err(f"platform-sweep-{code}", f"{p.desc()} would sweep through {what} at " + _where(cells), cells)
        if p.row > 0:                                           # a block right above the path: nobody can ride or stand there
            roof = [(r - 1, c) for r, c in _cells(p.row, x0, x1, w) if solid[r - 1, c]]
            if roof:
                warn("platform-headroom", f"{p.desc()} has solid tiles right above its path (no room to stand or ride) at "
                     + _where(roof), roof)
    for i, a in enumerate(plats):                               # nothing else on the same stretch
        for b in plats[i + 1:]:
            if a.row == b.row and _overlap(a.span, b.span):
                err("platform-overlap", f"{a.desc()} and {b.desc()} share part of their path (they would pass through "
                    "each other)", [(a.row, a.c0), (b.row, b.c0)])
        for e in enemies:
            if a.row == e.row and _overlap(a.span, e.span(enemy_half_w, plats)):
                err("platform-sweep-enemy", f"{a.desc()} shares its row and part of its path with {e.desc()}",
                    [(a.row, a.c0), (e.row, e.col)])

    # --- every enemy path on its own ------------------------------------------------------------------------------------
    for e in enemies:
        x0, x1 = e.span(hw, plats)
        if e.hi - e.lo < MIN_PATROL:
            err("enemy-no-travel", f"{e.desc()} has a path from x={e.lo:g} to x={e.hi:g}: no room to walk")
        if not e.lo <= e.x <= e.hi:
            err("enemy-start-outside", f"{e.desc()} starts at x={e.x:g}, outside its path x={e.lo:g}..{e.hi:g}",
                [(e.row, e.col)])
        if e.ride >= 0:                                         # standing on a platform: its body must stay on it
            pl = plats[e.ride]
            top, left, right = "moving", pl.start - pl.half, pl.start + pl.half
        elif e.run:
            top, left, right = "one-way", float(e.run[0]), float(e.run[1] + 1)
        else:
            top = None
        if top and (e.lo - hw < left - 1e-9 or e.hi + hw > right + 1e-9):
            at = " (where it is at the start)" if top == "moving" else ""
            err("enemy-rides-outside", f"{e.desc()} stands on a {top} platform covering x {left:g}..{right:g}{at}, but its "
                f"path x={e.lo:g}..{e.hi:g} puts its body at x {e.lo - hw:g}..{e.hi + hw:g}: off the platform", [(e.row, e.col)])
        if x0 < 0 or x1 > w:
            err("enemy-path-outside", f"{e.desc()} walks between x={e.lo:g} and x={e.hi:g}, so with its body it covers "
                f"x {x0:g}..{x1:g}, but the map spans 0..{w}", [(e.row, e.col)])
            continue
        cells = _cells(e.row, x0, x1, w)
        found = {}
        for r, c in cells:
            kind = IN_THE_WAY.get(str(grid[r, c]))
            if kind and kind[0] not in ("spawn", "start") and (kind[0] != "coin" or (r, c) not in sticky_cells):
                found.setdefault(kind, []).append((r, c))   # 'S': random starts push enemies away; stuck coins: see below
        for (code, what), bad in found.items():
            err(f"enemy-path-{code}", f"{e.desc()} would walk through {what} at " + _where(bad), bad)
        if e.ride >= 0:                                         # a coin stuck to the platform it walks on
            mine = [(r, c) for r, c, i in sticky if i == e.ride and r == e.row
                    and e.lo - hw - COIN_RADIUS < c + 0.5 < e.hi + hw + COIN_RADIUS]
            if mine:
                err("enemy-path-coin", f"{e.desc()} would walk through a coin stuck to its platform at " + _where(mine), mine)
        else:
            holes = [(e.row + 1, c) for _, c in cells
                     if not (e.row + 1 < h and support[e.row + 1, c] and grid[e.row + 1, c] != "M")]
            if holes:
                err("enemy-path-ground", f"{e.desc()} would walk over missing ground (enemies do not fall, it would float) at "
                    + _where(holes), holes)
    for i, a in enumerate(enemies):
        for b in enemies[i + 1:]:
            if a.row != b.row:
                continue
            if a.ride >= 0 and a.ride == b.ride:                # on one platform they move together: compare on the platform
                sa, sb = (a.lo - hw, a.hi + hw), (b.lo - hw, b.hi + hw)
            else:
                sa, sb = a.span(hw, plats), b.span(hw, plats)
            if _overlap(sa, sb):
                err("enemy-overlap", f"{a.desc()} and {b.desc()} share part of their path", [(a.row, a.col), (b.row, b.col)])

    # --- coins stuck to a moving platform travel along its whole path ---------------------------------------------------
    for r, c, i in sticky:
        p = plats[i]
        x0 = p.lo - p.start + c + 0.5 - COIN_RADIUS
        x1 = p.hi - p.start + c + 0.5 + COIN_RADIUS
        if x0 < 0 or x1 > w:
            continue                                            # (the platform path is reported as outside the map)
        bad = [(r, cc) for _, cc in _cells(r, x0, x1, w) if solid[r, cc]]
        if bad:
            err("coin-path-solid", f"the coin at row {r + 1}, col {c + 1} is stuck to {p.desc()} and would travel through "
                "solid tiles at " + _where(bad), bad)

    # --- an enemy that can reach the player at the real spawn (nothing pushes it away there) -------------------------
    if len(spawn) == 1:
        pr, pc = spawn[0]
        px, py = pc + 0.5, pr + 1.0
        for e in enemies:
            x0, x1 = e.span(enemy_half_w, plats)
            if x0 < px + PLAYER_HW and px - PLAYER_HW < x1 and e.y - enemy_h < py and py - PLAYER_H < e.y:
                err("enemy-hits-spawn", f"{e.desc()} can reach the player spawn at row {pr + 1}, col {pc + 1} (its path "
                    f"x {x0:g}..{x1:g} covers the spawn): the player could be hit at the start", [(e.row, e.col), (pr, pc)])
            elif max(x0 - px, px - x1, 0.0) < NEAR_SPAWN and abs(e.row - pr) <= 3:
                warn("enemy-near-spawn", f"{e.desc()} patrols within {NEAR_SPAWN:g} tiles of the player spawn at row "
                     f"{pr + 1}, col {pc + 1}", [(e.row, e.col), (pr, pc)])

    # --- warnings (the level still loads) -------------------------------------------------------------------------
    if not (grid == "G").any():
        warn("no-goal", "there is no goal 'G': the level cannot be won (an episode ends by death or after 900 steps)")

    def floor_ok(r, c):                               # solid / one-way / platform start within MAX_DROP tiles below
        for rr in range(r + 1, r + 1 + MAX_DROP):
            if rr >= h:
                return False
            if support[rr, c]:
                return True
        return False

    starts = [tuple(map(int, x)) for x in np.argwhere((grid == "P") | (grid == "S"))]
    no_floor = [(r, c) for r, c in starts if not floor_ok(r, c)]
    if no_floor:
        warn("no-ground-below", f"no ground within {MAX_DROP} tiles below a start point (it falls out of the map): " +
             _where(no_floor), no_floor)

    # completely walled-in goals / coins / starts (flood fill through non-solid cells from the spawn)
    if len(spawn) == 1:
        reach = _reachable(solid, spawn[0])
        for ch, what in (("o", "coin"), ("G", "goal"), ("S", "start point")):
            lost = [tuple(map(int, x)) for x in np.argwhere(grid == ch) if not reach[x[0], x[1]]]
            if lost:
                warn("walled-in", f"a {what} is completely walled in by solid tiles, nothing can reach it: " +
                     _where(lost), lost)
    return rep


def _reachable(solid: np.ndarray, start) -> np.ndarray:
    """Cells connected to `start` through non-solid cells (4-neighbours).  The space above the map counts as open
    (a jump can go over a wall that ends below the top row); the sides and the bottom do not."""
    h, w = solid.shape
    top = 3
    seen = np.zeros((h + top, w), bool)
    q = deque([(start[0] + top, start[1])])
    seen[start[0] + top, start[1]] = True
    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h + top and 0 <= cc < w and not seen[rr, cc] and (rr < top or not solid[rr - top, cc]):
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen[top:]


def validate(text: str, *limits, source: str | None = None) -> Report:
    """Raise LevelError (listing every error) if the level is unusable, log the warnings, return the report.
    source: where the text comes from (a file name), named in the messages; default: SOURCE."""
    rep = check(text, *limits)
    source = source or SOURCE
    if rep.errors:
        n = len(rep.errors)
        where = f" in {source}" if source else ""
        raise LevelError(f"the level{where} has {n} problem{'s' if n > 1 else ''} (rows and columns count from 1, "
                         "'line' = line of the level text):\n" + "\n".join(f"  {e}" for e in rep.errors))
    for w in rep.warnings:
        log.warning("level warning%s %s", f" ({source})" if source else "", w)
    return rep


def level_files(paths) -> list[str]:
    """Level files named on a command line: a folder stands for all its *.txt files (sorted by name)."""
    import os
    out = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".txt"))
            if not found:
                raise LevelError(f"no level files (*.txt) in the folder {p}")
            out += found
        else:
            out.append(p)
    return out


def main(argv=None) -> int:
    """Check the level in level.py, or every file / folder (all its *.txt) given.  Exit code 1 if any has an error."""
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        try:
            files = level_files(argv)
        except LevelError as e:
            print("ERROR  ", e)
            return 1
        texts = []
        for f in files:                            # a file is checked on its own: no need for a valid level.py
            try:
                with open(f) as fh:
                    texts.append((f, fh.read()))
            except OSError as e:
                texts.append((f, None))
                print(f"ERROR   cannot read {f}: {e.strerror}")
    else:
        from platformer_ai import level as L  # the level in level.py (importing it already validates it)
        texts = [(None, L.LEVEL)]
    bad = 0
    for name, text in texts:
        if text is None:
            bad += 1
            continue
        rep = check(text)                          # the limits are check()'s defaults, the same as in level.py
        if name and len(texts) > 1:
            print(f"==== {name}")
        if len(texts) == 1:
            print(text.strip("\n"))
            print()
        for e in rep.errors:
            print("ERROR  ", e)
        for w in rep.warnings:
            print("WARNING", w)
        print(f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
        bad += bool(rep.errors)
    if len(texts) > 1:
        print(f"{len(texts)} levels, {bad} with errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
