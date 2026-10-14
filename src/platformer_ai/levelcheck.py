"""Checks the ASCII level before the game is built from it.

    errors    the level cannot be built (raised as LevelError, all of them at once, with row / column)
    warnings  the level loads but something looks wrong (logged on the "platformer_ai.level" logger)

Run it on the level in level.py, or on a text file that holds only the ASCII rows:

    uv run python -m platformer_ai.levelcheck [my_level.txt]

Rows and columns in the messages count from 1, like the line numbers in an editor.
What it can NOT tell you is whether a level is *beatable*: it finds goals and coins that are completely walled in,
but not ones that are merely too high or too far for the jump (apex 2.7 tiles).  Train an agent or use a bot for that.
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("platformer_ai.level")

SYMBOLS = ".#-oGPESM"
HINTS = {"p": "P", "e": "E", "m": "M", "s": "S", "g": "G", "0": "o", "O": "o", "=": "-", "_": "-", " ": "."}
MAX_DROP = 4            # a start may be this many tiles above its floor (the built-in spawn drops 2)
MAX_LISTED = 5          # cells named in one message


class LevelError(ValueError):
    """The ASCII level cannot be used.  The message lists every problem found."""


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


def _where(cells) -> str:
    shown = [f"row {r + 1}, col {c + 1}" for r, c in cells[:MAX_LISTED]]
    more = f" (+{len(cells) - MAX_LISTED} more)" if len(cells) > MAX_LISTED else ""
    return "; ".join(shown) + more


def patrol_limits(solid: np.ndarray, ex: float, ey: float, ray_len: float, ray_h: float):
    """x range the enemy patrols: from the wall on its left to the wall on its right, at sensor height."""
    h, w = solid.shape
    row = int(np.floor(ey - ray_h))
    c = int(ex)
    while c >= 0 and not solid[row, c]:
        c -= 1
    left_face = c + 1
    c = int(ex)
    while c < w and not solid[row, c]:
        c += 1
    return left_face + ray_len, c - ray_len


def check(text: str, platform_travel: float = 6.0, enemy_ray_len: float = 0.5625, enemy_ray_h: float = 0.4375,
          enemy_half_w: float = 0.375) -> Report:
    """Find everything wrong with a level text.  Never raises."""
    rep = Report()
    err = lambda code, msg, cells=(): rep.errors.append(Problem(code, msg, list(cells)))
    warn = lambda code, msg, cells=(): rep.warnings.append(Problem(code, msg, list(cells)))

    rows = text.strip("\n").split("\n") if text.strip("\n") else []
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

    # --- exactly one player and one enemy -------------------------------------------------------------------------
    cells = {}
    for ch, name in (("P", "player spawn"), ("E", "enemy")):
        found = [tuple(map(int, x)) for x in np.argwhere(grid == ch)]
        cells[ch] = found
        if len(found) != 1:
            now = "none" if not found else f"{len(found)}: {_where(found)}"
            err(f"{name.split()[-1]}-count", f"needs exactly one '{ch}' ({name}), found {now}", found)

    # --- the moving platform: one row, no gaps, room to sweep ---------------------------------------------------------
    m_cells = [tuple(map(int, x)) for x in np.argwhere(grid == "M")]
    m_ok = False
    if not m_cells:
        err("platform-missing", "needs a moving platform: put 'M' tiles next to each other in one row (the original has 2)")
    elif len({r for r, _ in m_cells}) != 1:
        err("platform-rows", "all 'M' tiles must be in a single row; found " + _where(m_cells), m_cells)
    else:
        m_row = m_cells[0][0]
        cols = sorted(c for _, c in m_cells)
        if cols != list(range(cols[0], cols[-1] + 1)):
            err("platform-gap", f"the 'M' tiles in row {m_row + 1} must be one block with no gap "
                f"(columns {cols[0] + 1}..{cols[-1] + 1})", m_cells)
        else:
            m_ok = True
            last = int(np.ceil(cols[-1] + 1 + platform_travel)) - 1        # right-most column the platform covers
            if last >= w:
                err("platform-sweep-map", f"the moving platform sweeps {platform_travel:g} tiles to the right, up to "
                    f"column {last + 1}, but the map is only {w} wide", m_cells)
            else:
                blocked = [(m_row, c) for c in range(cols[0], last + 1) if solid[m_row, c]]
                if blocked:
                    err("platform-sweep-solid", "the moving platform would sweep through solid tiles at " +
                        _where(blocked), blocked)

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

    if len(cells["E"]) == 1:
        er, ec = cells["E"][0]
        if not (er + 1 < h and (solid[er + 1, ec] or oneway[er + 1, ec])):
            warn("enemy-floating", f"the enemy at row {er + 1}, col {ec + 1} has no ground directly below it "
                 "(enemies do not fall, it would float)", [(er, ec)])
        lo, hi = patrol_limits(solid, ec + 0.5, er + 1.0, enemy_ray_len, enemy_ray_h)
        if hi - lo < 2 * enemy_half_w:
            warn("enemy-boxed-in", f"walls leave the enemy at row {er + 1}, col {ec + 1} (almost) no room to patrol",
                 [(er, ec)])
        elif er + 1 < h:
            c_from, c_to = int(np.floor(lo)), int(np.floor(hi))
            holes = [(er + 1, c) for c in range(c_from, min(c_to, w - 1) + 1)
                     if not (solid[er + 1, c] or oneway[er + 1, c])]
            if holes:
                warn("enemy-patrol-ground", "the enemy patrols over missing ground (it keeps walking in mid-air) at " +
                     _where(holes), holes)

    # completely walled-in goals / coins / starts (flood fill through non-solid cells from the spawn)
    if len(cells["P"]) == 1:
        reach = _reachable(solid, cells["P"][0])
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


def validate(text: str, *limits) -> Report:
    """Raise LevelError (listing every error) if the level is unusable, log the warnings, return the report."""
    rep = check(text, *limits)
    if rep.errors:
        n = len(rep.errors)
        raise LevelError(f"the level has {n} problem{'s' if n > 1 else ''} (rows and columns count from 1):\n" +
                         "\n".join(f"  {e}" for e in rep.errors))
    for w in rep.warnings:
        log.warning("level warning %s", w)
    return rep


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from platformer_ai import level as L          # importing it already validates the built-in level
    text = open(argv[0]).read() if argv else L.LEVEL
    rep = check(text, L.PLATFORM_TRAVEL, L.ENEMY_RAY_LEN, L.ENEMY_RAY_H, L.ENEMY_HALF_W)
    print(text.strip("\n"))
    print()
    for e in rep.errors:
        print("ERROR  ", e)
    for w in rep.warnings:
        print("WARNING", w)
    print(f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
